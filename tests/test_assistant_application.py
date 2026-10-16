from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.applications.assistant import ControlledAssistant, Session  # noqa: E402
from home_lab_voice.contracts import ContractError, IntentEnvelope  # noqa: E402


NOW = datetime(2026, 9, 22, 20, 0, tzinfo=timezone.utc)


def session(role: str = "operator", *, session_id: str = "session-local-01", seconds: int = 300) -> Session:
    return Session(session_id, role, NOW + timedelta(seconds=seconds))


def intent(name: str, arguments: dict[str, str]) -> IntentEnvelope:
    return IntentEnvelope.from_dict({
        "schema_version": "intent.v1",
        "request_id": "req_01JASSIST99",
        "disposition": "accept",
        "intent": name,
        "arguments": arguments,
        "evidence": [{"segment_id": "seg_001"}],
    })


class AssistantApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.status_calls: list[str] = []
        self.diagnostic_calls: list[tuple[str, str]] = []

        def status_provider(service: str) -> dict[str, str]:
            self.status_calls.append(service)
            return {"service": service, "state": "healthy"}

        def diagnostic_provider(service: str, diagnostic: str) -> dict[str, str]:
            self.diagnostic_calls.append((service, diagnostic))
            return {"service": service, "diagnostic": diagnostic, "state": "completed"}

        self.assistant = ControlledAssistant(status_provider, diagnostic_provider)

    def test_viewer_can_read_allowlisted_status(self) -> None:
        result = self.assistant.read_status(
            intent("lab.status.read", {"service": "voice-service"}), session("viewer"), NOW
        )
        self.assertEqual(result.payload["state"], "healthy")
        self.assertEqual(self.status_calls, ["voice-service"])

    def test_unknown_target_and_arguments_are_rejected(self) -> None:
        cases = (
            ({"service": "database"}, "TOOL_TARGET_NOT_ALLOWED"),
            ({"service": "voice-service", "detail": "secrets"}, "TOOL_ARGUMENT_INVALID"),
        )
        for arguments, code in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ContractError) as caught:
                self.assistant.read_status(intent("lab.status.read", arguments), session("viewer"), NOW)
            self.assertEqual(caught.exception.code, code)

    def test_expired_session_cannot_read_or_prepare(self) -> None:
        expired = session(seconds=0)
        with self.assertRaises(ContractError) as status:
            self.assistant.read_status(intent("lab.status.read", {"service": "voice-service"}), expired, NOW)
        self.assertEqual(status.exception.code, "SESSION_EXPIRED")
        with self.assertRaises(ContractError) as diagnostic:
            self.assistant.prepare_diagnostic(
                intent("lab.diagnostic.run", {"service": "voice-service", "diagnostic": "healthcheck"}),
                expired,
                NOW,
                "idem_01JASSIST99",
            )
        self.assertEqual(diagnostic.exception.code, "SESSION_EXPIRED")

    def test_viewer_cannot_prepare_diagnostic(self) -> None:
        with self.assertRaises(ContractError) as caught:
            self.assistant.prepare_diagnostic(
                intent("lab.diagnostic.run", {"service": "voice-service", "diagnostic": "healthcheck"}),
                session("viewer"),
                NOW,
                "idem_01JASSIST98",
            )
        self.assertEqual(caught.exception.code, "TOOL_PERMISSION_DENIED")

    def test_diagnostic_requires_allowlisted_kind(self) -> None:
        with self.assertRaises(ContractError) as caught:
            self.assistant.prepare_diagnostic(
                intent("lab.diagnostic.run", {"service": "voice-service", "diagnostic": "shell"}),
                session(),
                NOW,
                "idem_01JASSIST97",
            )
        self.assertEqual(caught.exception.code, "TOOL_DIAGNOSTIC_NOT_ALLOWED")

    def test_confirmation_binds_session_and_operation(self) -> None:
        pending = self.assistant.prepare_diagnostic(
            intent("lab.diagnostic.run", {"service": "voice-service", "diagnostic": "healthcheck"}),
            session(),
            NOW,
            "idem_01JASSIST96",
        )
        with self.assertRaises(ContractError) as wrong_digest:
            self.assistant.confirm_and_run(pending, session(), NOW, "0" * 64)
        self.assertEqual(wrong_digest.exception.code, "CONFIRMATION_OPERATION_MISMATCH")
        with self.assertRaises(ContractError) as wrong_session:
            self.assistant.confirm_and_run(
                pending, session(session_id="session-local-02"), NOW, pending.operation_digest
            )
        self.assertEqual(wrong_session.exception.code, "CONFIRMATION_SESSION_MISMATCH")
        self.assertEqual(self.diagnostic_calls, [])

    def test_expired_confirmation_does_not_run(self) -> None:
        pending = self.assistant.prepare_diagnostic(
            intent("lab.diagnostic.run", {"service": "voice-service", "diagnostic": "healthcheck"}),
            session(),
            NOW,
            "idem_01JASSIST95",
            confirmation_seconds=1,
        )
        with self.assertRaises(ContractError) as caught:
            self.assistant.confirm_and_run(pending, session(), NOW + timedelta(seconds=2), pending.operation_digest)
        self.assertEqual(caught.exception.code, "CONFIRMATION_EXPIRED")
        self.assertEqual(self.diagnostic_calls, [])

    def test_confirmed_diagnostic_runs_once(self) -> None:
        pending = self.assistant.prepare_diagnostic(
            intent("lab.diagnostic.run", {"service": "media-indexer", "diagnostic": "queue_snapshot"}),
            session(),
            NOW,
            "idem_01JASSIST94",
        )
        result = self.assistant.confirm_and_run(pending, session(), NOW, pending.operation_digest)
        self.assertEqual(result.payload["state"], "completed")
        self.assertEqual(self.diagnostic_calls, [("media-indexer", "queue_snapshot")])
        with self.assertRaises(ContractError) as caught:
            self.assistant.confirm_and_run(pending, session(), NOW, pending.operation_digest)
        self.assertEqual(caught.exception.code, "CONFIRMATION_USED")
        self.assertEqual(len(self.diagnostic_calls), 1)


if __name__ == "__main__":
    unittest.main()
