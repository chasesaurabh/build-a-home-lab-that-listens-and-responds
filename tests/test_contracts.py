from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.contracts import (  # noqa: E402
    ActionRegistry,
    AudioEnvelope,
    CancellableJob,
    CancellationState,
    ContractError,
    IntentEnvelope,
    SpeechRequest,
)


CONTENT = b"deterministic fixture bytes, not recorded audio"


def valid_audio() -> dict:
    return {
        "request_id": "req_01JVOICE1234",
        "source_kind": "licensed_fixture",
        "media_type": "audio/wav",
        "sample_rate_hz": 16000,
        "channels": 1,
        "captured_at": "2026-09-22T18:14:03Z",
        "consent_basis": "licensed_fixture",
        "retention_class": "ephemeral",
        "content_sha256": sha256(CONTENT).hexdigest(),
    }


def valid_intent() -> dict:
    return {
        "schema_version": "intent.v1",
        "request_id": "req_01JVOICE1234",
        "disposition": "accept",
        "intent": "lab.status.read",
        "arguments": {"service": "media-indexer"},
        "evidence": [{"segment_id": "seg_004", "start_ms": 2180, "end_ms": 4610}],
    }


class AudioEnvelopeTests(unittest.TestCase):
    def test_valid_envelope_and_content(self) -> None:
        envelope = AudioEnvelope.from_dict(valid_audio())
        envelope.validate_content(CONTENT)
        self.assertEqual(envelope.sample_rate_hz, 16000)

    def test_negative_fields_fail_for_intended_reason(self) -> None:
        cases = {
            "unsupported media": ("media_type", "audio/mpeg", "AUDIO_FORMAT_UNSUPPORTED"),
            "impossible channels": ("channels", 0, "CHANNEL_COUNT_INVALID"),
            "missing consent": ("consent_basis", "", "CONSENT_BASIS_REQUIRED"),
        }
        for name, (field, value, code) in cases.items():
            with self.subTest(name=name):
                data = valid_audio()
                data[field] = value
                with self.assertRaises(ContractError) as caught:
                    AudioEnvelope.from_dict(data)
                self.assertEqual(caught.exception.code, code)

    def test_unknown_field_rejected(self) -> None:
        data = valid_audio()
        data["speaker_name"] = "not permitted"
        with self.assertRaises(ContractError) as caught:
            AudioEnvelope.from_dict(data)
        self.assertEqual(caught.exception.code, "UNKNOWN_FIELD")

    def test_content_substitution_rejected(self) -> None:
        envelope = AudioEnvelope.from_dict(valid_audio())
        with self.assertRaises(ContractError) as caught:
            envelope.validate_content(b"different")
        self.assertEqual(caught.exception.code, "CONTENT_DIGEST_MISMATCH")


class IntentTests(unittest.TestCase):
    def test_allowlisted_intent(self) -> None:
        intent = IntentEnvelope.from_dict(valid_intent())
        self.assertEqual(intent.intent, "lab.status.read")

    def test_unapproved_intent_rejected(self) -> None:
        data = valid_intent()
        data["intent"] = "shell.execute"
        with self.assertRaises(ContractError) as caught:
            IntentEnvelope.from_dict(data)
        self.assertEqual(caught.exception.code, "INTENT_NOT_ALLOWED")

    def test_abstention_names_no_intent(self) -> None:
        data = valid_intent()
        data.update({"disposition": "abstain", "intent": None, "arguments": {}})
        self.assertIsNone(IntentEnvelope.from_dict(data).intent)


class ActionTests(unittest.TestCase):
    def test_duplicate_returns_earlier_result(self) -> None:
        registry = ActionRegistry()
        first = registry.remember("idem_01JVOICE1234", "lab.status.read", {"service": "index"}, {"status": "completed"})
        second = registry.remember("idem_01JVOICE1234", "lab.status.read", {"service": "index"}, {"status": "different"})
        self.assertEqual(first, second)

    def test_conflicting_reuse_rejected(self) -> None:
        registry = ActionRegistry()
        registry.remember("idem_01JVOICE1234", "lab.status.read", {"service": "index"}, {"status": "completed"})
        with self.assertRaises(ContractError) as caught:
            registry.remember("idem_01JVOICE1234", "lab.status.read", {"service": "backup"}, {"status": "completed"})
        self.assertEqual(caught.exception.code, "IDEMPOTENCY_CONFLICT")


class SpeechAndCancellationTests(unittest.TestCase):
    def test_speech_length_bound(self) -> None:
        data = {
            "request_id": "req_01JVOICE1234",
            "text": "Status is ready.",
            "language": "en-US",
            "voice_revision": "voice-revision-sha256",
            "output_format": "audio/wav",
            "disclosure": "private_ui",
        }
        self.assertEqual(SpeechRequest.from_dict(data).language, "en-US")
        too_long = deepcopy(data)
        too_long["text"] = "x" * 2001
        with self.assertRaises(ContractError) as caught:
            SpeechRequest.from_dict(too_long)
        self.assertEqual(caught.exception.code, "SPEECH_TEXT_LENGTH_INVALID")

    def test_cancel_before_work(self) -> None:
        job = CancellableJob()
        job.cancel()
        self.assertEqual(job.state, CancellationState.CANCELLED)
        self.assertEqual(job.queued_chunks, [])

    def test_cancel_during_work(self) -> None:
        job = CancellableJob()
        job.begin()
        job.cancel()
        self.assertEqual(job.state, CancellationState.CANCELLED)

    def test_cancel_during_playback_flushes_chunks(self) -> None:
        job = CancellableJob()
        job.begin()
        job.play([b"one", b"two"])
        job.cancel()
        self.assertEqual(job.state, CancellationState.CANCELLED)
        self.assertEqual(job.queued_chunks, [])


if __name__ == "__main__":
    unittest.main()
