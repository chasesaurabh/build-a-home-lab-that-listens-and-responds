from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
from typing import Callable, Mapping, Any

from ..contracts import ActionRegistry, ContractError, IntentEnvelope


ALLOWED_SERVICES = {"media-indexer", "voice-service"}
ALLOWED_DIAGNOSTICS = {"healthcheck", "queue_snapshot"}


@dataclass(frozen=True)
class Session:
    session_id: str
    role: str
    expires_at: datetime


@dataclass
class PendingOperation:
    operation_digest: str
    session_id: str
    intent: str
    arguments: Mapping[str, str]
    idempotency_key: str
    expires_at: datetime
    used: bool = False


@dataclass(frozen=True)
class ToolResult:
    intent: str
    status: str
    payload: Mapping[str, Any]


class ControlledAssistant:
    """Policy boundary for two closed, model-independent assistant tools."""

    def __init__(
        self,
        status_provider: Callable[[str], Mapping[str, Any]],
        diagnostic_provider: Callable[[str, str], Mapping[str, Any]],
    ) -> None:
        self.status_provider = status_provider
        self.diagnostic_provider = diagnostic_provider
        self.actions = ActionRegistry()

    @staticmethod
    def _session(session: Session, now: datetime) -> None:
        if session.expires_at <= now:
            raise ContractError("SESSION_EXPIRED", "The trusted session has expired.")
        if session.role not in {"viewer", "operator", "administrator"}:
            raise ContractError("SESSION_ROLE_INVALID", "The session role is not recognized.")

    @staticmethod
    def _service(arguments: Mapping[str, Any]) -> str:
        if set(arguments) - {"service", "diagnostic"}:
            raise ContractError("TOOL_ARGUMENT_UNKNOWN", "Tool arguments contain an unknown field.")
        service = arguments.get("service")
        if service not in ALLOWED_SERVICES:
            raise ContractError("TOOL_TARGET_NOT_ALLOWED", "The requested service is not allowlisted.")
        return str(service)

    def read_status(self, intent: IntentEnvelope, session: Session, now: datetime) -> ToolResult:
        self._session(session, now)
        if intent.disposition != "accept" or intent.intent != "lab.status.read":
            raise ContractError("TOOL_INTENT_INVALID", "A validated status intent is required.")
        if set(intent.arguments) != {"service"}:
            raise ContractError("TOOL_ARGUMENT_INVALID", "Status requires exactly one service argument.")
        service = self._service(intent.arguments)
        payload = dict(self.status_provider(service))
        return ToolResult(intent.intent, "completed", payload)

    def prepare_diagnostic(
        self,
        intent: IntentEnvelope,
        session: Session,
        now: datetime,
        idempotency_key: str,
        *,
        confirmation_seconds: int = 60,
    ) -> PendingOperation:
        self._session(session, now)
        if session.role not in {"operator", "administrator"}:
            raise ContractError("TOOL_PERMISSION_DENIED", "The session cannot run diagnostics.")
        if intent.disposition != "accept" or intent.intent != "lab.diagnostic.run":
            raise ContractError("TOOL_INTENT_INVALID", "A validated diagnostic intent is required.")
        if set(intent.arguments) != {"service", "diagnostic"}:
            raise ContractError("TOOL_ARGUMENT_INVALID", "Diagnostic requires service and diagnostic arguments.")
        service = self._service(intent.arguments)
        diagnostic = intent.arguments.get("diagnostic")
        if diagnostic not in ALLOWED_DIAGNOSTICS:
            raise ContractError("TOOL_DIAGNOSTIC_NOT_ALLOWED", "The requested diagnostic is not allowlisted.")
        arguments = {"service": service, "diagnostic": str(diagnostic)}
        canonical = json.dumps(
            {"session": session.session_id, "intent": intent.intent, "arguments": arguments},
            sort_keys=True,
            separators=(",", ":"),
        )
        return PendingOperation(
            operation_digest=sha256(canonical.encode("utf-8")).hexdigest(),
            session_id=session.session_id,
            intent=intent.intent,
            arguments=arguments,
            idempotency_key=idempotency_key,
            expires_at=now + timedelta(seconds=confirmation_seconds),
        )

    def confirm_and_run(
        self,
        pending: PendingOperation,
        session: Session,
        now: datetime,
        operation_digest: str,
    ) -> ToolResult:
        self._session(session, now)
        if pending.used:
            raise ContractError("CONFIRMATION_USED", "The confirmation has already been used.")
        if pending.expires_at <= now:
            raise ContractError("CONFIRMATION_EXPIRED", "The confirmation has expired.")
        if pending.session_id != session.session_id:
            raise ContractError("CONFIRMATION_SESSION_MISMATCH", "The confirmation belongs to another session.")
        if pending.operation_digest != operation_digest:
            raise ContractError("CONFIRMATION_OPERATION_MISMATCH", "The confirmation does not match the pending operation.")
        pending.used = True
        service = pending.arguments["service"]
        diagnostic = pending.arguments["diagnostic"]
        observed = dict(self.diagnostic_provider(service, diagnostic))
        saved = self.actions.remember(pending.idempotency_key, pending.intent, pending.arguments, observed)
        return ToolResult(pending.intent, "completed", saved)
