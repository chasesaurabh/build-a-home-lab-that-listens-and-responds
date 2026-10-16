from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
import re
from typing import Any, Mapping


REQUEST_ID = re.compile(r"^req_[A-Za-z0-9]{8,64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_MEDIA_TYPES = {"audio/wav", "audio/flac"}
ALLOWED_SOURCES = {"microphone", "permitted_file", "licensed_fixture"}
ALLOWED_CONSENT = {
    "self_recorded",
    "household_opt_in",
    "licensed_fixture",
    "user_supplied_with_permission",
}
ALLOWED_RETENTION = {"ephemeral", "transcript_30d", "project_archive"}
ALLOWED_INTENTS = {"lab.status.read", "lab.diagnostic.run"}


class ContractError(ValueError):
    """A stable, safe validation failure suitable for a client response."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.safe_message = message


def _only_fields(data: Mapping[str, Any], allowed: set[str]) -> None:
    extras = sorted(set(data) - allowed)
    if extras:
        raise ContractError("UNKNOWN_FIELD", f"Unknown fields: {', '.join(extras)}")


def _require(data: Mapping[str, Any], names: set[str]) -> None:
    missing = sorted(name for name in names if name not in data)
    if missing:
        raise ContractError("MISSING_FIELD", f"Missing fields: {', '.join(missing)}")


def _offset_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as error:
        raise ContractError("TIMESTAMP_INVALID", "Timestamp must use ISO 8601 format.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError("TIMESTAMP_OFFSET_REQUIRED", "Timestamp must include an offset.")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class AudioEnvelope:
    request_id: str
    source_kind: str
    media_type: str
    sample_rate_hz: int
    channels: int
    captured_at: datetime
    consent_basis: str
    retention_class: str
    content_sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AudioEnvelope":
        allowed = {
            "request_id",
            "source_kind",
            "media_type",
            "sample_rate_hz",
            "channels",
            "captured_at",
            "consent_basis",
            "retention_class",
            "content_sha256",
        }
        _only_fields(data, allowed)
        _require(data, allowed)
        if not REQUEST_ID.fullmatch(str(data["request_id"])):
            raise ContractError("REQUEST_ID_INVALID", "Request identifier is invalid.")
        if data["source_kind"] not in ALLOWED_SOURCES:
            raise ContractError("SOURCE_KIND_UNSUPPORTED", "Audio source kind is not enabled.")
        if data["media_type"] not in ALLOWED_MEDIA_TYPES:
            raise ContractError("AUDIO_FORMAT_UNSUPPORTED", "The submitted audio format is not enabled.")
        if not isinstance(data["sample_rate_hz"], int) or not 8_000 <= data["sample_rate_hz"] <= 192_000:
            raise ContractError("SAMPLE_RATE_INVALID", "Sample rate is outside the supported range.")
        if not isinstance(data["channels"], int) or not 1 <= data["channels"] <= 8:
            raise ContractError("CHANNEL_COUNT_INVALID", "Channel count is outside the supported range.")
        if data["consent_basis"] not in ALLOWED_CONSENT:
            raise ContractError("CONSENT_BASIS_REQUIRED", "An approved consent basis is required.")
        if data["retention_class"] not in ALLOWED_RETENTION:
            raise ContractError("RETENTION_CLASS_INVALID", "Retention class is not configured.")
        digest = str(data["content_sha256"])
        if not SHA256.fullmatch(digest):
            raise ContractError("CONTENT_DIGEST_INVALID", "Content digest must be a lowercase SHA-256 value.")
        return cls(
            request_id=str(data["request_id"]),
            source_kind=str(data["source_kind"]),
            media_type=str(data["media_type"]),
            sample_rate_hz=data["sample_rate_hz"],
            channels=data["channels"],
            captured_at=_offset_datetime(str(data["captured_at"])),
            consent_basis=str(data["consent_basis"]),
            retention_class=str(data["retention_class"]),
            content_sha256=digest,
        )

    def validate_content(self, content: bytes) -> None:
        observed = hashlib.sha256(content).hexdigest()
        if observed != self.content_sha256:
            raise ContractError("CONTENT_DIGEST_MISMATCH", "Submitted content does not match its envelope.")


@dataclass(frozen=True)
class IntentEnvelope:
    request_id: str
    disposition: str
    intent: str | None
    arguments: Mapping[str, Any]
    evidence: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntentEnvelope":
        allowed = {"schema_version", "request_id", "disposition", "intent", "arguments", "evidence"}
        _only_fields(data, allowed)
        _require(data, {"schema_version", "request_id", "disposition", "arguments", "evidence"})
        if data["schema_version"] != "intent.v1":
            raise ContractError("SCHEMA_VERSION_UNSUPPORTED", "Intent schema version is not supported.")
        if not REQUEST_ID.fullmatch(str(data["request_id"])):
            raise ContractError("REQUEST_ID_INVALID", "Request identifier is invalid.")
        disposition = str(data["disposition"])
        if disposition not in {"accept", "clarify", "abstain"}:
            raise ContractError("DISPOSITION_INVALID", "Intent disposition is invalid.")
        intent = data.get("intent")
        if disposition == "accept" and intent not in ALLOWED_INTENTS:
            raise ContractError("INTENT_NOT_ALLOWED", "Intent is not allowlisted.")
        if disposition != "accept" and intent is not None:
            raise ContractError("INTENT_WITHOUT_ACCEPT", "A nonaccepted result must not name an intent.")
        if not isinstance(data["arguments"], Mapping):
            raise ContractError("ARGUMENTS_INVALID", "Intent arguments must be an object.")
        evidence = data["evidence"]
        if not isinstance(evidence, list):
            raise ContractError("EVIDENCE_INVALID", "Intent evidence must be a list.")
        return cls(
            request_id=str(data["request_id"]),
            disposition=disposition,
            intent=str(intent) if intent is not None else None,
            arguments=dict(data["arguments"]),
            evidence=tuple(dict(item) for item in evidence),
        )


class ActionRegistry:
    """Detect duplicate actions and conflicting reuse of idempotency keys."""

    def __init__(self) -> None:
        self._records: dict[str, tuple[str, Mapping[str, Any], Mapping[str, Any]]] = {}

    @staticmethod
    def _fingerprint(intent: str, arguments: Mapping[str, Any]) -> str:
        canonical = json.dumps({"intent": intent, "arguments": arguments}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def remember(self, key: str, intent: str, arguments: Mapping[str, Any], result: Mapping[str, Any]) -> Mapping[str, Any]:
        if not re.fullmatch(r"idem_[A-Za-z0-9]{8,64}", key):
            raise ContractError("IDEMPOTENCY_KEY_INVALID", "Idempotency key is invalid.")
        fingerprint = self._fingerprint(intent, arguments)
        if key in self._records:
            earlier_fingerprint, _, earlier_result = self._records[key]
            if earlier_fingerprint != fingerprint:
                raise ContractError("IDEMPOTENCY_CONFLICT", "Idempotency key was reused for a different action.")
            return earlier_result
        saved_result = dict(result)
        self._records[key] = (fingerprint, dict(arguments), saved_result)
        return saved_result


@dataclass(frozen=True)
class SpeechRequest:
    request_id: str
    text: str
    language: str
    voice_revision: str
    output_format: str
    disclosure: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], max_chars: int = 2_000) -> "SpeechRequest":
        allowed = {"request_id", "text", "language", "voice_revision", "output_format", "disclosure"}
        _only_fields(data, allowed)
        _require(data, allowed)
        text = str(data["text"])
        if not text.strip() or len(text) > max_chars:
            raise ContractError("SPEECH_TEXT_LENGTH_INVALID", "Speech text is empty or over the configured limit.")
        if data["output_format"] not in {"audio/wav", "audio/flac"}:
            raise ContractError("SPEECH_FORMAT_UNSUPPORTED", "Speech output format is not enabled.")
        if data["disclosure"] not in {"private_ui", "announce_synthetic"}:
            raise ContractError("DISCLOSURE_POLICY_INVALID", "Speech disclosure policy is not configured.")
        if not REQUEST_ID.fullmatch(str(data["request_id"])):
            raise ContractError("REQUEST_ID_INVALID", "Request identifier is invalid.")
        if not str(data["voice_revision"]).strip():
            raise ContractError("VOICE_REVISION_REQUIRED", "An immutable voice revision is required.")
        return cls(**{name: str(data[name]) for name in allowed})


class CancellationState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PLAYING = "playing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class CancellableJob:
    state: CancellationState = CancellationState.QUEUED
    queued_chunks: list[bytes] = field(default_factory=list)

    def begin(self) -> None:
        if self.state is not CancellationState.QUEUED:
            raise ContractError("STATE_TRANSITION_INVALID", "Job cannot begin from its current state.")
        self.state = CancellationState.RUNNING

    def play(self, chunks: list[bytes]) -> None:
        if self.state is not CancellationState.RUNNING:
            raise ContractError("STATE_TRANSITION_INVALID", "Job cannot play from its current state.")
        self.queued_chunks.extend(chunks)
        self.state = CancellationState.PLAYING

    def cancel(self) -> None:
        if self.state is CancellationState.COMPLETED:
            raise ContractError("JOB_ALREADY_COMPLETED", "Completed job cannot be cancelled.")
        self.queued_chunks.clear()
        self.state = CancellationState.CANCELLED

    def complete(self) -> None:
        if self.state not in {CancellationState.RUNNING, CancellationState.PLAYING}:
            raise ContractError("STATE_TRANSITION_INVALID", "Job cannot complete from its current state.")
        self.queued_chunks.clear()
        self.state = CancellationState.COMPLETED
