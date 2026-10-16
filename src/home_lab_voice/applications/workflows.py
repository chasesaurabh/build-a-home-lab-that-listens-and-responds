from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any
import wave

from .archive import ManagedArchive
from .dictation import IDEMPOTENCY_KEY, ManagedTextDestination, UndoToken
from .narration import plan_narration
from ..contracts import AudioEnvelope, ContractError


def _digest_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("WORKFLOW_RECORD_INVALID", "Managed workflow record is unavailable or invalid.") from error
    if not isinstance(value, dict):
        raise ContractError("WORKFLOW_RECORD_INVALID", "Managed workflow record must be a JSON object.")
    return value


def transcribe_speech(source: Path, vad_runtime: Any, asr_runtime: Any) -> tuple[Any, Any]:
    vad = vad_runtime.analyze(source)
    if not vad.regions:
        raise ContractError("WORKFLOW_NO_SPEECH", "No speech was detected; no transcript or action was created.")
    transcript = asr_runtime.transcribe(source)
    if not transcript.text.strip():
        raise ContractError("WORKFLOW_EMPTY_TRANSCRIPT", "Speech recognition returned no reviewable text.")
    return vad, transcript


@dataclass(frozen=True)
class ArchivedTranscript:
    source_digest: str
    managed_audio: Path
    transcript_record: Path
    text: str
    text_sha256: str


class SearchableAudioArchive:
    """Managed audio plus local JSON transcript records with search and deletion."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.archive = ManagedArchive(self.root)
        self.records = self.root / "records"
        self.records.mkdir(parents=True, exist_ok=True)

    def ingest(
        self,
        source: Path,
        envelope: AudioEnvelope,
        vad_runtime: Any,
        asr_runtime: Any,
    ) -> ArchivedTranscript:
        vad, transcript = transcribe_speech(source, vad_runtime, asr_runtime)
        archived = self.archive.import_wav(source, envelope)
        record_path = self.records / f"{archived.source_digest}.json"
        record = {
            "schema_version": "archive-transcript.v1",
            "source_digest": archived.source_digest,
            "managed_audio": archived.managed_path,
            "request_id": archived.request_id,
            "consent_basis": archived.consent_basis,
            "retention_class": archived.retention_class,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "vad": vad,
            "transcription": transcript,
            "text_sha256": _digest_text(transcript.text),
        }
        if record_path.exists():
            earlier = _read_json(record_path)
            if earlier.get("source_digest") != archived.source_digest or earlier.get("text_sha256") != record["text_sha256"]:
                raise ContractError("ARCHIVE_RECORD_CONFLICT", "An archive record already exists with different content.")
        else:
            _write_json(record_path, record)
        return ArchivedTranscript(
            archived.source_digest,
            archived.managed_path,
            record_path,
            transcript.text,
            record["text_sha256"],
        )

    def search(self, query: str) -> tuple[dict[str, Any], ...]:
        normalized = query.strip().casefold()
        if not normalized or len(normalized) > 200:
            raise ContractError("ARCHIVE_QUERY_INVALID", "Search query is empty or exceeds its limit.")
        matches: list[dict[str, Any]] = []
        for path in sorted(self.records.glob("*.json")):
            record = _read_json(path)
            transcription = record.get("transcription")
            if not isinstance(transcription, dict) or not isinstance(transcription.get("text"), str):
                raise ContractError("WORKFLOW_RECORD_INVALID", "Archive transcript record is invalid.")
            if normalized in transcription["text"].casefold():
                matches.append(
                    {
                        "source_digest": record.get("source_digest"),
                        "text": transcription["text"],
                        "record": path.resolve(),
                    }
                )
        return tuple(matches)

    def delete(self, source_digest: str) -> bool:
        if not re.fullmatch(r"[0-9a-f]{64}", source_digest):
            raise ContractError("ARCHIVE_DIGEST_INVALID", "Archive source digest is invalid.")
        record_path = self.records / f"{source_digest}.json"
        audio_path = self.archive.sources / f"{source_digest}.wav"
        changed = False
        if audio_path.exists():
            audio_path.unlink()
            changed = True
        if record_path.exists():
            record_path.unlink()
            changed = True
        return changed


@dataclass(frozen=True)
class DictationPreview:
    preview_id: str
    record: Path
    text: str
    approval_digest: str


class ReviewedDictation:
    """Persist an ASR preview, require its digest, then support conflict-aware undo."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.preview_root = self.root / ".dictation" / "previews"
        self.undo_root = self.root / ".dictation" / "undo"
        self.preview_root.mkdir(parents=True, exist_ok=True)
        self.undo_root.mkdir(parents=True, exist_ok=True)

    def preview(self, source: Path, vad_runtime: Any, asr_runtime: Any) -> DictationPreview:
        vad, transcript = transcribe_speech(source, vad_runtime, asr_runtime)
        source_digest = sha256(source.read_bytes()).hexdigest()
        approval_digest = _digest_text(transcript.text)
        preview_id = f"preview_{source_digest[:12]}_{approval_digest[:12]}"
        record = self.preview_root / f"{preview_id}.json"
        _write_json(
            record,
            {
                "schema_version": "dictation-preview.v1",
                "preview_id": preview_id,
                "source_digest": source_digest,
                "text": transcript.text,
                "approval_digest": approval_digest,
                "vad": vad,
                "transcription": transcript,
            },
        )
        return DictationPreview(preview_id, record, transcript.text, approval_digest)

    def commit(self, preview_id: str, approval_digest: str, target: str, idempotency_key: str) -> UndoToken:
        if not re.fullmatch(r"preview_[0-9a-f]{12}_[0-9a-f]{12}", preview_id):
            raise ContractError("DICTATION_PREVIEW_INVALID", "Dictation preview identifier is invalid.")
        preview = _read_json(self.preview_root / f"{preview_id}.json")
        text = preview.get("text")
        expected = preview.get("approval_digest")
        if not isinstance(text, str) or not isinstance(expected, str):
            raise ContractError("WORKFLOW_RECORD_INVALID", "Dictation preview record is invalid.")
        if approval_digest != expected:
            raise ContractError("DICTATION_APPROVAL_MISMATCH", "Approval digest does not match the reviewed transcript.")
        if not IDEMPOTENCY_KEY.fullmatch(idempotency_key):
            raise ContractError("IDEMPOTENCY_KEY_INVALID", "Idempotency key is invalid.")
        undo_record = self.undo_root / f"{idempotency_key}.json"
        if undo_record.exists():
            earlier = _read_json(undo_record)
            earlier_target = (self.root / str(earlier.get("target", ""))).resolve()
            requested_target = ManagedTextDestination(self.root)._target(target)
            if (
                earlier_target != requested_target
                or earlier.get("after_digest") != _digest_text(text)
                or earlier.get("idempotency_key") != idempotency_key
            ):
                raise ContractError("IDEMPOTENCY_CONFLICT", "Idempotency key was reused for different text or target.")
            current = earlier_target.read_bytes() if earlier_target.exists() else b""
            if sha256(current).hexdigest() != earlier.get("after_digest"):
                raise ContractError("IDEMPOTENCY_STATE_CONFLICT", "Destination changed after the completed dictation commit.")
            try:
                return UndoToken(
                    target=earlier_target,
                    before=b64decode(str(earlier["before_base64"]), validate=True),
                    before_digest=str(earlier["before_digest"]),
                    after_digest=str(earlier["after_digest"]),
                    idempotency_key=idempotency_key,
                )
            except (KeyError, ValueError) as error:
                raise ContractError("WORKFLOW_RECORD_INVALID", "Dictation undo record is invalid.") from error
        token = ManagedTextDestination(self.root).commit(target, text, idempotency_key)
        _write_json(
            undo_record,
            {
                "schema_version": "dictation-undo.v1",
                "target": token.target.relative_to(self.root).as_posix(),
                "before_base64": b64encode(token.before).decode("ascii"),
                "before_digest": token.before_digest,
                "after_digest": token.after_digest,
                "idempotency_key": token.idempotency_key,
            },
        )
        return token

    def undo(self, idempotency_key: str) -> Path:
        if not IDEMPOTENCY_KEY.fullmatch(idempotency_key):
            raise ContractError("IDEMPOTENCY_KEY_INVALID", "Idempotency key is invalid.")
        record_path = self.undo_root / f"{idempotency_key}.json"
        record = _read_json(record_path)
        try:
            target = (self.root / str(record["target"])).resolve()
            token = UndoToken(
                target=target,
                before=b64decode(str(record["before_base64"]), validate=True),
                before_digest=str(record["before_digest"]),
                after_digest=str(record["after_digest"]),
                idempotency_key=str(record["idempotency_key"]),
            )
        except (KeyError, ValueError) as error:
            raise ContractError("WORKFLOW_RECORD_INVALID", "Dictation undo record is invalid.") from error
        ManagedTextDestination(self.root).undo(token)
        record_path.unlink()
        return target


@dataclass(frozen=True)
class NarrationBuild:
    manifest: Path
    master_audio: Path
    chunk_count: int
    source_sha256: str


class NarrationProject:
    """Synthesize stable chunks, validate them, and assemble one PCM16 master."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.chunks_root = self.root / "chunks"
        self.manifest = self.root / "manifest.json"
        self.master = self.root / "narration.wav"

    def build(self, source_text: str, runtime: Any, *, max_chars: int = 500, speed: float = 1.0, overwrite: bool = False) -> NarrationBuild:
        chunks = plan_narration(source_text, max_chars=max_chars)
        if not chunks:
            raise ContractError("NARRATION_SOURCE_EMPTY", "Narration source contains no text.")
        if self.manifest.exists() and not overwrite:
            raise ContractError("NARRATION_PROJECT_EXISTS", "Narration project already exists; use explicit overwrite.")
        self.chunks_root.mkdir(parents=True, exist_ok=True)
        built: list[dict[str, Any]] = []
        format_key: tuple[int, int, int] | None = None
        frames: list[bytes] = []
        for chunk in chunks:
            output = self.chunks_root / f"{chunk.chunk_id}.wav"
            synthesis = runtime.synthesize(chunk.text, output, speed=speed, overwrite=overwrite)
            try:
                with wave.open(str(output), "rb") as reader:
                    key = (reader.getnchannels(), reader.getsampwidth(), reader.getframerate())
                    if key[0:2] != (1, 2):
                        raise ContractError("NARRATION_FORMAT_INVALID", "Narration chunks must be mono PCM16 WAV.")
                    if format_key is not None and key != format_key:
                        raise ContractError("NARRATION_FORMAT_MISMATCH", "Narration chunk formats do not match.")
                    format_key = key
                    frames.append(reader.readframes(reader.getnframes()))
            except (OSError, wave.Error) as error:
                raise ContractError("NARRATION_FORMAT_INVALID", "Narration runtime produced an invalid WAV.") from error
            built.append({"plan": chunk, "synthesis": synthesis, "audio_sha256": sha256(output.read_bytes()).hexdigest()})
        assert format_key is not None
        self.root.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.master), "wb") as writer:
            writer.setnchannels(format_key[0])
            writer.setsampwidth(format_key[1])
            writer.setframerate(format_key[2])
            for content in frames:
                writer.writeframes(content)
        source_digest = _digest_text(source_text)
        _write_json(
            self.manifest,
            {
                "schema_version": "narration-project.v1",
                "source_sha256": source_digest,
                "source_characters": len(source_text),
                "master_audio": self.master,
                "master_sha256": sha256(self.master.read_bytes()).hexdigest(),
                "chunks": built,
            },
        )
        return NarrationBuild(self.manifest, self.master, len(chunks), source_digest)


def interpret_assistant_command(text: str) -> tuple[str, dict[str, str]]:
    """Map a deliberately tiny spoken grammar to the assistant policy boundary."""

    normalized = " ".join(text.strip().casefold().split()).rstrip(".!?")
    service_pattern = r"(voice(?:-| )service|media(?:-| )indexer)"
    status = re.fullmatch(rf"(?:show |get )?status (?:for |of )?{service_pattern}", normalized)
    if status:
        return "status", {"service": status.group(1).replace(" ", "-")}
    diagnostic = re.fullmatch(
        rf"run (healthcheck|health check|queue_snapshot|queue snapshot) (?:on|for) {service_pattern}", normalized
    )
    if diagnostic:
        return "diagnostic", {
            "diagnostic": diagnostic.group(1).replace(" ", "_").replace("health_check", "healthcheck"),
            "service": diagnostic.group(2).replace(" ", "-"),
        }
    raise ContractError("ASSISTANT_COMMAND_REJECTED", "Command is outside the closed assistant grammar; no action was taken.")
