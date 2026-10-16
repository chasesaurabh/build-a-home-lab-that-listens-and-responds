from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..audio_fixtures import WaveInfo, inspect_pcm16_wav
from ..contracts import AudioEnvelope, ContractError


@dataclass(frozen=True)
class ArchiveRecord:
    request_id: str
    source_digest: str
    managed_path: Path
    wave: WaveInfo
    retention_class: str
    consent_basis: str


class ManagedArchive:
    """Import validated PCM WAV files into one explicitly managed root."""

    def __init__(self, root: Path, *, max_source_bytes: int = 32_000_000) -> None:
        self.root = root.resolve()
        self.max_source_bytes = max_source_bytes
        self.sources = self.root / "sources"
        self.sources.mkdir(parents=True, exist_ok=True)

    def import_wav(self, source: Path, envelope: AudioEnvelope) -> ArchiveRecord:
        source = source.resolve(strict=True)
        size = source.stat().st_size
        if size > self.max_source_bytes:
            raise ContractError("ARCHIVE_SOURCE_LIMIT", "Source file exceeds the archive import limit.")
        if envelope.media_type != "audio/wav":
            raise ContractError("ARCHIVE_MEDIA_UNSUPPORTED", "The archive reference path accepts PCM WAV only.")
        content = source.read_bytes()
        envelope.validate_content(content)
        wave = inspect_pcm16_wav(source, max_decoded_bytes=self.max_source_bytes)
        if wave.sample_rate_hz != envelope.sample_rate_hz or wave.channels != envelope.channels:
            raise ContractError("AUDIO_ENVELOPE_MISMATCH", "Decoded audio does not match its envelope.")

        managed = self.sources / f"{wave.file_sha256}.wav"
        if managed.exists() and managed.read_bytes() != content:
            raise ContractError("ARCHIVE_DIGEST_COLLISION", "Managed content conflicts with its digest path.")
        if not managed.exists():
            managed.write_bytes(content)
        return ArchiveRecord(
            request_id=envelope.request_id,
            source_digest=wave.file_sha256,
            managed_path=managed,
            wave=wave,
            retention_class=envelope.retention_class,
            consent_basis=envelope.consent_basis,
        )

    def delete_source(self, record: ArchiveRecord) -> bool:
        target = record.managed_path.resolve()
        if not target.is_relative_to(self.sources.resolve()):
            raise ContractError("ARCHIVE_PATH_UNMANAGED", "Refusing to delete outside the managed archive.")
        if target.name != f"{record.source_digest}.wav":
            raise ContractError("ARCHIVE_RECORD_MISMATCH", "Archive record does not match its managed path.")
        if not target.exists():
            return False
        target.unlink()
        return True
