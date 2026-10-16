from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
from typing import Callable, Sequence
from urllib.request import Request, urlopen

from .contracts import ContractError


@dataclass(frozen=True)
class SnapshotSpec:
    name: str
    purpose: str
    repository: str
    revision: str
    directory: str
    allow_patterns: tuple[str, ...]
    required_files: tuple[str, ...]
    license_spdx: str
    license_url: str


@dataclass(frozen=True)
class DirectFileSpec:
    name: str
    purpose: str
    source_url: str
    revision: str
    relative_path: str
    sha256: str
    license_spdx: str
    license_url: str


SILERO_VAD_ONNX = DirectFileSpec(
    name="silero-vad-6.2.3-onnx-opset16",
    purpose="vad",
    source_url="https://raw.githubusercontent.com/snakers4/silero-vad/5cd7945676eb32225748052e2e6a0580e4686a08/src/silero_vad/data/silero_vad.onnx",
    revision="5cd7945676eb32225748052e2e6a0580e4686a08",
    relative_path="vad/silero-vad-6.2.3/silero_vad.onnx",
    sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    license_spdx="MIT",
    license_url="https://github.com/snakers4/silero-vad/blob/5cd7945676eb32225748052e2e6a0580e4686a08/LICENSE",
)


ASR_BASE_EN = SnapshotSpec(
    name="faster-whisper-base-en",
    purpose="asr",
    repository="Systran/faster-whisper-base.en",
    revision="3d3d5dee26484f91867d81cb899cfcf72b96be6c",
    directory="asr/faster-whisper-base.en",
    allow_patterns=("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
    required_files=("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"),
    license_spdx="MIT",
    license_url="https://huggingface.co/Systran/faster-whisper-base.en/blob/3d3d5dee26484f91867d81cb899cfcf72b96be6c/README.md",
)


KOKORO_82M = SnapshotSpec(
    name="kokoro-82m-af-heart",
    purpose="tts",
    repository="hexgrad/Kokoro-82M",
    revision="f3ff3571791e39611d31c381e3a41a3af07b4987",
    directory="tts/Kokoro-82M",
    allow_patterns=("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"),
    required_files=("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"),
    license_spdx="Apache-2.0",
    license_url="https://huggingface.co/hexgrad/Kokoro-82M/blob/f3ff3571791e39611d31c381e3a41a3af07b4987/README.md",
)


RUNTIME_SNAPSHOTS = (ASR_BASE_EN, KOKORO_82M)


def fetch_direct_file(
    root: Path,
    spec: DirectFileSpec,
    *,
    reader: Callable[[str], bytes] | None = None,
) -> Path:
    if reader is None:
        def reader(url: str) -> bytes:
            request = Request(url, headers={"User-Agent": "home-lab-voice-model-fetch/0.1.0"})
            with urlopen(request, timeout=60) as response:
                return response.read()

    destination = root / spec.relative_path
    try:
        content = reader(spec.source_url)
    except Exception as error:
        raise ContractError("MODEL_DOWNLOAD_FAILED", f"Model artifact download failed: {spec.name}.") from error
    if sha256(content).hexdigest() != spec.sha256:
        raise ContractError("MODEL_DOWNLOAD_DIGEST_MISMATCH", f"Model artifact digest mismatch: {spec.name}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def verify_direct_file(root: Path, spec: DirectFileSpec) -> Path:
    target = root / spec.relative_path
    if not target.is_file():
        raise ContractError("MODEL_SNAPSHOT_MISSING", f"Model artifact is unavailable: {spec.name}.")
    if sha256(target.read_bytes()).hexdigest() != spec.sha256:
        raise ContractError("MODEL_DOWNLOAD_DIGEST_MISMATCH", f"Model artifact digest mismatch: {spec.name}.")
    return target


def verify_snapshot(path: Path, spec: SnapshotSpec) -> None:
    if not path.is_dir():
        raise ContractError("MODEL_SNAPSHOT_MISSING", f"Model snapshot is unavailable: {spec.name}.")
    missing = [relative for relative in spec.required_files if not (path / relative).is_file()]
    if missing:
        raise ContractError(
            "MODEL_SNAPSHOT_INCOMPLETE",
            f"Model snapshot is incomplete: {spec.name}; missing {', '.join(missing)}.",
        )


def fetch_snapshot(
    root: Path,
    spec: SnapshotSpec,
    *,
    downloader: Callable[..., str] | None = None,
) -> Path:
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except ImportError as error:
            raise ContractError(
                "MODEL_DOWNLOAD_DEPENDENCY_MISSING",
                "Model download requires the ASR or TTS optional dependencies.",
            ) from error
        downloader = snapshot_download
    destination = root / spec.directory
    destination.mkdir(parents=True, exist_ok=True)
    try:
        downloader(
            repo_id=spec.repository,
            revision=spec.revision,
            local_dir=str(destination),
            allow_patterns=list(spec.allow_patterns),
        )
    except Exception as error:
        raise ContractError("MODEL_DOWNLOAD_FAILED", f"Model snapshot download failed: {spec.name}.") from error
    verify_snapshot(destination, spec)
    return destination


def choose_snapshots(names: Sequence[str]) -> tuple[SnapshotSpec, ...]:
    requested = set(names) - {"vad"}
    available = {spec.purpose: spec for spec in RUNTIME_SNAPSHOTS}
    unknown = sorted(requested - set(available))
    if unknown:
        raise ContractError("MODEL_PURPOSE_INVALID", f"Unknown model purposes: {', '.join(unknown)}.")
    return tuple(available[name] for name in ("asr", "tts") if name in requested)
