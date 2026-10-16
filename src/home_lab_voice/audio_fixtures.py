from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import struct
import wave

from .contracts import ContractError


@dataclass(frozen=True)
class WaveInfo:
    channels: int
    sample_width_bytes: int
    sample_rate_hz: int
    frame_count: int
    duration_seconds: float
    data_bytes: int
    peak_per_channel: tuple[int, ...]
    mean_per_channel: tuple[float, ...]
    file_sha256: str


def _pcm16(value: float) -> int:
    bounded = max(-1.0, min(1.0, value))
    return int(round(bounded * 32767))


def write_mono_sine(
    path: Path,
    *,
    sample_rate_hz: int = 16_000,
    duration_seconds: float = 1.0,
    frequency_hz: float = 1_000.0,
    amplitude: float = 0.25,
    dc_offset: float = 0.0,
) -> Path:
    if sample_rate_hz <= 0 or duration_seconds <= 0:
        raise ValueError("sample rate and duration must be positive")
    if not 0 <= amplitude <= 1:
        raise ValueError("amplitude must be between 0 and 1")
    frames = round(sample_rate_hz * duration_seconds)
    samples = [
        _pcm16(dc_offset + amplitude * math.sin(2 * math.pi * frequency_hz * index / sample_rate_hz))
        for index in range(frames)
    ]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def write_stereo_tone_and_silence(
    path: Path,
    *,
    sample_rate_hz: int = 16_000,
    duration_seconds: float = 1.0,
    amplitude: float = 0.25,
) -> Path:
    frames = round(sample_rate_hz * duration_seconds)
    interleaved: list[int] = []
    for index in range(frames):
        left = _pcm16(amplitude * math.sin(2 * math.pi * 1_000 * index / sample_rate_hz))
        interleaved.extend((left, 0))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate_hz)
        writer.writeframes(struct.pack(f"<{len(interleaved)}h", *interleaved))
    return path


def inspect_pcm16_wav(path: Path, *, max_decoded_bytes: int = 16_000_000) -> WaveInfo:
    try:
        with wave.open(str(path), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            compression = reader.getcomptype()
            if compression != "NONE" or sample_width != 2:
                raise ContractError("WAV_ENCODING_UNSUPPORTED", "Only uncompressed 16-bit PCM WAV is enabled.")
            if not 1 <= channels <= 8:
                raise ContractError("CHANNEL_COUNT_INVALID", "Channel count is outside the supported range.")
            if not 8_000 <= sample_rate <= 192_000:
                raise ContractError("SAMPLE_RATE_INVALID", "Sample rate is outside the supported range.")
            expected_bytes = frame_count * channels * sample_width
            if expected_bytes > max_decoded_bytes:
                raise ContractError("AUDIO_DECODE_LIMIT", "Decoded audio exceeds the configured limit.")
            data = reader.readframes(frame_count)
    except (wave.Error, EOFError) as error:
        raise ContractError("WAV_MALFORMED", "WAV container is malformed.") from error
    if len(data) != expected_bytes:
        raise ContractError("WAV_TRUNCATED", "WAV data is shorter than its declared frame count.")
    values = struct.unpack(f"<{frame_count * channels}h", data)
    per_channel = [values[index::channels] for index in range(channels)]
    peaks = tuple(max(abs(value) for value in samples) if samples else 0 for samples in per_channel)
    means = tuple(sum(samples) / len(samples) if samples else 0.0 for samples in per_channel)
    file_hash = sha256(path.read_bytes()).hexdigest()
    return WaveInfo(
        channels=channels,
        sample_width_bytes=sample_width,
        sample_rate_hz=sample_rate,
        frame_count=frame_count,
        duration_seconds=frame_count / sample_rate,
        data_bytes=expected_bytes,
        peak_per_channel=peaks,
        mean_per_channel=means,
        file_sha256=file_hash,
    )
