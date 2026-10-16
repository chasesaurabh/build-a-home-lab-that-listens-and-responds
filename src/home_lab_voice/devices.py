from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import struct
import threading
from typing import Any, Mapping, Protocol, Sequence
import wave

from .audio_fixtures import WaveInfo, inspect_pcm16_wav
from .contracts import ALLOWED_CONSENT, ALLOWED_RETENTION, ContractError


ALLOWED_CAPTURE_RATES = {8_000, 16_000, 44_100, 48_000}
MAX_CAPTURE_SECONDS = 30.0
MAX_PLAYBACK_SECONDS = 120.0


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    input_channels: int
    output_channels: int
    default_sample_rate_hz: int
    default_input: bool
    default_output: bool


@dataclass(frozen=True)
class CaptureResult:
    device: AudioDevice
    output: Path
    captured_at: datetime
    consent_basis: str
    retention_class: str
    overflow_count: int
    wave: WaveInfo


@dataclass(frozen=True)
class PlaybackResult:
    device: AudioDevice
    source: Path
    played_at: datetime
    underflow_count: int
    playback_sample_rate_hz: int
    wave: WaveInfo


class AudioBackend(Protocol):
    def query_devices(self) -> Sequence[Mapping[str, Any]]: ...

    def query_hostapis(self) -> Sequence[Mapping[str, Any]]: ...

    def default_devices(self) -> tuple[int, int]: ...

    def check_input_settings(self, *, device: int, channels: int, dtype: str, samplerate: int) -> None: ...

    def check_output_settings(self, *, device: int, channels: int, dtype: str, samplerate: int) -> None: ...

    def capture_pcm16(
        self,
        *,
        device: int,
        channels: int,
        sample_rate_hz: int,
        frame_count: int,
        block_frames: int,
    ) -> tuple[bytes, int]: ...

    def play_pcm16(
        self,
        content: bytes,
        *,
        device: int,
        channels: int,
        sample_rate_hz: int,
        block_frames: int,
    ) -> int: ...


class SoundDeviceBackend:
    """Optional PortAudio boundary loaded only for live audio commands."""

    @staticmethod
    def _module():
        try:
            import sounddevice  # type: ignore[import-not-found]
        except ImportError as error:
            raise ContractError(
                "AUDIO_DEPENDENCY_MISSING",
                "Live audio requires the optional audio dependencies. Install this package with [audio].",
            ) from error
        return sounddevice

    def query_devices(self) -> Sequence[Mapping[str, Any]]:
        return tuple(dict(item) for item in self._module().query_devices())

    def query_hostapis(self) -> Sequence[Mapping[str, Any]]:
        return tuple(dict(item) for item in self._module().query_hostapis())

    def default_devices(self) -> tuple[int, int]:
        value = self._module().default.device
        return int(value[0]), int(value[1])

    def check_input_settings(self, *, device: int, channels: int, dtype: str, samplerate: int) -> None:
        try:
            self._module().check_input_settings(
                device=device,
                channels=channels,
                dtype=dtype,
                samplerate=samplerate,
            )
        except Exception as error:
            raise ContractError(
                "AUDIO_SETTINGS_UNSUPPORTED",
                "The selected capture device does not support the requested format.",
            ) from error

    def capture_pcm16(
        self,
        *,
        device: int,
        channels: int,
        sample_rate_hz: int,
        frame_count: int,
        block_frames: int,
    ) -> tuple[bytes, int]:
        module = self._module()
        captured = bytearray()
        overflows = 0
        completed = threading.Event()
        bytes_per_frame = channels * 2

        def callback(indata: Any, frames: int, _time: Any, status: Any) -> None:
            nonlocal overflows
            remaining_bytes = frame_count * bytes_per_frame - len(captured)
            if remaining_bytes <= 0:
                completed.set()
                raise module.CallbackStop
            content = bytes(indata)
            captured.extend(content[:remaining_bytes])
            overflows += int(bool(getattr(status, "input_overflow", False)))
            if len(captured) >= frame_count * bytes_per_frame:
                completed.set()
                raise module.CallbackStop

        try:
            with module.RawInputStream(
                device=device,
                channels=channels,
                dtype="int16",
                samplerate=sample_rate_hz,
                blocksize=block_frames,
                callback=callback,
            ):
                if not completed.wait(frame_count / sample_rate_hz + 5.0):
                    raise ContractError("AUDIO_CAPTURE_TIMEOUT", "Audio capture exceeded its deadline.")
        except ContractError:
            raise
        except Exception as error:
            raise ContractError("AUDIO_CAPTURE_FAILED", "Audio capture did not complete.") from error
        return bytes(captured), overflows

    def check_output_settings(self, *, device: int, channels: int, dtype: str, samplerate: int) -> None:
        try:
            self._module().check_output_settings(
                device=device,
                channels=channels,
                dtype=dtype,
                samplerate=samplerate,
            )
        except Exception as error:
            raise ContractError(
                "AUDIO_SETTINGS_UNSUPPORTED",
                "The selected playback device does not support the requested format.",
            ) from error

    def play_pcm16(
        self,
        content: bytes,
        *,
        device: int,
        channels: int,
        sample_rate_hz: int,
        block_frames: int,
    ) -> int:
        module = self._module()
        bytes_per_frame = channels * 2
        underflows = 0
        offset = 0
        completed = threading.Event()

        def callback(outdata: Any, frames: int, _time: Any, status: Any) -> None:
            nonlocal offset, underflows
            requested_bytes = frames * bytes_per_frame
            chunk = content[offset : offset + requested_bytes]
            offset += len(chunk)
            if len(chunk) < requested_bytes:
                chunk += b"\x00" * (requested_bytes - len(chunk))
            outdata[:] = chunk
            underflows += int(bool(getattr(status, "output_underflow", False)))
            if offset >= len(content):
                completed.set()
                raise module.CallbackStop

        try:
            with module.RawOutputStream(
                device=device,
                channels=channels,
                dtype="int16",
                samplerate=sample_rate_hz,
                blocksize=block_frames,
                callback=callback,
            ):
                if not completed.wait(len(content) / bytes_per_frame / sample_rate_hz + 5.0):
                    raise ContractError("AUDIO_PLAYBACK_TIMEOUT", "Audio playback exceeded its deadline.")
        except ContractError:
            raise
        except Exception as error:
            raise ContractError("AUDIO_PLAYBACK_FAILED", "Audio playback did not complete.") from error
        return underflows


def list_audio_devices(backend: AudioBackend | None = None) -> tuple[AudioDevice, ...]:
    selected_backend = backend or SoundDeviceBackend()
    devices = selected_backend.query_devices()
    host_apis = selected_backend.query_hostapis()
    default_input, default_output = selected_backend.default_devices()
    result: list[AudioDevice] = []
    for index, raw in enumerate(devices):
        host_index = int(raw.get("hostapi", -1))
        host_name = "unknown"
        if 0 <= host_index < len(host_apis):
            host_name = str(host_apis[host_index].get("name", "unknown"))
        result.append(
            AudioDevice(
                index=index,
                name=str(raw.get("name", "unnamed device")),
                host_api=host_name,
                input_channels=int(raw.get("max_input_channels", 0)),
                output_channels=int(raw.get("max_output_channels", 0)),
                default_sample_rate_hz=round(float(raw.get("default_samplerate", 0))),
                default_input=index == default_input,
                default_output=index == default_output,
            )
        )
    return tuple(result)


def select_audio_device(
    devices: Sequence[AudioDevice],
    *,
    direction: str,
    selector: str | None,
) -> AudioDevice:
    if direction not in {"input", "output"}:
        raise ValueError("direction must be input or output")
    eligible = [
        device
        for device in devices
        if (device.input_channels if direction == "input" else device.output_channels) > 0
    ]
    if selector is None:
        defaults = [
            device
            for device in eligible
            if (device.default_input if direction == "input" else device.default_output)
        ]
        if len(defaults) != 1:
            raise ContractError("AUDIO_DEFAULT_UNAVAILABLE", f"No unique default {direction} device is available.")
        return defaults[0]

    stripped = selector.strip()
    if not stripped:
        raise ContractError("AUDIO_DEVICE_INVALID", "Audio device selector must not be empty.")
    if stripped.isdecimal():
        matches = [device for device in eligible if device.index == int(stripped)]
    else:
        host_selector: str | None = None
        name_selector = stripped
        if "::" in stripped:
            host_selector, name_selector = (part.strip() for part in stripped.split("::", 1))
            if not host_selector or not name_selector:
                raise ContractError(
                    "AUDIO_DEVICE_INVALID",
                    "Qualified audio selectors must use HOST_API::DEVICE_NAME.",
                )
        folded_name = name_selector.casefold()
        folded_host = host_selector.casefold() if host_selector is not None else None
        matches = [
            device
            for device in eligible
            if folded_name in device.name.casefold()
            and (folded_host is None or folded_host == device.host_api.casefold())
        ]
        exact_name_matches = [device for device in matches if folded_name == device.name.casefold()]
        if exact_name_matches:
            matches = exact_name_matches
    if not matches:
        raise ContractError("AUDIO_DEVICE_NOT_FOUND", f"No matching {direction} device is available.")
    if len(matches) > 1:
        indexes = ", ".join(str(device.index) for device in matches)
        raise ContractError(
            "AUDIO_DEVICE_AMBIGUOUS",
            f"Audio device selector is ambiguous; use one of these indexes: {indexes}.",
        )
    return matches[0]


def capture_pcm16_wav(
    output: Path,
    *,
    device_selector: str | None = None,
    duration_seconds: float = 3.0,
    sample_rate_hz: int = 16_000,
    channels: int = 1,
    consent_basis: str = "self_recorded",
    retention_class: str = "ephemeral",
    overwrite: bool = False,
    backend: AudioBackend | None = None,
) -> CaptureResult:
    if not 0.1 <= duration_seconds <= MAX_CAPTURE_SECONDS:
        raise ContractError(
            "CAPTURE_DURATION_INVALID",
            f"Capture duration must be between 0.1 and {MAX_CAPTURE_SECONDS:g} seconds.",
        )
    if sample_rate_hz not in ALLOWED_CAPTURE_RATES:
        raise ContractError("CAPTURE_RATE_INVALID", "Capture sample rate is not enabled.")
    if channels not in {1, 2}:
        raise ContractError("CAPTURE_CHANNELS_INVALID", "Capture must use one or two channels.")
    if consent_basis not in ALLOWED_CONSENT:
        raise ContractError("CONSENT_BASIS_REQUIRED", "An approved consent basis is required.")
    if retention_class not in ALLOWED_RETENTION:
        raise ContractError("RETENTION_CLASS_INVALID", "Retention class is not configured.")
    if output.suffix.casefold() != ".wav":
        raise ContractError("CAPTURE_OUTPUT_INVALID", "Capture output must use the .wav extension.")
    if output.exists() and not overwrite:
        raise ContractError("CAPTURE_OUTPUT_EXISTS", "Capture output already exists; use explicit overwrite to replace it.")

    selected_backend = backend or SoundDeviceBackend()
    device = select_audio_device(
        list_audio_devices(selected_backend),
        direction="input",
        selector=device_selector,
    )
    capture_rate = device.default_sample_rate_hz
    if not 8_000 <= capture_rate <= 192_000:
        raise ContractError("CAPTURE_DEVICE_RATE_INVALID", "Capture device reported an invalid default sample rate.")
    selected_backend.check_input_settings(
        device=device.index,
        channels=channels,
        dtype="int16",
        samplerate=capture_rate,
    )
    capture_frame_count = round(capture_rate * duration_seconds)
    content, overflows = selected_backend.capture_pcm16(
        device=device.index,
        channels=channels,
        sample_rate_hz=capture_rate,
        frame_count=capture_frame_count,
        block_frames=min(1_024, capture_frame_count),
    )
    captured_bytes = capture_frame_count * channels * 2
    if len(content) != captured_bytes:
        raise ContractError("AUDIO_CAPTURE_TRUNCATED", "Captured audio was shorter than requested.")
    content = _resample_pcm16(
        content,
        channels=channels,
        source_rate_hz=capture_rate,
        target_rate_hz=sample_rate_hz,
    )
    output_bytes = round(sample_rate_hz * duration_seconds) * channels * 2
    content = (content + b"\x00" * max(0, output_bytes - len(content)))[:output_bytes]

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with wave.open(str(output), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate_hz)
            writer.writeframes(content)
    except (OSError, wave.Error) as error:
        raise ContractError("CAPTURE_WRITE_FAILED", "Captured audio could not be written.") from error

    return CaptureResult(
        device=device,
        output=output.resolve(),
        captured_at=datetime.now(timezone.utc),
        consent_basis=consent_basis,
        retention_class=retention_class,
        overflow_count=overflows,
        wave=inspect_pcm16_wav(output),
    )


def play_pcm16_wav(
    source: Path,
    *,
    device_selector: str | None = None,
    backend: AudioBackend | None = None,
) -> PlaybackResult:
    info = inspect_pcm16_wav(source)
    if info.duration_seconds > MAX_PLAYBACK_SECONDS:
        raise ContractError(
            "PLAYBACK_DURATION_INVALID",
            f"Playback duration must not exceed {MAX_PLAYBACK_SECONDS:g} seconds.",
        )
    selected_backend = backend or SoundDeviceBackend()
    device = select_audio_device(
        list_audio_devices(selected_backend),
        direction="output",
        selector=device_selector,
    )
    playback_rate = device.default_sample_rate_hz
    if not 8_000 <= playback_rate <= 192_000:
        raise ContractError("PLAYBACK_RATE_INVALID", "Playback device reported an invalid default sample rate.")
    selected_backend.check_output_settings(
        device=device.index,
        channels=info.channels,
        dtype="int16",
        samplerate=playback_rate,
    )
    try:
        with wave.open(str(source), "rb") as reader:
            content = reader.readframes(info.frame_count)
    except (OSError, wave.Error) as error:
        raise ContractError("PLAYBACK_READ_FAILED", "Playback audio could not be read.") from error
    playback_content = _resample_pcm16(
        content,
        channels=info.channels,
        source_rate_hz=info.sample_rate_hz,
        target_rate_hz=playback_rate,
    )
    underflows = selected_backend.play_pcm16(
        playback_content,
        device=device.index,
        channels=info.channels,
        sample_rate_hz=playback_rate,
        block_frames=min(1_024, max(1, len(playback_content) // (info.channels * 2))),
    )
    return PlaybackResult(
        device=device,
        source=source.resolve(),
        played_at=datetime.now(timezone.utc),
        underflow_count=underflows,
        playback_sample_rate_hz=playback_rate,
        wave=info,
    )


def _resample_pcm16(
    content: bytes,
    *,
    channels: int,
    source_rate_hz: int,
    target_rate_hz: int,
) -> bytes:
    if source_rate_hz == target_rate_hz:
        return content
    source_frames = len(content) // (channels * 2)
    if source_frames == 0:
        return b""
    values = struct.unpack(f"<{source_frames * channels}h", content)
    target_frames = max(1, round(source_frames * target_rate_hz / source_rate_hz))
    converted: list[int] = []
    for target_index in range(target_frames):
        source_position = target_index * source_rate_hz / target_rate_hz
        left = min(int(source_position), source_frames - 1)
        right = min(left + 1, source_frames - 1)
        fraction = source_position - left
        for channel in range(channels):
            left_value = values[left * channels + channel]
            right_value = values[right * channels + channel]
            converted.append(round(left_value + (right_value - left_value) * fraction))
    return struct.pack(f"<{len(converted)}h", *converted)
