from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
import math
import struct
import sys
from typing import Any, Callable, Iterable, Protocol, Sequence
import wave

from .audio_fixtures import WaveInfo, inspect_pcm16_wav
from .contracts import ContractError


MAX_TRANSCRIPTION_SECONDS = 600.0
MAX_SYNTHESIS_CHARACTERS = 2_000
KOKORO_SAMPLE_RATE_HZ = 24_000


@dataclass(frozen=True)
class SpeechRegion:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class VadResult:
    source: Path
    sample_rate_hz: int
    duration_seconds: float
    regions: tuple[SpeechRegion, ...]
    speech_seconds: float
    model_reference: str


@dataclass(frozen=True)
class TranscriptionSegment:
    start_seconds: float
    end_seconds: float
    text: str
    average_log_probability: float
    no_speech_probability: float


@dataclass(frozen=True)
class TranscriptionResult:
    source: Path
    text: str
    language: str
    language_probability: float
    duration_seconds: float
    segments: tuple[TranscriptionSegment, ...]
    model_reference: str
    device: str = "cpu"
    compute_type: str = "int8"


@dataclass(frozen=True)
class SynthesisResult:
    output: Path
    sample_rate_hz: int
    frame_count: int
    duration_seconds: float
    model_reference: str
    voice_reference: str
    device: str = "cpu"


class WhisperModelLike(Protocol):
    def transcribe(self, audio: str, **options: Any) -> tuple[Iterable[Any], Any]: ...


class PipelineLike(Protocol):
    def __call__(self, text: str, **options: Any) -> Iterable[Any]: ...


def _dependency_error(extra: str, detail: str) -> ContractError:
    return ContractError(
        "RUNTIME_DEPENDENCY_MISSING",
        f"{detail} Install this package with the [{extra}] extra.",
    )


def _require_pcm16_mono(path: Path, *, permitted_rates: set[int]) -> WaveInfo:
    info = inspect_pcm16_wav(path)
    if info.channels != 1:
        raise ContractError("RUNTIME_AUDIO_CHANNELS_UNSUPPORTED", "Runtime input must be mono PCM audio.")
    if info.sample_rate_hz not in permitted_rates:
        enabled = ", ".join(str(value) for value in sorted(permitted_rates))
        raise ContractError(
            "RUNTIME_AUDIO_RATE_UNSUPPORTED",
            f"Runtime input sample rate must be one of: {enabled} Hz.",
        )
    return info


def _read_pcm16_samples(path: Path) -> tuple[WaveInfo, tuple[float, ...]]:
    info = inspect_pcm16_wav(path)
    try:
        with wave.open(str(path), "rb") as reader:
            content = reader.readframes(info.frame_count)
    except (OSError, wave.Error) as error:
        raise ContractError("RUNTIME_AUDIO_READ_FAILED", "Runtime audio could not be read.") from error
    values = struct.unpack(f"<{info.frame_count * info.channels}h", content)
    if info.channels == 1:
        mono = values
    else:
        mono = tuple(
            sum(values[index : index + info.channels]) / info.channels
            for index in range(0, len(values), info.channels)
        )
    return info, tuple(float(value) / 32768.0 for value in mono)


class SileroVadRuntime:
    """CPU-only, offline Silero VAD boundary for bounded PCM WAV input."""

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        model: Any | None = None,
        timestamp_function: Callable[..., Sequence[dict[str, Any]]] | None = None,
        tensor_factory: Callable[[Sequence[float]], Any] | None = None,
        model_reference: str = "silero-vad==6.2.3/onnx-opset16",
    ) -> None:
        self.model_path = model_path
        self._model = model
        self._timestamp_function = timestamp_function
        self._tensor_factory = tensor_factory
        self.model_reference = model_path.resolve().as_posix() if model_path is not None else model_reference

    def _load(self) -> None:
        if self._model is not None and self._timestamp_function is not None and self._tensor_factory is not None:
            return
        try:
            import torch  # type: ignore[import-not-found]
            from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore[import-not-found]
            from silero_vad.utils_vad import OnnxWrapper  # type: ignore[import-not-found]
        except ImportError as error:
            raise _dependency_error("vad", "Voice activity detection dependencies are unavailable.") from error
        if self.model_path is not None:
            if not self.model_path.is_file():
                raise ContractError("VAD_MODEL_NOT_FOUND", "The configured local VAD model file is unavailable.")
            self._model = OnnxWrapper(str(self.model_path), force_onnx_cpu=True)
        else:
            self._model = load_silero_vad(onnx=True, opset_version=16)
        self._timestamp_function = get_speech_timestamps
        self._tensor_factory = lambda values: torch.tensor(values, dtype=torch.float32)

    def analyze(
        self,
        source: Path,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> VadResult:
        if not 0.0 < threshold < 1.0:
            raise ContractError("VAD_THRESHOLD_INVALID", "VAD threshold must be between zero and one.")
        for value, label in (
            (min_speech_duration_ms, "minimum speech"),
            (min_silence_duration_ms, "minimum silence"),
            (speech_pad_ms, "speech padding"),
        ):
            if value < 0 or value > 10_000:
                raise ContractError("VAD_DURATION_INVALID", f"VAD {label} duration is outside the supported range.")
        info = _require_pcm16_mono(source, permitted_rates={8_000, 16_000})
        if info.duration_seconds > MAX_TRANSCRIPTION_SECONDS:
            raise ContractError("RUNTIME_AUDIO_TOO_LONG", "Runtime audio exceeds the ten-minute limit.")
        _, samples = _read_pcm16_samples(source)
        self._load()
        assert self._timestamp_function is not None
        assert self._tensor_factory is not None
        try:
            raw_regions = self._timestamp_function(
                self._tensor_factory(samples),
                self._model,
                threshold=threshold,
                sampling_rate=info.sample_rate_hz,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=True,
            )
        except Exception as error:
            raise ContractError("VAD_INFERENCE_FAILED", "Voice activity detection did not complete.") from error
        regions: list[SpeechRegion] = []
        prior_end = 0.0
        for item in raw_regions:
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError) as error:
                raise ContractError("VAD_RESULT_INVALID", "Voice activity detector returned an invalid region.") from error
            if not (0.0 <= start < end <= info.duration_seconds + 0.01) or start < prior_end:
                raise ContractError("VAD_RESULT_INVALID", "Voice activity detector returned an invalid region.")
            regions.append(SpeechRegion(start, min(end, info.duration_seconds)))
            prior_end = end
        return VadResult(
            source=source.resolve(),
            sample_rate_hz=info.sample_rate_hz,
            duration_seconds=info.duration_seconds,
            regions=tuple(regions),
            speech_seconds=sum(region.end_seconds - region.start_seconds for region in regions),
            model_reference=self.model_reference,
        )


class FasterWhisperRuntime:
    """CPU/int8 Faster-Whisper boundary using a pre-fetched immutable model directory."""

    def __init__(
        self,
        model_path: Path,
        *,
        model_factory: Callable[..., WhisperModelLike] | None = None,
        cpu_threads: int = 0,
    ) -> None:
        self.model_path = model_path
        self._model_factory = model_factory
        self.cpu_threads = cpu_threads
        self._model: WhisperModelLike | None = None

    def _load(self) -> WhisperModelLike:
        if self._model is not None:
            return self._model
        if not self.model_path.is_dir():
            raise ContractError("ASR_MODEL_NOT_FOUND", "The configured local ASR model directory is unavailable.")
        factory = self._model_factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore[import-not-found]
            except ImportError as error:
                raise _dependency_error("asr", "Speech-recognition dependencies are unavailable.") from error
            factory = WhisperModel
        try:
            self._model = factory(
                str(self.model_path),
                device="cpu",
                compute_type="int8",
                cpu_threads=self.cpu_threads,
                local_files_only=True,
            )
        except Exception as error:
            raise ContractError("ASR_MODEL_LOAD_FAILED", "The local speech-recognition model could not be loaded.") from error
        return self._model

    def transcribe(
        self,
        source: Path,
        *,
        language: str | None = "en",
        beam_size: int = 5,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        info = inspect_pcm16_wav(source)
        if info.duration_seconds > MAX_TRANSCRIPTION_SECONDS:
            raise ContractError("RUNTIME_AUDIO_TOO_LONG", "Runtime audio exceeds the ten-minute limit.")
        if not 1 <= beam_size <= 10:
            raise ContractError("ASR_BEAM_SIZE_INVALID", "ASR beam size must be between one and ten.")
        if language is not None and (len(language) not in {2, 3} or not language.isalpha()):
            raise ContractError("ASR_LANGUAGE_INVALID", "ASR language must be a two- or three-letter code.")
        if initial_prompt is not None and len(initial_prompt) > 500:
            raise ContractError("ASR_PROMPT_TOO_LONG", "ASR initial prompt exceeds the configured limit.")
        try:
            raw_segments, transcription_info = self._load().transcribe(
                str(source.resolve()),
                language=language,
                task="transcribe",
                beam_size=beam_size,
                initial_prompt=initial_prompt,
                word_timestamps=False,
                vad_filter=False,
                condition_on_previous_text=True,
            )
            segments = tuple(
                TranscriptionSegment(
                    start_seconds=float(segment.start),
                    end_seconds=float(segment.end),
                    text=str(segment.text).strip(),
                    average_log_probability=float(getattr(segment, "avg_logprob", 0.0)),
                    no_speech_probability=float(getattr(segment, "no_speech_prob", 0.0)),
                )
                for segment in raw_segments
            )
        except ContractError:
            raise
        except Exception as error:
            raise ContractError("ASR_INFERENCE_FAILED", "Speech recognition did not complete.") from error
        text = " ".join(segment.text for segment in segments if segment.text).strip()
        return TranscriptionResult(
            source=source.resolve(),
            text=text,
            language=str(getattr(transcription_info, "language", language or "unknown")),
            language_probability=float(getattr(transcription_info, "language_probability", 0.0)),
            duration_seconds=float(getattr(transcription_info, "duration", info.duration_seconds)),
            segments=segments,
            model_reference=self.model_path.resolve().as_posix(),
        )


def _default_kokoro_pipeline(model_dir: Path, language_code: str) -> PipelineLike:
    try:
        from kokoro import KModel, KPipeline  # type: ignore[import-not-found]
    except ImportError as error:
        raise _dependency_error("tts", "Speech-synthesis dependencies are unavailable.") from error
    config = model_dir / "config.json"
    model_file = model_dir / "kokoro-v1_0.pth"
    if not config.is_file() or not model_file.is_file():
        raise ContractError("TTS_MODEL_NOT_FOUND", "The local Kokoro model snapshot is incomplete.")
    try:
        model = KModel(repo_id="hexgrad/Kokoro-82M", config=str(config), model=str(model_file))
        return KPipeline(
            lang_code=language_code,
            repo_id="hexgrad/Kokoro-82M",
            model=model.to("cpu").eval(),
            device="cpu",
        )
    except ContractError:
        raise
    except Exception as error:
        raise ContractError("TTS_MODEL_LOAD_FAILED", "The local speech-synthesis model could not be loaded.") from error


def _flatten_audio(chunk: Any) -> tuple[float, ...]:
    audio = getattr(chunk, "audio", None)
    if audio is None:
        try:
            audio = chunk[2]
        except (IndexError, TypeError) as error:
            raise ContractError("TTS_RESULT_INVALID", "Speech synthesizer returned an invalid audio chunk.") from error
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    if hasattr(audio, "reshape"):
        audio = audio.reshape(-1)
    try:
        values = tuple(float(value) for value in audio)
    except (TypeError, ValueError) as error:
        raise ContractError("TTS_RESULT_INVALID", "Speech synthesizer returned an invalid audio chunk.") from error
    if not values or any(not math.isfinite(value) for value in values):
        raise ContractError("TTS_RESULT_INVALID", "Speech synthesizer returned an invalid audio chunk.")
    return values


def _write_float_mono_pcm16(path: Path, samples: Sequence[float], sample_rate_hz: int) -> None:
    pcm = array("h", (round(max(-1.0, min(1.0, value)) * 32767) for value in samples))
    if sys.byteorder != "little":
        pcm.byteswap()
    try:
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate_hz)
            writer.writeframes(pcm.tobytes())
    except (OSError, wave.Error) as error:
        raise ContractError("TTS_WRITE_FAILED", "Synthesized audio could not be written.") from error


class KokoroRuntime:
    """CPU-only Kokoro boundary using an immutable local snapshot and voice pack."""

    def __init__(
        self,
        model_dir: Path,
        *,
        voice: str = "af_heart",
        language_code: str = "a",
        pipeline_factory: Callable[[Path, str], PipelineLike] | None = None,
    ) -> None:
        self.model_dir = model_dir
        self.voice = voice
        self.language_code = language_code
        self._pipeline_factory = pipeline_factory or _default_kokoro_pipeline
        self._pipeline: PipelineLike | None = None

    def _load(self) -> PipelineLike:
        if self._pipeline is None:
            if not self.model_dir.is_dir():
                raise ContractError("TTS_MODEL_NOT_FOUND", "The configured local TTS model directory is unavailable.")
            self._pipeline = self._pipeline_factory(self.model_dir, self.language_code)
        return self._pipeline

    def synthesize(
        self,
        text: str,
        output: Path,
        *,
        speed: float = 1.0,
        overwrite: bool = False,
    ) -> SynthesisResult:
        normalized = text.strip()
        if not normalized or len(normalized) > MAX_SYNTHESIS_CHARACTERS:
            raise ContractError("TTS_TEXT_LENGTH_INVALID", "TTS text is empty or over the configured limit.")
        if not 0.5 <= speed <= 2.0:
            raise ContractError("TTS_SPEED_INVALID", "TTS speed must be between 0.5 and 2.0.")
        if output.suffix.casefold() != ".wav":
            raise ContractError("TTS_OUTPUT_INVALID", "TTS output must use the .wav extension.")
        if output.exists() and not overwrite:
            raise ContractError("TTS_OUTPUT_EXISTS", "TTS output already exists; use explicit overwrite to replace it.")
        voice_path = self.model_dir / "voices" / f"{self.voice}.pt"
        if not voice_path.is_file():
            raise ContractError("TTS_VOICE_NOT_FOUND", "The configured local Kokoro voice pack is unavailable.")
        try:
            chunks = self._load()(
                normalized,
                voice=str(voice_path),
                speed=speed,
                split_pattern=r"\n+",
            )
            samples = tuple(value for chunk in chunks for value in _flatten_audio(chunk))
        except ContractError:
            raise
        except Exception as error:
            raise ContractError("TTS_INFERENCE_FAILED", "Speech synthesis did not complete.") from error
        if not samples:
            raise ContractError("TTS_EMPTY_AUDIO", "Speech synthesis returned no audio.")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_float_mono_pcm16(output, samples, KOKORO_SAMPLE_RATE_HZ)
        return SynthesisResult(
            output=output.resolve(),
            sample_rate_hz=KOKORO_SAMPLE_RATE_HZ,
            frame_count=len(samples),
            duration_seconds=len(samples) / KOKORO_SAMPLE_RATE_HZ,
            model_reference=self.model_dir.resolve().as_posix(),
            voice_reference=self.voice,
        )
