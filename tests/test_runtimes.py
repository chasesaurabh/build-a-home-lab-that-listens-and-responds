from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from home_lab_voice.audio_fixtures import inspect_pcm16_wav, write_mono_sine, write_stereo_tone_and_silence
from home_lab_voice.contracts import ContractError
from home_lab_voice.runtimes import FasterWhisperRuntime, KokoroRuntime, SileroVadRuntime


class FakeWhisperModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def transcribe(self, audio: str, **options):
        self.calls.append((audio, options))
        segments = iter(
            (
                SimpleNamespace(
                    start=0.1,
                    end=0.6,
                    text=" hello ",
                    avg_logprob=-0.2,
                    no_speech_prob=0.01,
                ),
                SimpleNamespace(
                    start=0.7,
                    end=1.0,
                    text="home lab",
                    avg_logprob=-0.1,
                    no_speech_prob=0.02,
                ),
            )
        )
        info = SimpleNamespace(language="en", language_probability=0.98, duration=1.0)
        return segments, info


class FakeKokoroPipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **options):
        self.calls.append((text, options))
        return iter((("first", "fəːst", [0.0, 0.5]), ("second", "sɛkənd", [-0.5, 0.0])))


class RuntimeTests(unittest.TestCase):
    def test_silero_runtime_converts_regions_and_totals_speech(self) -> None:
        calls = []

        def timestamps(audio, model, **options):
            calls.append((audio, model, options))
            return ({"start": 0.10, "end": 0.40}, {"start": 0.60, "end": 0.90})

        runtime = SileroVadRuntime(
            model="fixture-model",
            timestamp_function=timestamps,
            tensor_factory=tuple,
        )
        with tempfile.TemporaryDirectory() as temporary:
            source = write_mono_sine(Path(temporary) / "input.wav", duration_seconds=1.0)
            result = runtime.analyze(source)
        self.assertEqual(len(result.regions), 2)
        self.assertAlmostEqual(result.speech_seconds, 0.60)
        self.assertEqual(calls[0][2]["sampling_rate"], 16_000)
        self.assertTrue(calls[0][2]["return_seconds"])

    def test_silero_runtime_rejects_stereo_and_invalid_regions(self) -> None:
        runtime = SileroVadRuntime(
            model="fixture-model",
            timestamp_function=lambda *_args, **_kwargs: ({"start": 0.9, "end": 0.1},),
            tensor_factory=tuple,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stereo = write_stereo_tone_and_silence(root / "stereo.wav")
            with self.assertRaisesRegex(ContractError, "mono"):
                runtime.analyze(stereo)
            mono = write_mono_sine(root / "mono.wav")
            with self.assertRaisesRegex(ContractError, "invalid region"):
                runtime.analyze(mono)

    def test_whisper_runtime_is_cpu_int8_and_consumes_generator(self) -> None:
        model = FakeWhisperModel()
        factory_calls = []

        def factory(path, **options):
            factory_calls.append((path, options))
            return model

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "model"
            model_path.mkdir()
            source = write_mono_sine(root / "input.wav")
            result = FasterWhisperRuntime(model_path, model_factory=factory).transcribe(source)
        self.assertEqual(result.text, "hello home lab")
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(factory_calls[0][1]["device"], "cpu")
        self.assertEqual(factory_calls[0][1]["compute_type"], "int8")
        self.assertTrue(factory_calls[0][1]["local_files_only"])
        self.assertFalse(model.calls[0][1]["vad_filter"])

    def test_whisper_runtime_validates_model_language_and_beam(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "input.wav", duration_seconds=0.1)
            runtime = FasterWhisperRuntime(root / "missing")
            with self.assertRaisesRegex(ContractError, "beam size"):
                runtime.transcribe(source, beam_size=0)
            with self.assertRaisesRegex(ContractError, "language"):
                runtime.transcribe(source, language="english")
            with self.assertRaisesRegex(ContractError, "model directory"):
                runtime.transcribe(source)

    def test_kokoro_runtime_writes_bounded_pcm16(self) -> None:
        pipeline = FakeKokoroPipeline()
        factory_calls = []

        def factory(model_dir, language_code):
            factory_calls.append((model_dir, language_code))
            return pipeline

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            (model_dir / "voices").mkdir(parents=True)
            (model_dir / "voices" / "af_heart.pt").write_bytes(b"fixture")
            output = root / "speech.wav"
            result = KokoroRuntime(model_dir, pipeline_factory=factory).synthesize("Hello.", output)
            wave = inspect_pcm16_wav(output)
        self.assertEqual((result.frame_count, wave.frame_count), (4, 4))
        self.assertEqual(result.sample_rate_hz, 24_000)
        self.assertEqual(factory_calls[0][1], "a")
        self.assertTrue(str(pipeline.calls[0][1]["voice"]).endswith("af_heart.pt"))

    def test_kokoro_runtime_rejects_unbounded_or_destructive_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            (model_dir / "voices").mkdir(parents=True)
            (model_dir / "voices" / "af_heart.pt").write_bytes(b"fixture")
            runtime = KokoroRuntime(model_dir, pipeline_factory=lambda *_: FakeKokoroPipeline())
            with self.assertRaisesRegex(ContractError, "empty"):
                runtime.synthesize(" ", root / "output.wav")
            with self.assertRaisesRegex(ContractError, "between"):
                runtime.synthesize("Hello", root / "output.wav", speed=3.0)
            output = root / "output.wav"
            output.write_bytes(b"existing")
            with self.assertRaisesRegex(ContractError, "already exists"):
                runtime.synthesize("Hello", output)


if __name__ == "__main__":
    unittest.main()
