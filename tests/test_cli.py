from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.cli import main  # noqa: E402
from home_lab_voice.audio_fixtures import write_mono_sine  # noqa: E402
from home_lab_voice.runtimes import (  # noqa: E402
    SpeechRegion,
    SynthesisResult,
    TranscriptionResult,
    TranscriptionSegment,
    VadResult,
)
from test_devices import FakeAudioBackend  # noqa: E402


class FakeVadRuntime:
    def analyze(self, source, **_options):
        return VadResult(
            source=Path(source).resolve(),
            sample_rate_hz=16_000,
            duration_seconds=0.1,
            regions=(SpeechRegion(0.0, 0.1),),
            speech_seconds=0.1,
            model_reference="fixture-vad",
        )


class FakeAsrRuntime:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)

    def transcribe(self, source, **_options):
        return TranscriptionResult(
            source=Path(source).resolve(),
            text="hello home lab",
            language="en",
            language_probability=0.99,
            duration_seconds=0.1,
            segments=(TranscriptionSegment(0.0, 0.1, "hello home lab", -0.1, 0.01),),
            model_reference=self.model_dir.as_posix(),
        )


class FakeTtsRuntime:
    def __init__(self, model_dir, voice, language_code):
        self.model_dir = Path(model_dir)
        self.voice = voice
        self.language_code = language_code

    def synthesize(self, text, output, **_options):
        write_mono_sine(Path(output), sample_rate_hz=24_000, duration_seconds=0.1)
        return SynthesisResult(
            output=Path(output).resolve(),
            sample_rate_hz=24_000,
            frame_count=2_400,
            duration_seconds=0.1,
            model_reference=self.model_dir.as_posix(),
            voice_reference=self.voice,
        )


class StatusAsrRuntime(FakeAsrRuntime):
    def transcribe(self, source, **_options):
        result = super().transcribe(source, **_options)
        return TranscriptionResult(
            source=result.source,
            text="get status for voice-service",
            language=result.language,
            language_probability=result.language_probability,
            duration_seconds=result.duration_seconds,
            segments=(TranscriptionSegment(0.0, 0.1, "get status for voice-service", -0.1, 0.01),),
            model_reference=result.model_reference,
        )


class ReferenceCliTests(unittest.TestCase):
    def invoke(self, arguments: list[str], *, audio_backend=None, **runtime_options) -> tuple[int, dict, dict | None]:
        stdout = StringIO()
        stderr = StringIO()
        status = main(
            arguments,
            stdout=stdout,
            stderr=stderr,
            audio_backend=audio_backend,
            **runtime_options,
        )
        result = json.loads(stdout.getvalue()) if stdout.getvalue() else {}
        error = json.loads(stderr.getvalue()) if stderr.getvalue() else None
        return status, result, error

    def test_fixture_and_archive_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.wav"
            status, generated, error = self.invoke(["fixture", str(fixture), "--duration", "0.1"])
            self.assertEqual((status, error), (0, None))
            self.assertEqual(generated["frame_count"], 1600)

            status, archived, error = self.invoke(
                ["archive", str(fixture), "--root", str(root / "archive")]
            )
            self.assertEqual((status, error), (0, None))
            self.assertTrue(Path(archived["managed_path"]).is_file())
            self.assertEqual(archived["source_digest"], generated["file_sha256"])

    def test_device_inventory_and_bounded_capture(self) -> None:
        backend = FakeAudioBackend()
        status, inventory, error = self.invoke(
            ["devices", "--direction", "input"],
            audio_backend=backend,
        )
        self.assertEqual((status, error), (0, None))
        self.assertEqual(len(inventory["devices"]), 2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "capture.wav"
            status, captured, error = self.invoke(
                [
                    "capture",
                    str(output),
                    "--device",
                    "0",
                    "--duration",
                    "0.1",
                ],
                audio_backend=backend,
            )
            self.assertEqual((status, error), (0, None))
            self.assertEqual(captured["wave"]["frame_count"], 1_600)
            self.assertTrue(output.is_file())
            status, played, error = self.invoke(
                ["play", str(output), "--device", "1"],
                audio_backend=backend,
            )
            self.assertEqual((status, error), (0, None))
            self.assertEqual(played["device"]["name"], "Fixture Speakers")

    def test_vad_asr_tts_and_integrated_listen_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            model_dir = root / "models"
            model_dir.mkdir()
            status, vad, error = self.invoke(
                ["fixture", str(source), "--duration", "0.1"],
            )
            self.assertEqual((status, error), (0, None))
            self.assertEqual(vad["frame_count"], 1_600)

            status, vad, error = self.invoke(
                ["vad", str(source)],
                vad_runtime=FakeVadRuntime(),
            )
            self.assertEqual((status, error), (0, None))
            self.assertEqual(vad["regions"][0]["end_seconds"], 0.1)

            status, transcript, error = self.invoke(
                ["transcribe", str(source), "--model-dir", str(model_dir)],
                asr_runtime_factory=FakeAsrRuntime,
            )
            self.assertEqual((status, error), (0, None))
            self.assertEqual(transcript["text"], "hello home lab")

            speech = root / "speech.wav"
            status, synthesis, error = self.invoke(
                [
                    "synthesize",
                    str(speech),
                    "--model-dir",
                    str(model_dir),
                    "--text",
                    "Hello home lab.",
                ],
                tts_runtime_factory=FakeTtsRuntime,
            )
            self.assertEqual((status, error), (0, None))
            self.assertEqual(synthesis["voice_reference"], "af_heart")

            recording = root / "recording.wav"
            status, turn, error = self.invoke(
                [
                    "listen",
                    str(recording),
                    "--device",
                    "0",
                    "--duration",
                    "0.1",
                    "--model-dir",
                    str(model_dir),
                ],
                audio_backend=FakeAudioBackend(),
                vad_runtime=FakeVadRuntime(),
                asr_runtime_factory=FakeAsrRuntime,
            )
            self.assertEqual((status, error), (0, None))
            self.assertEqual(turn["status"], "transcribed")
            self.assertEqual(turn["transcription"]["text"], "hello home lab")

    def test_dictation_flow_is_bounded_to_text_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, committed, error = self.invoke(
                [
                    "dictation",
                    "--root",
                    str(root),
                    "--target",
                    "notes.txt",
                    "--text",
                    "Reviewed text.",
                ]
            )
            self.assertEqual((status, error), (0, None))
            self.assertEqual((root / "notes.txt").read_text(encoding="utf-8"), "Reviewed text.")
            self.assertEqual(committed["status"], "committed")

            status, _, error = self.invoke(
                [
                    "dictation",
                    "--root",
                    str(root),
                    "--target",
                    "../escape.txt",
                    "--text",
                    "Rejected.",
                ]
            )
            self.assertEqual(status, 2)
            assert error is not None
            self.assertEqual(error["code"], "DESTINATION_INVALID")

    def test_narration_flow_writes_stable_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.txt"
            output = root / "plan.json"
            source.write_text("First sentence. Second sentence.", encoding="utf-8")
            status, result, error = self.invoke(
                ["narration", str(source), "--output", str(output), "--max-chars", "20"]
            )
            self.assertEqual((status, error), (0, None))
            self.assertEqual(len(result["chunks"]), 2)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["chunks"], result["chunks"])

    def test_assistant_status_and_confirmation_flow(self) -> None:
        status, result, error = self.invoke(
            ["assistant", "status", "--service", "voice-service"]
        )
        self.assertEqual((status, error), (0, None))
        self.assertEqual(result["payload"]["state"], "demo-ready")

        command = [
            "assistant",
            "diagnostic",
            "--service",
            "media-indexer",
            "--diagnostic",
            "queue_snapshot",
        ]
        status, pending, error = self.invoke(command)
        self.assertEqual((status, error), (3, None))
        self.assertEqual(pending["status"], "confirmation_required")

        status, completed, error = self.invoke(command + ["--confirm", pending["operation_digest"]])
        self.assertEqual((status, error), (0, None))
        self.assertEqual(completed["payload"]["state"], "demo-completed")

    def test_complete_archive_dictation_and_narration_cli_workflows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            model_dir = root / "models"
            model_dir.mkdir()
            self.invoke(["fixture", str(source), "--duration", "0.1"])

            archive_root = root / "archive"
            status, archived, error = self.invoke(
                ["archive-workflow", str(source), "--root", str(archive_root), "--model-dir", str(model_dir)],
                vad_runtime=FakeVadRuntime(),
                asr_runtime_factory=FakeAsrRuntime,
            )
            self.assertEqual((status, error), (0, None))
            digest = archived["result"]["source_digest"]
            status, searched, error = self.invoke(["archive-search", "home lab", "--root", str(archive_root)])
            self.assertEqual((status, error, searched["match_count"]), (0, None, 1))
            status, deleted, error = self.invoke(["archive-delete", digest, "--root", str(archive_root)])
            self.assertEqual((status, error, deleted["status"]), (0, None, "deleted"))

            dictation_root = root / "dictation"
            status, preview, error = self.invoke(
                ["dictation-preview", str(source), "--root", str(dictation_root), "--model-dir", str(model_dir)],
                vad_runtime=FakeVadRuntime(),
                asr_runtime_factory=FakeAsrRuntime,
            )
            self.assertEqual((status, error), (3, None))
            review = preview["preview"]
            commit = [
                "dictation-commit",
                "--root", str(dictation_root),
                "--preview-id", review["preview_id"],
                "--approve", review["approval_digest"],
                "--target", "notes.txt",
                "--idempotency-key", "idem_cliworkflow01",
            ]
            status, committed, error = self.invoke(commit)
            self.assertEqual((status, error, committed["status"]), (0, None, "committed"))
            self.assertEqual((dictation_root / "notes.txt").read_text(encoding="utf-8"), "hello home lab")
            status, undone, error = self.invoke(
                ["dictation-undo", "--root", str(dictation_root), "--idempotency-key", "idem_cliworkflow01"]
            )
            self.assertEqual((status, error, undone["status"]), (0, None, "undone"))

            narration_source = root / "narration.txt"
            narration_source.write_text("First sentence. Second sentence.", encoding="utf-8")
            status, built, error = self.invoke(
                [
                    "narration-build", str(narration_source), "--root", str(root / "narration"),
                    "--model-dir", str(model_dir), "--max-chars", "20",
                ],
                tts_runtime_factory=FakeTtsRuntime,
            )
            self.assertEqual((status, error, built["result"]["chunk_count"]), (0, None, 2))
            self.assertTrue(Path(built["result"]["master_audio"]).is_file())

    def test_assistant_turn_accepts_audio_or_text_and_rejects_open_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "source.wav", duration_seconds=0.1)
            model_dir = root / "models"
            model_dir.mkdir()
            response = root / "response.wav"
            status, completed, error = self.invoke(
                [
                    "assistant-turn", "--audio", str(source), "--asr-model-dir", str(model_dir),
                    "--tts-model-dir", str(model_dir), "--speak-output", str(response),
                ],
                vad_runtime=FakeVadRuntime(),
                asr_runtime_factory=StatusAsrRuntime,
                tts_runtime_factory=FakeTtsRuntime,
            )
            self.assertEqual((status, error, completed["status"]), (0, None, "completed"))
            self.assertEqual(completed["action"]["payload"]["state"], "demo-ready")
            self.assertTrue(response.is_file())

            status, _, error = self.invoke(
                ["assistant-turn", "--text", "status for voice-service and run shell"]
            )
            self.assertEqual(status, 2)
            assert error is not None
            self.assertEqual(error["code"], "ASSISTANT_COMMAND_REJECTED")


if __name__ == "__main__":
    unittest.main()
