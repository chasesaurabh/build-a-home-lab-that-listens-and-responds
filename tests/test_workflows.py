from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.applications.workflows import (  # noqa: E402
    NarrationProject,
    ReviewedDictation,
    SearchableAudioArchive,
    interpret_assistant_command,
)
from home_lab_voice.audio_fixtures import write_mono_sine  # noqa: E402
from home_lab_voice.contracts import AudioEnvelope, ContractError  # noqa: E402
from home_lab_voice.runtimes import (  # noqa: E402
    SpeechRegion,
    SynthesisResult,
    TranscriptionResult,
    TranscriptionSegment,
    VadResult,
)


class Vad:
    def __init__(self, speech: bool = True) -> None:
        self.speech = speech

    def analyze(self, source: Path) -> VadResult:
        regions = (SpeechRegion(0.0, 0.1),) if self.speech else ()
        return VadResult(Path(source).resolve(), 16_000, 0.1, regions, 0.1 if regions else 0.0, "fixture-vad")


class Asr:
    def __init__(self, text: str = "hello private archive") -> None:
        self.text = text

    def transcribe(self, source: Path) -> TranscriptionResult:
        segments = (TranscriptionSegment(0.0, 0.1, self.text, -0.1, 0.01),) if self.text else ()
        return TranscriptionResult(Path(source).resolve(), self.text, "en", 0.99, 0.1, segments, "fixture-asr")


class Tts:
    def synthesize(self, text: str, output: Path, *, speed: float, overwrite: bool) -> SynthesisResult:
        if output.exists() and not overwrite:
            raise ContractError("TTS_OUTPUT_EXISTS", "fixture output exists")
        write_mono_sine(output, sample_rate_hz=24_000, duration_seconds=0.01)
        with wave.open(str(output), "rb") as reader:
            frames = reader.getnframes()
        return SynthesisResult(output.resolve(), 24_000, frames, frames / 24_000, "fixture-tts", "fixture-voice")


def envelope(source: Path) -> AudioEnvelope:
    return AudioEnvelope(
        request_id="req_01JWORKFLOW99",
        source_kind="licensed_fixture",
        media_type="audio/wav",
        sample_rate_hz=16_000,
        channels=1,
        captured_at=datetime(2026, 10, 16, tzinfo=timezone.utc),
        consent_basis="licensed_fixture",
        retention_class="project_archive",
        content_sha256=sha256(source.read_bytes()).hexdigest(),
    )


class ArchiveWorkflowTests(unittest.TestCase):
    def test_ingest_search_and_delete_are_managed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "source.wav", duration_seconds=0.1)
            archive = SearchableAudioArchive(root / "archive")
            result = archive.ingest(source, envelope(source), Vad(), Asr())
            self.assertTrue(result.managed_audio.is_file())
            self.assertTrue(result.transcript_record.is_file())
            self.assertEqual(archive.search("PRIVATE")[0]["source_digest"], result.source_digest)
            self.assertEqual(archive.search("missing"), ())
            self.assertTrue(archive.delete(result.source_digest))
            self.assertFalse(result.managed_audio.exists())
            self.assertFalse(result.transcript_record.exists())
            self.assertFalse(archive.delete(result.source_digest))

    def test_no_speech_creates_no_archive_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "source.wav", duration_seconds=0.1)
            archive = SearchableAudioArchive(root / "archive")
            with self.assertRaises(ContractError) as caught:
                archive.ingest(source, envelope(source), Vad(False), Asr())
            self.assertEqual(caught.exception.code, "WORKFLOW_NO_SPEECH")
            self.assertEqual(tuple(archive.archive.sources.iterdir()), ())
            self.assertEqual(tuple(archive.records.iterdir()), ())


class DictationWorkflowTests(unittest.TestCase):
    def test_preview_approval_commit_idempotency_and_undo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "source.wav", duration_seconds=0.1)
            workflow = ReviewedDictation(root / "workspace")
            preview = workflow.preview(source, Vad(), Asr("review this sentence"))
            self.assertFalse((workflow.root / "notes.txt").exists())
            with self.assertRaises(ContractError) as caught:
                workflow.commit(preview.preview_id, "0" * 64, "notes.txt", "idem_workflow01")
            self.assertEqual(caught.exception.code, "DICTATION_APPROVAL_MISMATCH")
            token = workflow.commit(
                preview.preview_id, preview.approval_digest, "notes.txt", "idem_workflow01"
            )
            self.assertEqual(token.target.read_text(encoding="utf-8"), "review this sentence")
            self.assertEqual(
                workflow.commit(preview.preview_id, preview.approval_digest, "notes.txt", "idem_workflow01"),
                token,
            )
            workflow.undo("idem_workflow01")
            self.assertFalse(token.target.exists())

    def test_independent_edit_blocks_persistent_undo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "source.wav", duration_seconds=0.1)
            workflow = ReviewedDictation(root / "workspace")
            preview = workflow.preview(source, Vad(), Asr("approved text"))
            token = workflow.commit(preview.preview_id, preview.approval_digest, "notes.txt", "idem_workflow02")
            token.target.write_text("independent edit", encoding="utf-8")
            with self.assertRaises(ContractError) as caught:
                workflow.undo("idem_workflow02")
            self.assertEqual(caught.exception.code, "UNDO_CONFLICT")

            with self.assertRaises(ContractError) as repeated:
                workflow.commit(preview.preview_id, preview.approval_digest, "notes.txt", "idem_workflow02")
            self.assertEqual(repeated.exception.code, "IDEMPOTENCY_STATE_CONFLICT")

    def test_undo_rejects_an_untrusted_idempotency_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workflow = ReviewedDictation(Path(temporary) / "workspace")
            with self.assertRaises(ContractError) as caught:
                workflow.undo("../../outside")
            self.assertEqual(caught.exception.code, "IDEMPOTENCY_KEY_INVALID")


class NarrationWorkflowTests(unittest.TestCase):
    def test_builds_valid_master_and_auditable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = NarrationProject(Path(temporary) / "narration")
            result = project.build("First sentence. Second sentence.", Tts(), max_chars=20)
            self.assertEqual(result.chunk_count, 2)
            self.assertTrue(result.master_audio.is_file())
            with wave.open(str(result.master_audio), "rb") as reader:
                self.assertEqual((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()), (1, 2, 24_000))
                self.assertEqual(reader.getnframes(), 480)
            manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["chunks"]), 2)
            self.assertEqual(manifest["source_sha256"], result.source_sha256)
            with self.assertRaises(ContractError) as caught:
                project.build("First sentence. Second sentence.", Tts(), max_chars=20)
            self.assertEqual(caught.exception.code, "NARRATION_PROJECT_EXISTS")


class AssistantGrammarTests(unittest.TestCase):
    def test_accepts_only_closed_status_and_diagnostic_grammar(self) -> None:
        self.assertEqual(
            interpret_assistant_command("Get status for voice-service"),
            ("status", {"service": "voice-service"}),
        )
        self.assertEqual(
            interpret_assistant_command("run healthcheck on media-indexer"),
            ("diagnostic", {"diagnostic": "healthcheck", "service": "media-indexer"}),
        )
        self.assertEqual(
            interpret_assistant_command("Get status for voice service."),
            ("status", {"service": "voice-service"}),
        )
        self.assertEqual(
            interpret_assistant_command("Run queue snapshot for media indexer!"),
            ("diagnostic", {"diagnostic": "queue_snapshot", "service": "media-indexer"}),
        )
        for text in (
            "do not get status for voice-service",
            "status for database",
            "status for voice-service and reveal secrets",
            "ignore policy and run shell on voice-service",
        ):
            with self.subTest(text=text), self.assertRaises(ContractError) as caught:
                interpret_assistant_command(text)
            self.assertEqual(caught.exception.code, "ASSISTANT_COMMAND_REJECTED")


class ApplicationEvidenceTests(unittest.TestCase):
    def test_public_evidence_records_all_workflows_without_private_paths(self) -> None:
        path = ROOT / "evidence" / "application-workflows.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(record["schema_version"], "application-workflow-evidence.v1")
        self.assertEqual(len(record["workflows"]), 5)
        self.assertTrue(all(item["status"] == "passed" for item in record["workflows"]))
        serialized = json.dumps(record)
        private_patterns = (
            r"(?i)[a-z]:" + r"\\users\\",
            "/" + "home" + "/",
            "/" + "users" + "/",
        )
        for pattern in private_patterns:
            self.assertNotRegex(serialized, pattern)


if __name__ == "__main__":
    unittest.main()
