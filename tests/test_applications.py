from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.applications.archive import ArchiveRecord, ManagedArchive  # noqa: E402
from home_lab_voice.applications.dictation import ManagedTextDestination  # noqa: E402
from home_lab_voice.applications.narration import plan_narration  # noqa: E402
from home_lab_voice.audio_fixtures import write_mono_sine  # noqa: E402
from home_lab_voice.contracts import AudioEnvelope, ContractError  # noqa: E402


def envelope_for(path: Path, *, channels: int = 1) -> AudioEnvelope:
    return AudioEnvelope(
        request_id="req_01JAPPTEST99",
        source_kind="licensed_fixture",
        media_type="audio/wav",
        sample_rate_hz=16_000,
        channels=channels,
        captured_at=datetime(2026, 9, 22, tzinfo=timezone.utc),
        consent_basis="licensed_fixture",
        retention_class="project_archive",
        content_sha256=sha256(path.read_bytes()).hexdigest(),
    )


class ArchiveApplicationTests(unittest.TestCase):
    def test_import_uses_digest_path_and_preserves_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "fixture.wav", duration_seconds=0.1)
            record = ManagedArchive(root / "archive").import_wav(source, envelope_for(source))
            self.assertEqual(record.managed_path.name, f"{record.source_digest}.wav")
            self.assertEqual(record.wave.frame_count, 1_600)
            self.assertEqual(record.consent_basis, "licensed_fixture")

    def test_duplicate_import_is_content_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "fixture.wav", duration_seconds=0.1)
            archive = ManagedArchive(root / "archive")
            first = archive.import_wav(source, envelope_for(source))
            second = archive.import_wav(source, envelope_for(source))
            self.assertEqual(first.managed_path, second.managed_path)
            self.assertEqual(len(list((root / "archive" / "sources").iterdir())), 1)

    def test_envelope_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "fixture.wav", duration_seconds=0.1)
            with self.assertRaises(ContractError) as caught:
                ManagedArchive(root / "archive").import_wav(source, envelope_for(source, channels=2))
            self.assertEqual(caught.exception.code, "AUDIO_ENVELOPE_MISMATCH")

    def test_delete_is_managed_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_mono_sine(root / "fixture.wav", duration_seconds=0.1)
            archive = ManagedArchive(root / "archive")
            record = archive.import_wav(source, envelope_for(source))
            self.assertTrue(archive.delete_source(record))
            self.assertFalse(archive.delete_source(record))

            outside = ArchiveRecord(record.request_id, record.source_digest, source, record.wave, record.retention_class, record.consent_basis)
            with self.assertRaises(ContractError):
                archive.delete_source(outside)


class DictationApplicationTests(unittest.TestCase):
    def test_commit_and_undo_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = ManagedTextDestination(Path(temporary))
            token = destination.commit("notes.txt", "Reviewed dictation.", "idem_01JDICTATE99")
            self.assertEqual(token.target.read_text(encoding="utf-8"), "Reviewed dictation.")
            destination.undo(token)
            self.assertFalse(token.target.exists())

    def test_undo_restores_prior_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "notes.txt"
            target.write_text("Earlier text", encoding="utf-8")
            destination = ManagedTextDestination(root)
            token = destination.commit("notes.txt", "New text", "idem_01JDICTATE98")
            destination.undo(token)
            self.assertEqual(target.read_text(encoding="utf-8"), "Earlier text")

    def test_changed_destination_blocks_undo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = ManagedTextDestination(Path(temporary))
            token = destination.commit("notes.txt", "New text", "idem_01JDICTATE97")
            token.target.write_text("Independent edit", encoding="utf-8")
            with self.assertRaises(ContractError) as caught:
                destination.undo(token)
            self.assertEqual(caught.exception.code, "UNDO_CONFLICT")

    def test_duplicate_commit_and_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = ManagedTextDestination(Path(temporary))
            first = destination.commit("notes.txt", "Same text", "idem_01JDICTATE96")
            second = destination.commit("notes.txt", "Same text", "idem_01JDICTATE96")
            self.assertEqual(first, second)
            with self.assertRaises(ContractError) as caught:
                destination.commit("notes.txt", "Different text", "idem_01JDICTATE96")
            self.assertEqual(caught.exception.code, "IDEMPOTENCY_CONFLICT")

    def test_destination_escape_and_wrong_extension_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = ManagedTextDestination(Path(temporary))
            for name in ("../outside.txt", "notes.html"):
                with self.subTest(name=name), self.assertRaises(ContractError):
                    destination.commit(name, "Text", "idem_01JDICTATE95")


class NarrationApplicationTests(unittest.TestCase):
    def test_sentence_chunks_keep_source_offsets(self) -> None:
        source = "First sentence. Second sentence!\n\nThird paragraph?"
        chunks = plan_narration(source, max_chars=30)
        self.assertEqual([chunk.text for chunk in chunks], ["First sentence.", "Second sentence!", "Third paragraph?"])
        for chunk in chunks:
            self.assertEqual(source[chunk.source_start:chunk.source_end], chunk.text)

    def test_long_sentence_splits_on_word_boundary(self) -> None:
        source = "one two three four five six seven eight nine ten"
        chunks = plan_narration(source, max_chars=20)
        self.assertTrue(all(len(chunk.text) <= 20 for chunk in chunks))
        self.assertEqual(" ".join(chunk.text for chunk in chunks), source)

    def test_chunk_ids_are_stable_and_content_sensitive(self) -> None:
        first = plan_narration("One sentence. Two sentences.", max_chars=20)
        second = plan_narration("One sentence. Two sentences.", max_chars=20)
        changed = plan_narration("One sentence. New second.", max_chars=20)
        self.assertEqual(first, second)
        self.assertNotEqual(first[-1].chunk_id, changed[-1].chunk_id)

    def test_empty_source_produces_no_work(self) -> None:
        self.assertEqual(plan_narration("  \n "), ())


if __name__ == "__main__":
    unittest.main()
