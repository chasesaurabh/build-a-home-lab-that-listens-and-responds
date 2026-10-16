from pathlib import Path
import tempfile
import unittest

from home_lab_voice.contracts import ContractError
from home_lab_voice.runtime_models import (
    ASR_BASE_EN,
    DirectFileSpec,
    choose_snapshots,
    fetch_direct_file,
    fetch_snapshot,
    verify_direct_file,
    verify_snapshot,
)


class RuntimeModelTests(unittest.TestCase):
    def test_selection_is_explicit_and_stable(self):
        self.assertEqual(tuple(spec.purpose for spec in choose_snapshots(("tts", "asr"))), ("asr", "tts"))
        with self.assertRaisesRegex(ContractError, "Unknown model"):
            choose_snapshots(("vision",))

    def test_fetch_uses_immutable_revision_and_required_allowlist(self):
        calls = []

        def downloader(**options):
            calls.append(options)
            destination = Path(options["local_dir"])
            for relative in ASR_BASE_EN.required_files:
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(relative.encode("utf-8"))
            return str(destination)

        with tempfile.TemporaryDirectory() as temporary:
            destination = fetch_snapshot(Path(temporary), ASR_BASE_EN, downloader=downloader)
            verify_snapshot(destination, ASR_BASE_EN)
        self.assertEqual(calls[0]["revision"], ASR_BASE_EN.revision)
        self.assertEqual(tuple(calls[0]["allow_patterns"]), ASR_BASE_EN.allow_patterns)

    def test_incomplete_snapshot_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.mkdir(exist_ok=True)
            with self.assertRaisesRegex(ContractError, "incomplete"):
                verify_snapshot(root, ASR_BASE_EN)

    def test_direct_artifact_requires_the_pinned_digest(self):
        content = b"model-bytes"
        from hashlib import sha256

        spec = DirectFileSpec(
            name="fixture-vad",
            purpose="vad",
            source_url="https://models.example.test/vad.onnx",
            revision="abc123",
            relative_path="vad/fixture/model.onnx",
            sha256=sha256(content).hexdigest(),
            license_spdx="MIT",
            license_url="https://models.example.test/LICENSE",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = fetch_direct_file(root, spec, reader=lambda _url: content)
            self.assertEqual(verify_direct_file(root, spec), target)
            with self.assertRaisesRegex(ContractError, "digest mismatch"):
                fetch_direct_file(root, spec, reader=lambda _url: b"substitution")


if __name__ == "__main__":
    unittest.main()
