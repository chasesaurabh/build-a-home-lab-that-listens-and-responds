from pathlib import Path
import tempfile
import unittest

from home_lab_voice.contracts import ContractError
from home_lab_voice.model_artifacts import (
    ModelArtifact,
    artifact_sha256,
    load_manifest,
    parse_manifest,
    verify_local_artifacts,
)


def valid_artifact(**overrides):
    data = {
        "name": "example-asr-small",
        "purpose": "asr",
        "source_url": "https://models.example.test/example-asr.bin",
        "revision": "v1.2.3",
        "sha256": "12" * 32,
        "license_spdx": "MIT",
        "license_url": "https://models.example.test/LICENSE",
        "redistribution": "not_bundled",
        "local_path": "models/asr/example-asr.bin",
        "gated": False,
    }
    data.update(overrides)
    return data


class ModelArtifactTests(unittest.TestCase):
    def test_empty_manifest_is_valid_and_enables_nothing(self):
        self.assertEqual(parse_manifest({"schema_version": "model-artifacts.v1", "artifacts": []}), ())

    def test_complete_artifact_is_accepted(self):
        artifact = ModelArtifact.from_dict(valid_artifact())
        self.assertEqual(artifact.purpose, "asr")
        self.assertEqual(artifact.local_path, "models/asr/example-asr.bin")

    def test_moving_revision_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "immutable"):
            ModelArtifact.from_dict(valid_artifact(revision="latest"))

    def test_placeholder_digest_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "non-placeholder"):
            ModelArtifact.from_dict(valid_artifact(sha256="0" * 64))

    def test_credentials_in_url_are_rejected(self):
        with self.assertRaisesRegex(ContractError, "without credentials"):
            ModelArtifact.from_dict(valid_artifact(source_url="https://user:secret@example.test/model.bin"))

    def test_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(ContractError, "traversal-free"):
            ModelArtifact.from_dict(valid_artifact(local_path="models/../outside.bin"))

    def test_duplicate_paths_are_rejected(self):
        first = valid_artifact()
        second = valid_artifact(name="second-asr", revision="v2")
        with self.assertRaisesRegex(ContractError, "paths must be unique"):
            parse_manifest({"schema_version": "model-artifacts.v1", "artifacts": [first, second]})

    def test_example_manifest_loads_without_artifacts(self):
        path = Path(__file__).parents[1] / "configs" / "model-artifacts.example.json"
        self.assertEqual(load_manifest(path), ())

    def test_locked_manifest_names_three_immutable_runtime_artifacts(self):
        path = Path(__file__).parents[1] / "configs" / "model-artifacts.lock.json"
        artifacts = load_manifest(path)
        self.assertEqual(tuple(item.purpose for item in artifacts), ("vad", "asr", "tts"))
        self.assertTrue(all(len(item.revision) == 40 for item in artifacts))

    def test_malformed_json_has_safe_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "could not be read"):
                load_manifest(path)

    def test_tree_digest_is_stable_and_cache_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models" / "asr" / "tiny"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            first = artifact_sha256(model)
            (model / ".cache").mkdir()
            (model / ".cache" / "metadata").write_text("moving", encoding="utf-8")
            self.assertEqual(first, artifact_sha256(model))
            (model / "config.json").write_text('{"changed": true}', encoding="utf-8")
            self.assertNotEqual(first, artifact_sha256(model))

    def test_local_artifact_verification_detects_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models" / "asr" / "tiny"
            model.mkdir(parents=True)
            (model / "model.bin").write_bytes(b"fixture")
            artifact = ModelArtifact.from_dict(
                valid_artifact(
                    local_path="models/asr/tiny",
                    sha256=artifact_sha256(model),
                )
            )
            verify_local_artifacts(root, (artifact,))
            (model / "model.bin").write_bytes(b"substituted")
            with self.assertRaisesRegex(ContractError, "does not match"):
                verify_local_artifacts(root, (artifact,))
