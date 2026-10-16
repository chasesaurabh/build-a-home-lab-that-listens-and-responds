from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping
from urllib.parse import urlparse

from .contracts import ContractError


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SPDX_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
PURPOSES = {"vad", "asr", "tts", "diarization", "audio_event"}
REDISTRIBUTION = {"not_bundled", "bundled_if_permitted"}
MOVING_REVISIONS = {"main", "master", "head", "latest", "stable"}


def _https_url(value: Any, field_name: str) -> str:
    parsed = urlparse(str(value))
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ContractError("ARTIFACT_URL_INVALID", f"{field_name} must be an HTTPS URL without credentials.")
    return str(value)


def _local_relative_path(value: Any) -> str:
    raw = str(value).replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or path.parts[0] != "models":
        raise ContractError(
            "ARTIFACT_PATH_INVALID",
            "Artifact path must be relative, traversal-free, and inside models/.",
        )
    return str(path)


@dataclass(frozen=True)
class ModelArtifact:
    name: str
    purpose: str
    source_url: str
    revision: str
    sha256: str
    license_spdx: str
    license_url: str
    redistribution: str
    local_path: str
    gated: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelArtifact":
        required = {
            "name",
            "purpose",
            "source_url",
            "revision",
            "sha256",
            "license_spdx",
            "license_url",
            "redistribution",
            "local_path",
            "gated",
        }
        extras = sorted(set(data) - required)
        missing = sorted(required - set(data))
        if extras:
            raise ContractError("ARTIFACT_UNKNOWN_FIELD", f"Unknown artifact fields: {', '.join(extras)}")
        if missing:
            raise ContractError("ARTIFACT_MISSING_FIELD", f"Missing artifact fields: {', '.join(missing)}")

        name = str(data["name"]).strip()
        if not name or len(name) > 100:
            raise ContractError("ARTIFACT_NAME_INVALID", "Artifact name must contain 1 to 100 characters.")
        purpose = str(data["purpose"])
        if purpose not in PURPOSES:
            raise ContractError("ARTIFACT_PURPOSE_INVALID", "Artifact purpose is not supported.")
        revision = str(data["revision"]).strip()
        if not revision or revision.casefold() in MOVING_REVISIONS or any(char.isspace() for char in revision):
            raise ContractError("ARTIFACT_REVISION_MUTABLE", "Artifact revision must be immutable and contain no spaces.")
        digest = str(data["sha256"])
        if not SHA256.fullmatch(digest) or len(set(digest)) == 1:
            raise ContractError("ARTIFACT_DIGEST_INVALID", "Artifact digest must be a non-placeholder lowercase SHA-256 value.")
        license_spdx = str(data["license_spdx"])
        if not SPDX_ID.fullmatch(license_spdx):
            raise ContractError("ARTIFACT_LICENSE_INVALID", "Artifact license must use a single SPDX identifier.")
        redistribution = str(data["redistribution"])
        if redistribution not in REDISTRIBUTION:
            raise ContractError("ARTIFACT_REDISTRIBUTION_INVALID", "Artifact redistribution policy is invalid.")
        if not isinstance(data["gated"], bool):
            raise ContractError("ARTIFACT_GATED_INVALID", "Artifact gated value must be true or false.")

        return cls(
            name=name,
            purpose=purpose,
            source_url=_https_url(data["source_url"], "source_url"),
            revision=revision,
            sha256=digest,
            license_spdx=license_spdx,
            license_url=_https_url(data["license_url"], "license_url"),
            redistribution=redistribution,
            local_path=_local_relative_path(data["local_path"]),
            gated=data["gated"],
        )


def parse_manifest(data: Mapping[str, Any]) -> tuple[ModelArtifact, ...]:
    allowed = {"schema_version", "artifacts"}
    extras = sorted(set(data) - allowed)
    if extras:
        raise ContractError("MANIFEST_UNKNOWN_FIELD", f"Unknown manifest fields: {', '.join(extras)}")
    if data.get("schema_version") != "model-artifacts.v1":
        raise ContractError("MANIFEST_VERSION_UNSUPPORTED", "Model artifact manifest version is not supported.")
    raw_artifacts = data.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ContractError("MANIFEST_ARTIFACTS_INVALID", "Manifest artifacts must be a list.")

    artifacts = tuple(ModelArtifact.from_dict(item) for item in raw_artifacts if isinstance(item, Mapping))
    if len(artifacts) != len(raw_artifacts):
        raise ContractError("MANIFEST_ARTIFACT_INVALID", "Each artifact must be an object.")
    names = [artifact.name.casefold() for artifact in artifacts]
    paths = [artifact.local_path.casefold() for artifact in artifacts]
    if len(names) != len(set(names)):
        raise ContractError("ARTIFACT_NAME_DUPLICATE", "Artifact names must be unique.")
    if len(paths) != len(set(paths)):
        raise ContractError("ARTIFACT_PATH_DUPLICATE", "Artifact paths must be unique.")
    return artifacts


def load_manifest(path: str | Path) -> tuple[ModelArtifact, ...]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError("MANIFEST_READ_FAILED", "Model artifact manifest could not be read.") from error
    if not isinstance(data, Mapping):
        raise ContractError("MANIFEST_INVALID", "Model artifact manifest must be an object.")
    return parse_manifest(data)


def artifact_sha256(path: str | Path) -> str:
    """Hash one artifact file or a directory as a canonical path-and-content tree."""

    target = Path(path)
    if target.is_file():
        return sha256(target.read_bytes()).hexdigest()
    if not target.is_dir():
        raise ContractError("ARTIFACT_LOCAL_MISSING", "Configured model artifact is unavailable locally.")
    files = sorted(
        item
        for item in target.rglob("*")
        if item.is_file() and ".cache" not in item.relative_to(target).parts
    )
    if not files:
        raise ContractError("ARTIFACT_LOCAL_EMPTY", "Configured model artifact directory is empty.")
    digest = sha256()
    for item in files:
        relative = item.relative_to(target).as_posix().encode("utf-8")
        content_digest = sha256(item.read_bytes()).digest()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(content_digest)
    return digest.hexdigest()


def verify_local_artifacts(root: str | Path, artifacts: tuple[ModelArtifact, ...]) -> None:
    base = Path(root).resolve()
    for artifact in artifacts:
        target = (base / artifact.local_path).resolve()
        try:
            target.relative_to(base)
        except ValueError as error:
            raise ContractError("ARTIFACT_PATH_INVALID", "Model artifact resolved outside the project root.") from error
        if artifact_sha256(target) != artifact.sha256:
            raise ContractError(
                "ARTIFACT_DIGEST_MISMATCH",
                f"Local model artifact does not match the manifest: {artifact.name}.",
            )
