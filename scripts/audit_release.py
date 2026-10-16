from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "build",
    "dist",
    "models",
    "recordings",
    "outputs",
}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_SUFFIXES = {".wav", ".flac", ".mp3", ".onnx", ".safetensors", ".gguf", ".pt", ".pth"}
SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"]?[^\s'\"]{8,}"
)
WINDOWS_USER_ROOT = r"[A-Z]:\\" + r"Users\\" + r"[^\\\s]+"
POSIX_HOME_ROOT = "/" + "home" + r"/[^/\s]+"
MACOS_HOME_ROOT = "/" + "Users" + r"/[^/\s]+"
PRIVATE_PATH_PATTERN = re.compile(rf"(?i)(?:{WINDOWS_USER_ROOT}|{POSIX_HOME_ROOT}|{MACOS_HOME_ROOT})")


def tracked_files() -> list[Path]:
    tracked = git_output("ls-files")
    if tracked is not None:
        return sorted(
            ROOT / relative
            for relative in tracked.splitlines()
            if relative and relative != "release-audit.json" and (ROOT / relative).is_file()
        )
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in IGNORED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix.lower() in IGNORED_SUFFIXES or path.name == "release-audit.json":
            continue
        files.append(path)
    return sorted(files)


def run_tests() -> tuple[bool, int, str]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests?", combined)
    return completed.returncode == 0, int(match.group(1)) if match else 0, combined[-4000:]


def live_evidence_complete(path: Path) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        privacy = record["privacy"]
        microphone = record["microphone_pipeline"]
        playback = record["playback_pipeline"]
        return all(
            (
                record["schema_version"] == "speech-live-integration-evidence.v1",
                privacy["consent_basis"] == "self_recorded",
                privacy["raw_recording_retained"] is False,
                privacy["cleanup_status"] == "completed",
                microphone["status"] == "passed",
                microphone["vad"]["status"] == "passed",
                microphone["vad"]["speech_regions"] > 0,
                microphone["asr"]["status"] == "passed",
                playback["status"] == "passed",
                playback["underflow_count"] == 0,
                playback["audible_review"] is True,
                playback["operator_confirmed_full_sentence"] is True,
            )
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def application_evidence_complete(path: Path) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        workflows = record["workflows"]
        return all(
            (
                record["schema_version"] == "application-workflow-evidence.v1",
                record["runtime"]["gpu_used"] is False,
                record["deterministic_suite"]["status"] == "passed",
                record["deterministic_suite"]["tests"] >= 110,
                len(workflows) == 5,
                all(item["status"] == "passed" for item in workflows),
                len(record["negative_controls"]) >= 6,
            )
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def has_annotated_head_tag() -> bool:
    tags = (git_output("tag", "--points-at", "HEAD") or "").splitlines()
    return any(git_output("cat-file", "-t", tag) == "tag" for tag in tags)


def tracked_tree_clean() -> bool:
    completed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=ROOT,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return completed.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local safety and public release readiness.")
    parser.add_argument("--output", type=Path, default=ROOT / "release-audit.json")
    args = parser.parse_args()

    files = tracked_files()
    binary_findings = [str(path.relative_to(ROOT)) for path in files if path.suffix.lower() in FORBIDDEN_SUFFIXES]
    secret_findings: list[str] = []
    private_path_findings: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = str(path.relative_to(ROOT))
        if SECRET_PATTERN.search(text):
            secret_findings.append(relative)
        if PRIVATE_PATH_PATTERN.search(text):
            private_path_findings.append(relative)

    tests_passed, test_count, test_tail = run_tests()
    expected = [
        "README.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "docs/REPRODUCIBILITY.md",
        "docs/REMOVAL.md",
        "configs/model-artifacts.example.json",
        "configs/model-artifacts.lock.json",
        "requirements/full-py311.lock.txt",
        "evidence/synthetic-integration.json",
        "evidence/live-integration.json",
        "evidence/application-workflows.json",
    ]
    missing_expected = [name for name in expected if not (ROOT / name).is_file()]
    local_findings = {
        "forbidden_binary_or_audio_files": binary_findings,
        "possible_secrets": secret_findings,
        "private_absolute_paths": private_path_findings,
        "missing_expected_files": missing_expected,
        "tests_passed": tests_passed,
        "test_count": test_count,
    }
    local_passed = not any((binary_findings, secret_findings, private_path_findings, missing_expected)) and tests_passed

    release_blockers: list[str] = []
    if not (ROOT / "LICENSE").is_file():
        release_blockers.append("No author-approved code license file.")
    repository_root = git_output("rev-parse", "--show-toplevel")
    if repository_root is None or Path(repository_root).resolve() != ROOT.resolve() or not has_annotated_head_tag():
        release_blockers.append("No version-control repository identity or immutable release tag.")
    if not tracked_tree_clean():
        release_blockers.append("Tracked files differ from the tagged source revision.")
    if not live_evidence_complete(ROOT / "evidence/live-integration.json"):
        release_blockers.append(
            "No completed consented microphone-to-VAD/ASR and audible playback evidence record."
        )
    if not application_evidence_complete(ROOT / "evidence/application-workflows.json"):
        release_blockers.append("No completed application-workflow evidence record.")
    if not git_output("remote", "get-url", "origin"):
        release_blockers.append("No author-approved public remote destination.")

    report = {
        "schema_version": "companion-release-audit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root_name": ROOT.name,
        "files_scanned": len(files),
        "local_safety": {"passed": local_passed, "findings": local_findings},
        "public_release": {"ready": local_passed and not release_blockers, "blockers": release_blockers},
        "test_output_tail": test_tail,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "local_safety_passed": local_passed, "public_release_ready": report["public_release"]["ready"], "test_count": test_count}))
    return 0 if local_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
