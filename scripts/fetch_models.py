from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from home_lab_voice.contracts import ContractError  # noqa: E402
from home_lab_voice.model_artifacts import artifact_sha256  # noqa: E402
from home_lab_voice.runtime_models import (  # noqa: E402
    SILERO_VAD_ONNX,
    choose_snapshots,
    fetch_direct_file,
    fetch_snapshot,
    verify_direct_file,
    verify_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch or verify immutable CPU speech-model snapshots.")
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument("--purpose", choices=("vad", "asr", "tts"), action="append")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    purposes = args.purpose or ["vad", "asr", "tts"]
    try:
        reports = []
        if "vad" in purposes:
            destination = (
                verify_direct_file(args.models_root, SILERO_VAD_ONNX)
                if args.verify_only
                else fetch_direct_file(args.models_root, SILERO_VAD_ONNX)
            )
            reports.append(
                {
                    "name": SILERO_VAD_ONNX.name,
                    "purpose": SILERO_VAD_ONNX.purpose,
                    "source_url": SILERO_VAD_ONNX.source_url,
                    "revision": SILERO_VAD_ONNX.revision,
                    "local_path": destination.resolve().as_posix(),
                    "sha256": artifact_sha256(destination),
                    "license_spdx": SILERO_VAD_ONNX.license_spdx,
                }
            )
        for spec in choose_snapshots(purposes):
            destination = args.models_root / spec.directory
            if args.verify_only:
                verify_snapshot(destination, spec)
            else:
                destination = fetch_snapshot(args.models_root, spec)
            reports.append(
                {
                    "name": spec.name,
                    "purpose": spec.purpose,
                    "repository": spec.repository,
                    "revision": spec.revision,
                    "local_path": destination.resolve().as_posix(),
                    "tree_sha256": artifact_sha256(destination),
                    "license_spdx": spec.license_spdx,
                }
            )
    except ContractError as error:
        print(json.dumps({"status": "error", "code": error.code, "message": error.safe_message}))
        return 2
    print(json.dumps({"status": "verified" if args.verify_only else "fetched", "models": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
