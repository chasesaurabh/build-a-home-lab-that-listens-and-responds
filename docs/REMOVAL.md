# Removal and Data Cleanup

The companion writes only to paths explicitly selected by the operator and to the project-local `models/` directory used by the fetcher.

## Remove recordings and generated speech

Stop any active command. Resolve the exact capture or output path from its command result, confirm it is the intended file, and delete that file. Review managed archive and dictation records before deletion because their undo and retention semantics differ.

Do not run recursive cleanup against a user profile, workspace root, drive root, or shared model directory.

## Remove model artifacts

The default artifacts are confined to these ignored project-local paths:

- `models/vad/silero-vad-6.2.3/`
- `models/asr/faster-whisper-base.en/`
- `models/tts/Kokoro-82M/`

Stop model processes, verify the resolved paths remain inside this repository's `models/` directory, and remove only the specific directories required. Downloader metadata below those directories may be removed with the corresponding snapshot. The checked-in manifest and fetch script remain so the artifacts can be reproduced.

## Remove the environment and package

Deactivate and remove the project-specific virtual environment, or use that environment's package uninstaller. Removing the environment does not remove recordings, generated speech, archives, model snapshots, or evidence stored elsewhere.

## Sensitive data

If a recording, transcript, credential, or private path is accidentally committed, treat it as exposed. Stop public release, rotate affected credentials, remove the material from active use, and follow the hosting provider's history-removal process. A later deletion commit alone does not erase prior repository history.
