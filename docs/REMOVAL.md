# Removal and Data Cleanup

Application data uses the operator-selected roots; the standard fetcher uses the project-local `models/` directory. Research acquisition helpers and third-party installers can also use their configured Hugging Face, package or Docker caches. Inspect those locations separately before cleanup; they may be shared with other projects.

## Remove recordings and generated speech

Stop any active command. Resolve the exact capture or output path from its command result, confirm it is the intended file, and delete that file. Review managed archive and dictation records before deletion because their undo and retention semantics differ.

Do not run recursive cleanup against a user profile, workspace root, drive root, or shared model directory.

For `archive-workflow`, prefer `archive-delete SOURCE_SHA256 --root MANAGED_ROOT`; it removes the digest-named managed WAV and its paired transcript record. It does not remove the original source file. For reviewed dictation, `dictation-undo` succeeds only while the destination still matches the committed digest. Narration projects are self-contained below the explicit `--root`; verify that exact project root before removing its chunks, master WAV, and manifest.

## Remove model artifacts

The default artifacts are confined to these ignored project-local paths:

- `models/vad/silero-vad-6.2.3/`
- `models/asr/faster-whisper-base.en/`
- `models/tts/Kokoro-82M/`

Stop model processes, verify the resolved paths remain inside this repository's `models/` directory, and remove only the specific directories required. Downloader metadata below those directories may be removed with the corresponding snapshot. The checked-in manifest and fetch script remain so the artifacts can be reproduced.

## Remove the environment and package

For interpretation studies, stop the exact named container you created and inspect it before removal. Removing a container does not remove its downloaded image or the model directory mounted into it. Do not remove a shared image, model cache or unrelated service merely to clean this project.

For LocalLab, inspect pending requests and retain any needed results before removing its exact owned root and SQLite files. Archive deletion does not remove source inputs; dictation cleanup also needs the destination and its undo records; narration cleanup includes chunks, master and manifests. File deletion is not a guarantee of erasure from backups or storage media.

Deactivate and remove the project-specific virtual environment, or use that environment's package uninstaller. Removing the environment does not remove recordings, generated speech, archives, model snapshots, or evidence stored elsewhere.

## Sensitive data

If a recording, transcript, credential, or private path is accidentally committed, treat it as exposed. Stop public release, rotate affected credentials, remove the material from active use, and follow the hosting provider's history-removal process. A later deletion commit alone does not erase prior repository history.
