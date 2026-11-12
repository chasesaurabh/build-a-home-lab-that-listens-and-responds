# Build a Home Lab That Listens and Responds

Runnable companion code for Saurabh Chase’s practical guide to local speech recognition, generated speech and controlled audio applications. For the first-edition code, use [book-v1.0.0](https://github.com/chasesaurabh/build-a-home-lab-that-listens-and-responds/tree/book-v1.0.0). Version 2.0.0 adds the research and application work described below.

Start with model-free controls, then add a pinned CPU speech stack. Separate research environments compare recognition models and small-to-large local text models. The repository includes code, setup instructions, original test cases and reviewable research results. It excludes the manuscript, covers, personal recordings, generated audio and model weights.

## What you can run

- Inspect and normalize audio; explicitly select a device for timed recording or playback.
- Detect speech with Silero, transcribe with Faster-Whisper and generate a WAV with Kokoro’s stock `af_heart` voice.
- Import and search a small transcript archive; delete its managed audio/transcript pair.
- Preview dictation, approve an exact replacement and undo it only when the destination still matches.
- Plan narration chunks, generate them and assemble a checked PCM master.
- Propose local status or diagnostic operations from typed or transcribed input. A persistent local diagnostic requires a separate exact-digest confirmation.
- Reproduce the declared ASR, speaker-grouping, audio-event and text-interpretation studies.

This is a CLI and Python reference implementation. It has no always-listening wake-word service, graphical assistant, live barge-in, meeting transcription service, remote-server administration or voice-based authentication. Model-free cancellation invalidates control state; it does not establish when hardware becomes silent.

## Start without models

Use Python 3.11 and a short project path on Windows. The full speech environment contains packages with deeply nested installation paths.

```powershell
py -3.11 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install --no-deps -e .
& ./.venv/Scripts/python.exe -m unittest discover -s tests -v
& ./.venv/Scripts/python.exe -m home_lab_voice --help
```

The standard tests require no models, device access, network or GPU. Counts and source identity belong to the exact release result; a pass covers the exercised cases, not every possible input.

Try a real local queue and diagnostic. Choose a new private root; the application rejects an existing unowned directory.

```powershell
& ./.venv/Scripts/python.exe -m home_lab_voice lab --root my-lab init
& ./.venv/Scripts/python.exe -m home_lab_voice lab --root my-lab status --service voice-service
& ./.venv/Scripts/python.exe -m home_lab_voice lab-turn --root my-lab `
  --text "Run healthcheck for voice-service." --idempotency-key reader-healthcheck-001
```

The last command creates a pending request and exits with status 3. Inspect its request ID and digest. After reviewing the exact operation, use `lab --root my-lab confirm --request-id REQUEST_ID --digest DIGEST`. Repeat delivery returns the recorded result; changed or expired requests are rejected. These are local SQLite-backed diagnostics, not checks of unrelated services on your machine. The older `assistant` and `assistant-turn` examples use in-memory providers and have different persistence boundaries.

## Add the CPU speech stack

The Windows x86-64 Python 3.11 lock is separate from the model-free install:

```powershell
& ./.venv/Scripts/python.exe -m pip install -c requirements/full-py311.lock.txt ".[full]"
& ./.venv/Scripts/python.exe scripts/fetch_models.py
& ./.venv/Scripts/python.exe scripts/fetch_models.py --verify-only
```

Acquisition needs Internet access. It fetches the allowlisted files at immutable revisions, outside Git. Subsequent VAD, CPU ASR and TTS commands use local artifacts. The optional interpretation adapter instead calls a loopback model server; it does not send requests to a hosted provider.

Create and inspect an original synthetic command without recording anyone:

```powershell
& ./.venv/Scripts/python.exe -m home_lab_voice synthesize outputs/command-24k.wav `
  --model-dir models/tts/Kokoro-82M --text "Get status for voice-service."
& ./.venv/Scripts/python.exe -m home_lab_voice normalize outputs/command-24k.wav `
  outputs/command-16k.wav
& ./.venv/Scripts/python.exe -m home_lab_voice transcribe outputs/command-16k.wav `
  --model-dir models/asr/faster-whisper-base.en --language en
```

Generated speech is a useful integration fixture, not a representative human-speech benchmark. Device capture and playback are separate deliberate commands: inspect `devices`, `capture`, `listen` and `play` help before using them. Recording requires an explicit device and declared permission. No audio is automatically deleted merely because a retention label was supplied.

## Four application paths

| Application | Commands | Important boundary |
|---|---|---|
| Archive | `archive-workflow`, `archive-search`, `archive-delete` | Small literal-substring search; deleting a managed pair preserves the original source |
| Dictation | `dictation-preview`, `dictation-commit`, `dictation-undo` | Preview exits 3; exact approval; single writer; file and receipt are not crash-atomic |
| Narration | `narration`, `narration-build` | Inspect chunk boundaries; no automatic resume or audio mastering |
| Assistant | `lab`, `lab-turn` | Local queue/status/diagnostics; explicit confirmation outside the model |

See [reproduction instructions](docs/REPRODUCIBILITY.md), [research commands](docs/RESEARCH.md), [research evidence](evidence/research/README.md), [security boundary](SECURITY.md) and [removal guidance](docs/REMOVAL.md). Run each command with `--help` for its required arguments. Use a new output directory for each research run so earlier evidence remains intact.

## Models and rights

The companion code is MIT-licensed. Downloaded models, runtime packages, voices and data have their own terms. [Third-party notices](THIRD_PARTY_NOTICES.md) identify the components. LibriSpeech-derived reference material retains CC BY 4.0 attribution; model weights and audio are not distributed here.

Do not commit private recordings, transcripts, credentials, local environment files, model weights or output folders. Research and diagnostic records can include input text and machine paths; inspect them before sharing. A model proposal is never proof of permission.
