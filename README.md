# Build a Home Lab That Listens and Responds

Companion code for the book by Saurabh Chase.

## The book

This repository accompanies *Build a Home Lab That Listens and Responds: Create Private Voice Assistants, Speech Recognition, and Audio Applications with Open Models*. The publication link will be added here after the book is published.

The project provides a tested local control plane plus optional CPU speech runtimes. Operational audio stays on the machine. The repository does not contain the manuscript, recordings, credentials, model weights, or generated speech.

## Verified scope

- Versioned audio, intent, speech, consent, retention, and idempotency contracts
- Deterministic PCM WAV fixtures, inspection, malformed-input rejection, and WER accounting
- Bounded endpointing, conversation states, cancellation, recovery, and confirmation
- Managed archive, reviewed dictation, narration planning, and controlled-assistant examples
- Live audio-device inventory, explicit device selection, bounded capture, and bounded playback
- Silero VAD 6.2.3 through a pinned CPU ONNX artifact
- Faster-Whisper 1.2.1 with a pinned `base.en` snapshot on CPU/int8
- Kokoro 0.9.4 with a pinned Kokoro-82M snapshot and `af_heart` voice on CPU
- Immutable model revisions, local artifact digest verification, and model-free unit doubles
- A standard deterministic suite that does not need a microphone, models, network, or GPU

The standard suite verifies code behavior. Separate integration evidence records the exact device and model lanes exercised for the book; it does not imply universal speech accuracy or voice quality.

## Run the deterministic suite

From the repository root with Python 3.11 or later:

```powershell
py -m unittest discover -s tests -v
```

The exact count is recorded by the release audit rather than duplicated here.

## Install the lightweight control plane

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install --no-deps -e .
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The model-free commands remain useful without the full speech stack:

```powershell
home-lab-voice fixture lab-fixture.wav --duration 0.1
home-lab-voice archive lab-fixture.wav --root .\demo\archive
home-lab-voice dictation --root .\demo\dictation --target notes.txt --text "Reviewed reference text."
home-lab-voice narration permitted-source.txt --output .\demo\narration-plan.json
home-lab-voice assistant status --service voice-service
```

## Install the verified Windows CPU speech lane

The checked-in lock was resolved and integration-tested on Windows x86-64 with CPython 3.11.9:

```powershell
.\.venv\Scripts\python -m pip install -c requirements/full-py311.lock.txt ".[full]"
.\.venv\Scripts\python scripts/fetch_models.py
.\.venv\Scripts\python scripts/fetch_models.py --verify-only
```

`fetch_models.py` obtains only the allowlisted files at the immutable revisions in `configs/model-artifacts.lock.json`. It verifies the standalone Silero file during download and reports canonical tree hashes for the Faster-Whisper and Kokoro directories. The `models/` directory is ignored by Git and must not be committed.

Inspect devices before recording. Select an explicit index for a one-off session, or use the stable
`HOST_API::DEVICE_NAME` form when Windows exposes the same device through multiple audio systems:

```powershell
home-lab-voice devices --direction input
home-lab-voice devices --direction output
```

Capture is limited to 30 seconds and refuses replacement unless `--overwrite` is present:

```powershell
home-lab-voice capture .\demo\self-recorded.wav `
  --device "Windows WDM-KS::Microphone (HyperX QuadCast S)" `
  --duration 5 --consent-basis self_recorded
```

Run each pinned runtime independently:

```powershell
home-lab-voice vad .\demo\self-recorded.wav `
  --model-file .\models\vad\silero-vad-6.2.3\silero_vad.onnx

home-lab-voice transcribe .\demo\self-recorded.wav `
  --model-dir .\models\asr\faster-whisper-base.en --language en

home-lab-voice synthesize .\demo\response.wav `
  --model-dir .\models\tts\Kokoro-82M `
  --voice af_heart --text "The private voice service is ready."

home-lab-voice play .\demo\response.wav `
  --device "Windows WDM-KS::Speakers (HyperX QuadCast S)"
```

Or capture, detect speech, and transcribe one bounded turn:

```powershell
home-lab-voice listen .\demo\turn.wav `
  --device "Windows WDM-KS::Microphone (HyperX QuadCast S)" --duration 8 `
  --vad-model-file .\models\vad\silero-vad-6.2.3\silero_vad.onnx `
  --model-dir .\models\asr\faster-whisper-base.en --language en
```

The full verified lane is deliberately CPU-only. It neither needs nor selects a GPU. Model acquisition is the only networked runtime setup step; normal VAD, ASR, and TTS operation uses local files.

## Safety and release boundaries

- Use only audio you are permitted to capture, process, and retain.
- Prefer headphones for capture-plus-playback tests to prevent acoustic feedback.
- Do not commit `models/`, `recordings/`, `outputs/`, generated audio, or private evaluation records.
- Treat transcripts as fallible observations, never as authorization.
- Keep consequential actions behind allowlists, session checks, and operation-bound confirmation.
- Review `THIRD_PARTY_NOTICES.md` and the locked artifact manifest before every public release.

The diagnostic assistant first returns an operation digest and exits without running the action. Review it, then repeat the same command with `--confirm DIGEST`. Its providers are safe local doubles and do not contact home-lab services.

## Reproducibility and removal

- `configs/model-artifacts.lock.json` freezes the enabled model and voice artifacts.
- `configs/model-artifacts.example.json` is a deliberately empty starting template.
- `docs/REPRODUCIBILITY.md` defines clean-room verification.
- `docs/REMOVAL.md` explains removal of generated data, models, environments, and the package.
- `SECURITY.md` documents reporting and the threat boundary.
- `THIRD_PARTY_NOTICES.md` records direct dependencies and artifact licensing boundaries.

Run the local release audit with:

```powershell
py scripts/audit_release.py --output release-audit.json
```

Local safety and public-release readiness are reported separately. Release readiness requires the approved license and remote plus an annotated tag on the audited commit.
