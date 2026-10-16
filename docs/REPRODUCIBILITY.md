# Reproducibility

The companion has two verification lanes: a dependency-free deterministic suite and an optional pinned Windows CPU speech stack. Keep their results separate.

## Deterministic clean-room verification

From the repository root on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install --no-deps -e .
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python scripts\audit_release.py --output release-audit.json
```

The tests must pass and the audit must report `local_safety.passed` as true. This lane does not open devices, download models, use a GPU, or contact a service.

## Pinned Windows CPU speech lane

Use CPython 3.11 on Windows x86-64 for the frozen dependency lock:

```powershell
.\.venv\Scripts\python -m pip install -c requirements\full-py311.lock.txt ".[full]"
.\.venv\Scripts\python scripts\fetch_models.py
.\.venv\Scripts\python scripts\fetch_models.py --verify-only
```

The fetcher uses immutable revisions and an allowlist of required files. `configs/model-artifacts.lock.json` records source, revision, canonical file or tree digest, license, redistribution decision, and local path. A tree digest hashes each relative path and content digest in sorted order while excluding downloader cache metadata.

Run normal model commands with network access disabled after acquisition. The VAD adapter opens the pinned ONNX path, Faster-Whisper uses `local_files_only=True`, and Kokoro receives explicit local config, weights, and voice paths.

## Evidence record

For every live run, record:

1. source permission and retention class;
2. source digest, rate, channels, duration, and selected device;
3. package lock, artifact manifest digest, and source revision;
4. CPU model, device mode, numeric precision, and thread settings;
5. exact command, start time, elapsed time, exit status, and raw structured output;
6. expected text or labels and the scored result;
7. whether listening or subjective review was performed;
8. cleanup path and completion.

Never place a personal recording or private path in the public evidence directory. Public evidence should contain only redacted measurements, artifact identities, and reproducible commands.

## Evidence limits

Passing the deterministic suite proves the exercised contracts and control paths. Passing one device/model integration proves only that exact host, artifact, configuration, and fixture. It does not establish universal accuracy, noise robustness, far-field behavior, accessibility, or legal suitability for every use.
