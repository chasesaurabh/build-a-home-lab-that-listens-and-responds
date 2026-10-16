# Security Policy

## Supported versions

There is no supported public release yet. The private release candidate is maintained while live evidence, license approval, repository identity, and an immutable tag are completed.

## Reporting a vulnerability

Report suspected vulnerabilities privately to the project owner. Do not include credentials, personal recordings, transcripts, model files, or unredacted machine paths. A public security contact will be added when a public repository is approved.

## Threat boundary

The base package is model-free and network-free. The optional full lane adds three explicit boundaries:

- device access through sounddevice/PortAudio;
- artifact acquisition through the model fetch script;
- local CPU inference through Silero VAD, Faster-Whisper, and Kokoro.

Operational model commands are designed for local files and do not need network access. They do not provide operating-system sandboxing, service authentication, speaker identity, or authorization from speech.

The implementation preserves:

- explicit consent and retention metadata for capture;
- explicit device selection and bounded capture/playback durations;
- bounded audio, text, decoding, and output parameters;
- allowlisted actions and confirmation for consequential operations;
- immutable revisions and digest verification for external artifacts;
- local-only CPU model configuration in the verified lane;
- safe structured errors without credentials or transcripts;
- ignored model, recording, output, and environment paths.

The sample assistant providers are local test doubles, not production authentication or authorization controls. A transcript must never be treated as proof of identity or permission.
