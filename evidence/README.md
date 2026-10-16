# Integration Evidence

`synthetic-integration.json` is the public-safe summary of the pinned local CPU smoke test. `live-integration.json` records the consented HyperX microphone-to-VAD/ASR run and the operator-confirmed C34J79x playback run. Both records contain measurements and hashes without including generated audio, personal recordings, model files, or absolute machine paths.

The synthetic loop proves that the selected Kokoro snapshot can generate PCM audio, the selected Faster-Whisper snapshot can transcribe that controlled sentence exactly, and the selected Silero ONNX artifact treats a pure tone as a negative control. The live record proves one exact consented device and model configuration; it does not establish noisy-room, far-field, universal-accuracy, or application-specific performance.

Private raw command output and disposable WAV files belong outside the repository. Public release evidence may record their digests and redacted results only.
