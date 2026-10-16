# Third-Party Notices

This repository distributes companion source and manifests. It does not bundle model weights, voices, recordings, generated audio, or datasets. Installing optional extras obtains third-party packages under their own terms; running the fetcher obtains the separately listed model artifacts.

## Direct optional runtime dependencies

| Component | Pinned version | Role | Reported license |
|---|---:|---|---|
| sounddevice | 0.5.6 | Device capture and playback | MIT |
| faster-whisper | 1.2.1 | ASR adapter | MIT |
| silero-vad | 6.2.3 | VAD API and packaged runtime support | MIT |
| onnxruntime | 1.30.0 | CPU ONNX inference | MIT |
| Kokoro Python library | 0.9.4 | TTS inference | Apache-2.0 |
| en-core-web-sm | 3.8.0 | English linguistic model used by Misaki | MIT |

The full Python dependency graph and versions are frozen in `requirements/full-py311.lock.txt`. Significant transitive components include CTranslate2 (MIT), PyAV (BSD-3-Clause), Hugging Face Hub (Apache-2.0), PyTorch (multiple permissive notices reported in its package metadata), spaCy (MIT), Misaki (Apache-2.0), phonemizer-fork (GPL-3.0-only), num2words (LGPL), and the espeak-ng distribution loaded for fallback phonemization. Review the license files shipped in the installed distributions before redistributing an environment or executable bundle.

On Windows, the sounddevice wheel includes PortAudio under its upstream permissive license. Device access does not grant permission to record or redistribute audio.

## Model and voice artifacts

| Artifact | Immutable revision | Locked digest | Reported license | Redistribution |
|---|---|---|---|---|
| Silero VAD ONNX opset 16 | `5cd7945676eb32225748052e2e6a0580e4686a08` | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` | MIT | Not bundled |
| SYSTRAN Faster-Whisper `base.en` snapshot | `3d3d5dee26484f91867d81cb899cfcf72b96be6c` | `f858b3ee8ea62c5a63132d4493583a42f2900b8168df7bc13184ab96594786af` tree digest | MIT | Not bundled |
| Kokoro-82M config, v1.0 weights, and `af_heart` voice | `f3ff3571791e39611d31c381e3a41a3af07b4987` | `7d2b8a58772454d049efa0ad8911c963c510349efe911aa2d5c14b2c25b836e3` tree digest | Apache-2.0 | Not bundled |

The authoritative machine-readable record is `configs/model-artifacts.lock.json`. A model license does not license input recordings, generated text, or third-party datasets, and the companion's future code license will not relicense these artifacts.

## Release obligation

Before any public release, regenerate the dependency and artifact inventories, inspect upstream notices and model cards at the pinned revisions, confirm that no third-party binaries entered the wheel or repository, and retain applicable notices. This file is an engineering inventory, not legal advice.
