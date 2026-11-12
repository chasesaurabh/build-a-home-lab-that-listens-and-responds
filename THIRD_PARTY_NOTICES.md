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

The authoritative machine-readable record is `configs/model-artifacts.lock.json`. A model license does not license input recordings, generated text, or third-party datasets, and the companion's MIT code license does not relicense these artifacts.

## Redistribution

The MIT license covers this companion's original code. Before redistributing a modified application or a complete environment, inspect the upstream notices and model cards at the revisions you use and retain their applicable notices. Downloading an artifact does not transfer rights to input recordings or voices. This inventory does not replace those terms.


## Research components and data

The separate environment locks record installed packages for the recognition, events and speaker studies. Additional direct components include TensorFlow 2.20 (Apache-2.0), SpeechBrain 1.1.2 (Apache-2.0), SoundFile (BSD-3-Clause), PyArrow (Apache-2.0) and vLLM (Apache-2.0). The pinned server image is an upstream distribution; its bundled components retain their own notices. NVIDIA runtime libraries obtained through optional packages are governed by NVIDIA’s terms, not this repository’s MIT license. No installed environment, vendor runtime or container image is redistributed here.

Additional downloaded models are SYSTRAN Faster-Whisper tiny.en, small.en and large-v3 (MIT model repositories), SpeechBrain ECAPA VoxCeleb (Apache-2.0), Google YAMNet (Apache-2.0), Qwen3.5-2B and Qwen3.5-27B (Apache-2.0), and NVIDIA Qwen3.5-122B-A10B-NVFP4 (Apache-2.0). Exact revisions and file identities are in acquisition recipes and evidence. Downloaded model weights, configurations and voices remain outside Git.

LibriSpeech-derived reference text and fixture annotations in `evidence/research` retain CC BY 4.0. See its README for full attribution, source revision and declared transformations. Original authored intent cases and synthetic-signal generators use the repository’s MIT license. No LibriSpeech audio is bundled.
