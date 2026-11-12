# Research evidence

These records support the companion’s declared studies. They contain public read-speech references, original synthetic-case definitions, model outputs, settings and measurements. No model weights, audio recordings or generated WAV files are included.

## Coverage

- Five ASR configurations: 192 first-pass evaluation clips each, plus 72 repeated timing requests each. `asr-comparison.json` aggregates word edits and declares the nearest-rank timing percentile.
- Closed grammar and three local text models: 64 evaluation cases, with model warm repeats separate. Earlier 2B development attempts remain visible. All 300 recorded final-model requests were checked against the same stored system prompt.
- Twenty-six original non-speech signals, split between development and evaluation. They test a bounded synthetic-negative scenario, not room-noise prevalence.
- YAMNet output on 256 selected LibriSpeech clips and 26 synthetic signals. Full class means and selected-label decisions are retained.
- Two stitched four-reader fixtures with supplied grouping counts 2, 4 and 6. The labelled-time error measure excludes unknown regions and is not diarization error rate or speech-activity recall.
- Six original generated commands through GPU ASR, a resident large text model and CPU speech output. This is a sequential integration/capacity check, not concurrency or population accuracy evidence.

The selection manifests identify the exact input bytes and source revisions. `export-manifest.json` records source and exported hashes. JSON was serialized for portability; a small number of absolute local paths were replaced by an explicit omission marker. Measurements, expected labels and model responses were preserved. Executed source snapshots are included where suitable for redistribution; current runnable acquisition and study code is in `experiments/`.

## Rights and attribution

LibriSpeech reference text and derived fixture annotations retain the dataset’s [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) terms. Attribution: Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, *LibriSpeech: An ASR Corpus Based on Public Domain Audio Books*, ICASSP 2015; source recordings from LibriVox. Dataset source: [OpenSLR 12](https://www.openslr.org/12), acquired through [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) at revision `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`.

Changes from that dataset are selection, format extraction, deterministic stitching with inserted gaps, generated recognition hypotheses and evaluation annotations. The repository does not include its audio. Model-generated hypotheses are identified as outputs, not corrected reference truth. Original authored intent cases and synthetic-signal generators are covered by the repository’s MIT license. That license does not supersede third-party dataset or model terms.

## Interpretation limits

The ASR clips are English audiobooks. The intent set is deliberately balanced by behavior rather than sampled from household use. No training-data decontamination claim is made. A zero count on a small selected set does not establish zero future risk. Model size, architecture and precision change together in the interpretation comparison.

Warm repetitions and later reproductions reuse cases. Device memory is sampled total allocation, not an exact peak or minimum capacity. The largest-model evaluation overlapped CPU setup work; do not use its wall time as an isolated benchmark. Older device evidence elsewhere in `evidence/` is dated separately and is not a new microphone or listening study.
