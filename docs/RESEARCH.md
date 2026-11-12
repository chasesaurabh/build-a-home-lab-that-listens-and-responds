# Reproduce the research

The studies have separate environments because speech applications, TensorFlow event scoring, SpeechBrain grouping and CUDA recognition use different dependencies. Use Python 3.11 on Windows x86-64 for these locks. Create each environment separately and invoke its interpreter explicitly. Do not combine every lock in one environment.

## Environments

| Lane | Requirements file | Purpose |
|---|---|---|
| CPU applications | `full-py311.lock.txt` | Silero, base.en and Kokoro |
| ASR study | `asr-study-py311-windows.lock.txt` | Acquisition and CPU study |
| CUDA ASR | `asr-gpu-py311-windows.lock.txt` | Recognition with NVIDIA wheel libraries |
| Audio events | `events-py311-windows.lock.txt` | TensorFlow SavedModel inference |
| Speaker groups | `speakers-py311-windows.lock.txt` | SpeechBrain ECAPA and Silero |

For example, create `.venv-asr`, install with `python -m pip install -r requirements/asr-gpu-py311-windows.lock.txt`, then `python -m pip install --no-deps -e .` using that environment’s interpreter. CUDA needs a compatible NVIDIA driver; a Python package alone cannot supply one. The interpretation server uses the pinned Docker image in `start_interpretation_server.py`, Docker Desktop’s WSL2 GPU support and verified model files.

## Recognition

Run from the repository root using the ASR environment:

```powershell
python experiments/prepare_librispeech_parquet.py --root outputs/librispeech
python experiments/fetch_asr_models.py --root models/asr-study
python experiments/run_asr_study.py --corpus outputs/librispeech `
  --models models/asr-study --model base.en --device cpu --compute-type int8 `
  --split evaluation --warm-repeats 3 --output outputs/asr-base-cpu
```

The generator pins the dataset revision and selection rule. It downloads the source parquet data and writes selected audio plus a manifest. The acquisition script downloads all four listed ASR models; allow sufficient storage and download time. The selected evaluation partition contains 192 clips. `--limit` changes the evaluated set and must not be described as the full study.

For the four GPU runs, choose `tiny.en`, `base.en`, `small.en` or `large-v3`, use `--device cuda --compute-type float16 --nvidia-wheels`, and give each run a new output directory. Keep all other settings unchanged. The harness fixes English, beam 5, temperature zero, no previous-text conditioning and no VAD filtering. Warm repeats cover 24 selected clips; they are not extra independent accuracy cases.

## Interpretation

The original 24 development and 64 evaluation cases can be generated without models:

```powershell
python experiments/prepare_interpretation_cases.py --output outputs/intent-cases.json
python experiments/run_interpretation_study.py --cases outputs/intent-cases.json `
  --split evaluation --model grammar --baseline --output outputs/grammar
python experiments/fetch_interpretation_model.py --size 2b --root models/interpretation
```

The fetcher writes `2b-acquisition.json` beneath the selected root. Its `model` and `revision` fields identify the directory: replace the slash in the model repository name with `--`, then append the revision. For this 2B artifact the path is `models/interpretation/Qwen--Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc`. Use that exact verified directory for `--model-path`. Pull the pinned image before starting a server:

```powershell
docker pull vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
python experiments/start_interpretation_server.py --name voice-study-2b `
  --model-path YOUR_VERIFIED_MODEL_DIRECTORY --alias voice-2b `
  --gpu-fraction 0.09 --output outputs/start-2b
python experiments/run_interpretation_study.py --cases outputs/intent-cases.json `
  --split evaluation --model voice-2b --warm-repeats 3 --output outputs/intent-2b
docker stop voice-study-2b
```

`YOUR_VERIFIED_MODEL_DIRECTORY` is an explicit substitution, not a literal path. The starter refuses an existing container name, binds only `127.0.0.1:18105`, mounts the model read-only and records readiness. Inspect failed startup and stop only the named container you created. Do not stop unrelated services.

The 27B and 122B artifact choices use `--size 27b` and `--size 122b`. Recorded GPU fractions were 0.68 and 0.88. The 122B conversion additionally used `--quantization modelopt_fp4 --kv-cache-dtype fp8`. Context was 4096, one sequence, eager mode, text-only input and 512 output tokens. These fractions describe the recorded workstation configuration, not minimum hardware requirements. Run one text server at a time and verify available memory before attempting a larger artifact.

The three final model runs used the same prompt and schema. Earlier failed development output is retained separately. No interpretation study executes tools. A fresh download and rerun of the small model used the same cases and is a reproduction check, not a new holdout.

## Non-speech, events and speaker groups

Use the CPU application environment to run `run_nonspeech_study.py --models-root models --output outputs/nonspeech`. It generates original signals, saves fixtures and compares ungated and VAD-gated recognition. Synthetic negatives do not estimate household false-alarm rates.

Use the event environment for `fetch_yamnet.py --output models/yamnet` and `run_event_study.py --librispeech outputs/librispeech --nonspeech outputs/nonspeech --acquisition models/yamnet/acquisition.json --output outputs/events`. Check each helper’s `--help`. The event study saves all class means and top labels; selected speech success does not establish general sound detection quality.

Use the speaker environment for `fetch_speaker_model.py --output models/ecapa` and `prepare_speaker_fixtures.py --output outputs/speaker-fixtures`. Then run `run_speaker_study.py --fixtures outputs/speaker-fixtures --weights models/ecapa/embedding_model.ckpt --vad-model models/vad/silero-vad-6.2.3/silero_vad.onnx --output outputs/speakers`. The VAD file comes from the standard model fetcher. The two fixtures contain four known audiobook readers each. Supplying two or six groups is a sensitivity experiment; its error metric is not meeting diarization error rate.

## Compare records, not impressions

Keep raw predictions, settings, acquisition hashes and environment locks. [Published research evidence](../evidence/research/README.md) explains denominators, timing limitations and which records have local paths omitted. GPU memory samples include display and runtime allocations. The largest-model timing overlapped unrelated CPU installation work and is not an isolated speed comparison. The six-case simultaneous-residency check is a capacity/integration observation, not a recommendation to operate with very little free memory.

Use your own permitted recordings for application acceptance tests. Do not publish them by default. Re-running the supplied data can verify reproduction; it cannot establish new generalization evidence.
