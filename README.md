# Five-Dialect Amharic ASR

Reproducible research code for evaluating and fine-tuning multilingual Whisper for Amharic speech from Addis Ababa, Gojjam, Gonder, Shewa, and Wello varieties.

The training path uses `WhisperProcessor`, `WhisperForConditionalGeneration`, and `Seq2SeqTrainer` from Hugging Face Transformers, which is the supported fine-tuning path. OpenAI's [`whisper`](https://github.com/openai/whisper) repository is the upstream reference implementation for Whisper inference and preprocessing, not a complete fine-tuning trainer.

> **Research status:** this repository implements the workflow; it does not claim that the source data has passed native-speaker review or that all licenses permit redistribution. The Gojjam license must be clarified before public model release.

## What is included

- Immutable-source audit with archive hash, Parquet schema, audio-header, transcript, duplicate, and speaker-overlap checks.
- Derived-data preparation with explicit review flags, provenance, real 16 kHz mono resampling, and aligned segmentation.
- Speaker- and prompt-safe split generation with leakage reports.
- Unchanged Whisper baseline evaluation with Amharic task/language settings.
- Transformers fine-tuning, smoke-test overfit, generation-based evaluation, per-dialect WER/CER, and inference.
- A Colab guide that runs from clone through Phase 10 without requiring the 15 GB archive to be committed.

## Repository layout

```text
configs/default.yaml                 reproducible defaults
src/amharic_asr/                     importable pipeline modules
scripts/                              command-line entry points
notebooks/                            Colab starter notebook
docs/                                 review protocol and release checklist
data_manifests/                      generated manifests (data stays out of Git)
tests/                               lightweight unit tests
COLAB_GUIDE.md                       clone-to-finish Colab instructions
```

## Quick start

```bash
git clone --branch v2 --single-branch https://github.com/ethcocoder/paravoice.git
cd paravoice
pip install -e .
python scripts/audit_dataset.py --help
```

The Colab workflow must use the `v2` branch. The `--single-branch` option prevents accidentally running an older or unrelated branch.

The pipeline expects either the verified ZIP from the roadmap or a directory of Parquet shards. Set paths in `configs/default.yaml` or pass command-line overrides. Never commit source audio, credentials, Hugging Face tokens, or generated model weights.

For a Colab run, the verified source can be cloned directly from Hugging Face with Git LFS:

```bash
apt-get -qq update && apt-get -qq install -y git-lfs
git lfs install
git clone https://huggingface.co/datasets/paradoxquantum/paradox-amharic-data /content/asr-work/source/paradox-amharic-data
git -C /content/asr-work/source/paradox-amharic-data lfs pull
```

The archive is about 15.84 GB. Verify SHA-256 against the roadmap before processing. In a Colab notebook, prefix shell commands with `!`; in a normal Bash shell, do not use `!`.

## Ten-phase execution order

1. `audit_dataset.py` — verify and inventory the immutable source.
2. `review_sample.py` — create a stratified native-speaker review sample.
3. `prepare_dataset.py` — apply only approved review decisions and segment audio.
4. `make_splits.py` — freeze speaker/prompt-safe train/validation/test manifests.
5. `evaluate_baseline.py` — score unchanged Whisper by dialect.
6. `smoke_train.py` — run the tiny overfit and reload check in Colab.
7. `train.py` — run controlled pilot experiments.
8. `train.py` on the approved full split — track per-dialect validation.
9. `evaluate_baseline.py` or `evaluate_model.py` — independent frozen-test evaluation.
10. `infer.py` and the release checklist — scoped release or private deployment.

Every later phase checks for the relevant manifest and review/license gate. The scripts fail closed when required approvals are missing.

## Citation and data policy

The code is intended for research. Attribute every source dataset according to its current dataset card. The source archive and any derived audio remain external and must be handled under their respective terms. See `docs/REVIEW_PROTOCOL.md` and `docs/RELEASE_CHECKLIST.md`.
