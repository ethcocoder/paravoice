# Google Colab: clone to finish

This guide is the operational path from a fresh Colab runtime to a scoped Phase 10 result. It deliberately stops at human-review and license gates instead of pretending that machine checks are sufficient.

## 0. Runtime and project setup

Use a GPU runtime (`Runtime > Change runtime type > T4 GPU` or better). Clone the required `v2` branch of the public repository; do not omit `--branch v2`:

```python
!git clone --branch v2 --single-branch https://github.com/ethcocoder/paravoice.git
%cd paravoice
!git branch --show-current
!pip install -q -e .
!nvidia-smi
```

These commands are **Colab notebook syntax**. If you are inside a normal Bash shell, remove the leading `!`: use `pip install -q -e .` and `nvidia-smi`, not `!nvidia-smi`. The dependency-conflict messages that Colab may print after the editable install are warnings from preinstalled packages; the install is successful when it ends with `Successfully built five-dialect-amharic-asr` and the CLI checks work.

The branch check must print `v2`. If it prints another branch, stop, remove the checkout, and rerun the clone cell. To verify the exact remote and commit, run:

```python
!git remote -v
!git log -1 --oneline
```

Create working directories outside Git:

```python
from pathlib import Path
for p in ["/content/asr-work/source", "/content/asr-work/derived", "/content/asr-work/runs", "/content/asr-work/reviews"]:
    Path(p).mkdir(parents=True, exist_ok=True)
```

### Option A — clone the verified source archive directly from Hugging Face

The dataset repository contains the 15.84 GB `leyu_all_datasets.zip` file through Git LFS. Make sure the Colab runtime has enough disk space before starting. This clone downloads the source outside the code repository:

```python
!apt-get -qq update && apt-get -qq install -y git-lfs
!git lfs install
!git clone https://huggingface.co/datasets/paradoxquantum/paradox-amharic-data /content/asr-work/source/paradox-amharic-data
!git -C /content/asr-work/source/paradox-amharic-data lfs pull
!ls -lh /content/asr-work/source/paradox-amharic-data/leyu_all_datasets.zip
```

The expected archive hash is `433982aa7bc2f504bbffe6a38d4cbb63f93a17cd9fd35d8442276ef980afe2d3`, matching the roadmap and the Hugging Face LFS object. If the hash does not match, stop and investigate instead of training.

### Option B — upload or mount an existing verified archive

Do not put the archive in the Git repository:

```python
from google.colab import drive
drive.mount('/content/drive')
# Example: copy from Drive after checking the path
!cp /content/drive/MyDrive/leyu_all_datasets.zip /content/asr-work/source/
```

## Phase 1 — audit

Use the archive hash from the roadmap as an expected value. If your archive is different, record its new hash and source URL; do not silently continue.

```python
!python scripts/audit_dataset.py \
  --source /content/asr-work/source/paradox-amharic-data/leyu_all_datasets.zip \
  --output /content/asr-work/audit \
  --expected-sha256 433982aa7bc2f504bbffe6a38d4cbb63f93a17cd9fd35d8442276ef980afe2d3
```

For Option B, change `--source` to `/content/asr-work/source/leyu_all_datasets.zip`.

Inspect `audit/report.json`, `audit/rows.jsonl`, and `audit/issues.json`. Phase 1 is not signed off until a native-speaker sample and Gojjam license status are recorded.

## Phase 2 — native review

Generate a stratified sample without editing the source:

```python
!python scripts/review_sample.py \
  --audit /content/asr-work/audit/rows.jsonl \
  --output /content/asr-work/reviews/sample.csv \
  --n-per-dialect 30 --seed 42
```

Have fluent reviewers listen to each audio path and complete `docs/REVIEW_PROTOCOL.md`. Save a decision file with columns `source_row_id, transcript_decision, dialect_decision, reviewer_id, notes`; allowed decisions are `keep`, `correct`, and `exclude`. Double-review disagreements.

## Phase 3 — cleaning and segmentation

After review decisions exist:

```python
!python scripts/prepare_dataset.py \
  --source /content/asr-work/audit/extracted \
  --audit /content/asr-work/audit/rows.jsonl \
  --reviews /content/asr-work/reviews/decisions.csv \
  --output /content/asr-work/derived \
  --sample-rate 16000 --max-seconds 30
```

The source remains immutable. The preparation step writes mono 16 kHz WAV files and currently excludes clips longer than 30 seconds until transcript-aligned segmentation is reviewed; it never truncates audio or duplicates a full transcript across chunks. Inspect `derived/manifest.jsonl`, `derived/provenance.json`, and the retained-row counts. Do not use `--allow-unreviewed` for a release run.

## Phase 4 — benchmark split

Resolve whether numeric speaker IDs are globally reusable before this command. If they are local, the script namespaces them by dialect; if they are global, pass `--speaker-scope global`.

```python
!python scripts/make_splits.py \
  --manifest /content/asr-work/derived/manifest.jsonl \
  --output /content/asr-work/splits \
  --speaker-scope dialect_local --seed 42 \
  --test-speakers 0.20 --validation-speakers 0.10
```

Check `splits/leakage_report.json`: speaker overlap must be zero, and prompt overlap in test must be explicitly reviewed. Freeze this directory before any model selection.

## Phase 5 — unchanged baseline

```python
!python scripts/evaluate_baseline.py \
  --manifest /content/asr-work/splits/test.jsonl \
  --output /content/asr-work/runs/baseline \
  --model openai/whisper-small \
  --batch-size 4
```

Review `metrics.json`, `predictions.jsonl`, and `errors.csv`. The language is Amharic (`am`), task is transcription, and generation is used for evaluation.

## Phase 6 — smoke test

Run a tiny, intentionally overfit experiment. It validates decoding, resampling, labels, generation, checkpoint saving, and reload:

```python
!python scripts/smoke_train.py \
  --train /content/asr-work/splits/train.jsonl \
  --validation /content/asr-work/splits/validation.jsonl \
  --output /content/asr-work/runs/smoke \
  --model openai/whisper-small \
  --steps 30 --max-train-items 8
```

Do not proceed if the reload check cannot generate Amharic text or if labels are misaligned.

## Phase 7 — pilot

Start with a small sample and one conservative recipe:

```python
!python scripts/train.py \
  --train /content/asr-work/splits/train.jsonl \
  --validation /content/asr-work/splits/validation.jsonl \
  --output /content/asr-work/runs/pilot-small \
  --model openai/whisper-small \
  --max-steps 500 --gradient-accumulation 8 \
  --gradient-checkpointing --fp16
```

Compare full fine-tuning against any parameter-efficient variant only after the smoke test. Select using validation metrics and native error review, never the frozen test set.

## Phase 8 — full training

Run the selected configuration on the complete approved training split. Keep each run directory immutable and record the exact command, package versions, model revision, GPU, and config. Track aggregate and per-dialect validation results.

## Phase 9 — independent evaluation

Evaluate the selected checkpoint once on the frozen test manifest:

```python
!python scripts/evaluate_model.py \
  --model /content/asr-work/runs/full/best \
  --manifest /content/asr-work/splits/test.jsonl \
  --output /content/asr-work/runs/independent-test
```

Add bootstrap confidence intervals or report sample counts, native-review findings, and license decisions to the model card. Do not release weights while the Gojjam terms are unresolved.

## Phase 10 — inference and release

```python
!python scripts/infer.py \
  --model /content/asr-work/runs/full/best \
  --audio /content/example.wav \
  --output /content/prediction.json
```

Complete `docs/RELEASE_CHECKLIST.md`. If distribution rights are not documented, keep the model private and publish only code and reproducibility metadata. Save run artifacts to Drive or an approved model registry; never commit them to Git.
