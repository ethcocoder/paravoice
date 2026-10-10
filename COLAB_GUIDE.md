# Google Colab: clone to finish

This guide is the operational path from a fresh Colab runtime to a scoped Phase 10 result. It deliberately stops at human-review and license gates instead of pretending that machine checks are sufficient.

## 0. Runtime and project setup

Use a GPU runtime (`Runtime > Change runtime type > T4 GPU` or better). Then run:

```python
!git clone -b v2 https://github.com/ethcocoder/five-dialect-amharic-asr.git
%cd five-dialect-amharic-asr
!pip install -q -e .
!nvidia-smi
```

Create working directories outside Git:

```python
from pathlib import Path
for p in ["/content/asr-work/source", "/content/asr-work/derived", "/content/asr-work/runs", "/content/asr-work/reviews"]:
    Path(p).mkdir(parents=True, exist_ok=True)
```

Upload or mount the verified source archive. Do not put it in the repository:

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
  --source /content/asr-work/source/leyu_all_datasets.zip \
  --output /content/asr-work/audit \
  --expected-sha256 433982aa7bc2f504bbffe6a38d4cbb63f93a17cd9fd35d8442276ef980afe2d3
```

Inspect `audit/report.json`, `audit/rows.csv`, and `audit/issues.json`. Phase 1 is not signed off until a native-speaker sample and Gojjam license status are recorded.

## Phase 2 — native review

Generate a stratified sample without editing the source:

```python
!python scripts/review_sample.py \
  --audit /content/asr-work/audit/rows.csv \
  --output /content/asr-work/reviews/sample.csv \
  --n-per-dialect 30 --seed 42
```

Have fluent reviewers listen to each audio path and complete `docs/REVIEW_PROTOCOL.md`. Save a decision file with columns `source_row_id, transcript_decision, dialect_decision, reviewer_id, notes`; allowed decisions are `keep`, `correct`, and `exclude`. Double-review disagreements.

## Phase 3 — cleaning and segmentation

After review decisions exist:

```python
!python scripts/prepare_dataset.py \
  --source /content/asr-work/source/extracted \
  --audit /content/asr-work/audit/rows.csv \
  --reviews /content/asr-work/reviews/decisions.csv \
  --output /content/asr-work/derived \
  --sample-rate 16000 --max-seconds 30
```

The source remains immutable. Inspect `derived/manifest.jsonl`, `derived/issues.json`, and `derived/provenance.json`. Do not use `--allow-unreviewed` for a release run.

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
