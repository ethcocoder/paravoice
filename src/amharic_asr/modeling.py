"""Whisper evaluation and fine-tuning helpers; heavy ML imports are runtime-only."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def score(model_name, manifest, output, batch_size=2, max_steps=None, train=False, max_items=None, fp16=False, grad_ckpt=False, grad_accumulation=8):
    import torch
    from datasets import Audio, Dataset
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
    rows = load_rows(manifest)
    rows = rows[:max_items] if max_items else rows
    ds = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))
    processor = WhisperProcessor.from_pretrained(model_name, language="am", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = "am"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    def prepare(batch):
        audio = batch["audio"]
        features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_attention_mask=True
        )
        batch["input_features"] = features.input_features[0]
        if getattr(features, "attention_mask", None) is not None:
            batch["attention_mask"] = features.attention_mask[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds.column_names)

    def collate(features):
        input_features = torch.tensor([f["input_features"] for f in features], dtype=torch.float32)
        batch = {"input_features": input_features}
        if "attention_mask" in features[0]:
            batch["attention_mask"] = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == model.config.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if train:
        kwargs = dict(
            output_dir=str(out), per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accumulation, learning_rate=1e-6,
            warmup_steps=100, max_steps=max_steps or 1000, fp16=fp16 and torch.cuda.is_available(),
            gradient_checkpointing=grad_ckpt, logging_steps=10, save_steps=100,
            report_to="none", remove_unused_columns=False,
        )
        try:
            args = Seq2SeqTrainingArguments(eval_strategy="no", **kwargs)
        except TypeError:
            args = Seq2SeqTrainingArguments(evaluation_strategy="no", **kwargs)
        try:
            trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds, data_collator=collate, processing_class=processor)
        except TypeError:
            trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds, data_collator=collate, tokenizer=processor)
        trainer.train()
        trainer.save_model(out / "best")
        processor.save_pretrained(out / "best")
        return

    pred = []
    model.eval()
    for i in range(0, len(ds), batch_size):
        batch = collate([ds[j] for j in range(i, min(i + batch_size, len(ds)))])
        with torch.no_grad():
            ids = model.generate(
                input_features=batch["input_features"].to(model.device),
                attention_mask=batch.get("attention_mask", None).to(model.device) if "attention_mask" in batch else None,
                language="am", task="transcribe",
            )
        pred.extend(processor.batch_decode(ids, skip_special_tokens=True))
    refs = [r["text"] for r in rows]
    from jiwer import cer, wer
    metrics = {"wer": wer(refs, pred), "cer": cer(refs, pred), "count": len(refs)}
    by_dialect = {}
    for row, prediction in zip(rows, pred):
        by_dialect.setdefault(row.get("dialect", ""), []).append((row["text"], prediction))
    for dialect, pairs in by_dialect.items():
        metrics[f"{dialect}_wer"] = wer([x[0] for x in pairs], [x[1] for x in pairs])
        metrics[f"{dialect}_cer"] = cer([x[0] for x in pairs], [x[1] for x in pairs])
        metrics[f"{dialect}_count"] = len(pairs)
    json.dump(metrics, open(out / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump([{"reference": r, "prediction": p} for r, p in zip(refs, pred)], open(out / "predictions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def main_eval():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--output", required=True); p.add_argument("--model", default="openai/whisper-small"); p.add_argument("--batch-size", type=int, default=2); a = p.parse_args(); score(a.model, a.manifest, a.output, a.batch_size)

def main_train():
    p = argparse.ArgumentParser(); p.add_argument("--train", required=True); p.add_argument("--validation"); p.add_argument("--output", required=True); p.add_argument("--model", default="openai/whisper-small"); p.add_argument("--max-steps", "--steps", dest="max_steps", type=int, default=1000); p.add_argument("--batch-size", type=int, default=2); p.add_argument("--gradient-accumulation", type=int, default=8); p.add_argument("--fp16", action="store_true"); p.add_argument("--gradient-checkpointing", action="store_true"); p.add_argument("--max-train-items", type=int); a = p.parse_args(); score(a.model, a.train, a.output, a.batch_size, a.max_steps, True, a.max_train_items, a.fp16, a.gradient_checkpointing, a.gradient_accumulation)

def main_infer():
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--audio", required=True); p.add_argument("--output", required=True); a = p.parse_args()
    import torch
    from transformers import pipeline
    result = pipeline("automatic-speech-recognition", model=a.model, chunk_length_s=30, device=0 if torch.cuda.is_available() else -1)(a.audio, generate_kwargs={"language": "am", "task": "transcribe"})
    json.dump(result, open(a.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
