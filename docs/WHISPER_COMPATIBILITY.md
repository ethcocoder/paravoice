# Whisper compatibility review

**Reviewed:** 2026-10-10

## Conclusion

The codebase is compatible with multilingual Whisper fine-tuning through Hugging Face Transformers. It is not intended to import OpenAI Whisper's internal model classes directly. OpenAI's repository is the upstream inference and preprocessing reference; it does not provide a complete fine-tuning trainer, optimizer, or dataset-collator workflow.

The project now uses the supported Transformers path: `WhisperProcessor` for feature extraction/tokenization, `WhisperForConditionalGeneration` for the model, and `Seq2SeqTrainer` for training. Amharic is configured as language `am` and the task is `transcribe`.

## Upstream checks

The current OpenAI Whisper `main` branch resolves to commit `86098128c0b4f24f0e2aa2994de830614b474227` during this review. Its reference implementation uses 30-second sliding-window inference and supports an explicit language/task token sequence. The upstream tokenizer includes Amharic under code `am`.

The current Transformers Whisper documentation recommends passing the actual sampling rate, using an attention mask for batched inference, setting language and task during generation, and padding labels independently from input features. The fine-tuning guide recommends resampling to 16 kHz, feature extraction, tokenizer label creation, masking label padding with `-100`, and generation-based evaluation.

## Changes made in this review

- Added explicit `return_attention_mask=True` during feature extraction and passed masks during generation.
- Added current `generate(language="am", task="transcribe")` arguments while retaining generation configuration.
- Added compatibility fallback from `eval_strategy` to `evaluation_strategy`.
- Added compatibility fallback from `processing_class` to the older `tokenizer` Trainer argument.
- Added robust decoder-start-token removal in the label collator.
- Added real mono 16 kHz resampling in derived-data preparation.
- Added the missing `--sample-rate` preparation argument.
- Added the `--steps` alias used by the smoke-test Colab command.
- Corrected Colab references from `rows.csv` to the generated `rows.jsonl` and from `errors.csv` to `predictions.json`.

## Known limitation

The preparation phase excludes recordings longer than 30 seconds until transcript-aligned segmentation is reviewed. It does not truncate audio or copy a complete long transcript onto multiple chunks. A future segmentation implementation must create aligned segment-level labels before long recordings are admitted to a release split.

## Sources

- [OpenAI Whisper repository](https://github.com/openai/whisper)
- [OpenAI Whisper tokenizer](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)
- [Hugging Face Whisper model documentation](https://huggingface.co/docs/transformers/en/model_doc/whisper)
- [Hugging Face Whisper fine-tuning guide](https://huggingface.co/blog/fine-tune-whisper)
