# Release checklist

- [ ] Source archive hash and version recorded.
- [ ] Native-speaker review protocol, sample, decisions, and qualifications documented.
- [ ] Speaker ID scope resolved; no speaker leakage across splits.
- [ ] Prompt/text leakage measured; final test split frozen before tuning.
- [ ] Every dialect has test counts and WER/CER.
- [ ] Baseline and selected model use recorded Transformers/Whisper revisions and decoding settings.
- [ ] Independent test evaluation run once after model selection.
- [ ] Error examples and native-review feedback included.
- [ ] Dataset cards, attribution, and each license reviewed.
- [ ] Gojjam “other” license explicitly clarified for intended use and distribution.
- [ ] No source audio or restricted derived data is bundled.
- [ ] Model card distinguishes measured facts from source-owner claims.
- [ ] If any right is unresolved, weights remain private and only authorized users receive access.
- [ ] Inference example and known limitations documented.
