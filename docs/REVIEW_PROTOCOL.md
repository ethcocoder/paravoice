# Native-speaker review protocol

## Sampling

Stratify by dialect, speaker, gender when available, duration bucket, and transcript length. Include flagged rows: replacement characters, non-Ethiopic text, exact 500-character transcripts, malformed or unusual audio, and representative unflagged controls.

## Per-row review

1. Listen without reading the transcript first.
2. Record whether the audio is intelligible and complete.
3. Compare the spoken words with the transcript; do not silently rewrite.
4. Judge whether the dialect label is plausible, or mark uncertain.
5. Mark spelling/orthography issues separately from speech-content errors.
6. Record code-switching, transliteration, clipping, noise, and multiple speakers.

Allowed final decisions: `keep`, `correct`, `exclude`. A correction must include the corrected transcript and reviewer ID. Disagreements receive a second independent review; unresolved cases are excluded from the approved release split.

## Required outputs

- reviewer qualifications and dialect coverage
- anonymized sample manifest
- per-category counts and rates
- disagreement and adjudication log
- explicit policy for code-switching and spelling normalization
- sign-off date and responsible reviewer
