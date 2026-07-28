# Changelog

## Unreleased

### Bug fixes

- Regenerated the `sft_12hz.py` patch context against the README-pinned upstream
  commit `0c6a7cbb6c8421a46332f8c2434c7825c4c855ef`. The patch now applies cleanly
  instead of failing at line 69 because of a stale sub-talker loss coefficient.

## v0.3.0

### Bug fixes (Tier 1 — closes credibility gap between documented pitfalls and shipped fixes)

- **Fix double label-shift bug** (PR #178) — patch now includes the fix in `sft_12hz_lora.py`. Replaces HF internal label shifting with explicit `F.cross_entropy()`. Previously only documented; now shipped.
- **Fix missing `text_projection` call** (PR #188) — patch adds `model.talker.text_projection()` to input embedding computation. Prevents silent wrong embeddings on 1.7B and hard crash on 0.6B.
- **Fix LR default** from 2e-5 to 2e-6 in `sft_12hz_lora.py` — previously only fixed in `run_lora_train.sh` wrapper; now fixed at the Python script level too.

### Community pain points (Tier 2)

- **EOS token fix** — `infer_lora_custom_voice.py` now includes `--max_new_tokens 4096` default to cap generation and prevent infinite loops (~0.5% of inferences)
- **LoRA scale default** changed from 1.0 to 0.3 in inference script — matches validated production range
- **Seed control** — `--seed` flag added to inference for reproducible output and chunk consistency
- **Epoch sweep script** — `scripts/run_infer_epochs.sh` generates one sample per checkpoint for listening comparison
- **0.6B model guard** — `text_projection` fix handles the embedding dimension mismatch (issue #198)

### Differentiation (Tier 3)

- **`scripts/resample_to_24k.sh`** — resamples audio dir to 24kHz before codec prep (prevents #1 training failure)
- **`scripts/compare_checkpoints.sh`** — generates samples across multiple checkpoints × multiple LoRA scales × multiple test sentences for systematic A/B comparison
- **`scripts/infer_long_text.sh`** — chunked long-text inference with per-chunk seed fixing to prevent timbre shift across boundaries

### Patch regenerated

- Patch `0001-qwen3-tts-lora.patch` regenerated to include all Tier 1 bug fixes

## v0.2.0

- **Fix default LR** from 2e-5 to 2e-6 in `run_lora_train.sh`
- **Add `resample_to_24k.sh`** — resamples audio dir to 24kHz before codec prep
- **Expand README** with known pitfalls table, upstream PR tracker, recommended configuration table, data preparation section, and blog cross-links
- **Add upstream PR tracker**

## v0.1.1

- Patch release

## v0.1.0

- Initial companion repo for Qwen3-TTS LoRA fine-tuning
- Patch includes LoRA training/inference/eval helpers
