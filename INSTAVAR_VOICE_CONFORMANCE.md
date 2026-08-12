# Instavar Voice conformance

This repository declares its model-specific adaptation and runtime surface in `instavar-voice-capabilities.json`. The manifest and executable [`instavar-voice-backend.json`](instavar-voice-backend.json) recipe use the public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation) pinned by CI to commit `e1229101703582dd0db0b84ddf1698a9348bc70e`.

The backend runs corpus audit, content-addressed dataset-lineage verification, the existing LoRA launcher, fresh-process reload, frozen-plan evaluation, and adapter packaging through one fail-closed lifecycle. It verifies lineage before and after preflight and training so an audited manifest cannot be silently substituted between those stages. CI validates the binding and exercises dependency-free wrapper behavior; it does not run GPU training.

Capability schema 1.2 records each fine-tuning lifecycle stage separately and names the exact blocker for the matched base-model comparison. A repository-level `supported` label no longer implies corpus audit, evaluation, or packaging completeness.

A capability marked `supported` means the referenced repository evidence reaches the stated engineering boundary. It does not prove perceptual quality, accent fidelity, commercial suitability, or equivalence across untested runtimes. `unverified_for_adapter` keeps an upstream or community runtime visible without implying that this repository's adapted artifact works there.

The common evaluation pack separates deterministic audio diagnostics and objective proxies from blinded human listening. It intentionally defines no universal composite score.

For a reference and candidate runtime, generate the same frozen prompt with recorded settings and run `instavar-voice-eval compare-audio reference.wav candidate.wav`. The result exposes format and signal-level deltas while explicitly refusing to claim runtime equivalence. Establish intelligibility, speaker identity, accent, cadence, and naturalness separately through objective proxies and the blind listening pack.

Before training, use the contract's `audit-corpus` command with explicit train, validation, and test manifests. Supply a parent recording or source identifier through `--group-field` so the audit can reject leakage across splits. File presence and manifest integrity do not prove transcript accuracy or audio quality, which remain separate checks.

`scripts/run_lora_train.sh` can enforce that gate before GPU work. Set `AUDIT_CORPUS=1`, `VAL_JSONL`, `TEST_JSONL`, and `INSTAVAR_VOICE_EVAL_DIR`. Set `CORPUS_GROUP_FIELD` when the JSONL rows carry a recording or source identifier. The existing training behavior remains unchanged when the gate is not enabled.

Validate locally with a checkout of the pinned contract:

```bash
python /path/to/instavar-voice-evaluation/main.py validate-repository .
```
