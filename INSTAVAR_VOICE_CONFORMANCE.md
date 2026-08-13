# Instavar Voice conformance

This repository declares its model-specific adaptation and runtime surface in `instavar-voice-capabilities.json`. The LoRA recipe in [`instavar-voice-backend.json`](instavar-voice-backend.json), the experimental full-SFT recipe in [`instavar-voice-backend-full-sft.json`](instavar-voice-backend-full-sft.json), and [`instavar-voice-backend-registry.json`](instavar-voice-backend-registry.json) use the public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation) pinned by CI to commit `8feadf7bbda75abe1c305c63e362c41b86451cda`.

The backend runs corpus audit, content-addressed dataset-lineage verification, the existing LoRA launcher, fresh-process reload, frozen-plan evaluation, and adapter packaging through one fail-closed lifecycle. It verifies lineage before and after preflight and training so an audited manifest cannot be silently substituted between those stages. CI validates the binding and exercises dependency-free wrapper behavior; it does not run GPU training.

The full-SFT backend applies the same audit and evidence chain to a full model.
Its companion-owned trainer keeps the known text-projection and label-alignment
fixes separate from the LoRA patch, saves processor assets with the checkpoint,
and saves a content-verified Accelerate state for trusted same-contract resume at
the next epoch boundary. It rejects manifest, hyperparameter, speaker-control,
state-content, and target-epoch drift, requires explicit trust acknowledgement,
excludes optimizer-bearing state from inference packages, and fails closed when
more than one process is requested. The full-SFT
capability and `pytorch_full_sft` runtime remain `experimental` and `not_run`.
Dependency-free tests prove routing and packaging behavior only. They do not
prove that the model trains, reloads, fits GPU memory, or improves speech.
They also do not establish real interrupted-run restoration, mid-epoch resume,
or resume across dependency versions.

Full-model lifecycle isolation can require several times the checkpoint size
because training, archive, fresh reload, evaluation, and final packaging are
kept as separate evidence surfaces. Capacity planning is an operator
responsibility and is not inferred from a passing preflight.

Capability schema 1.2 records each fine-tuning lifecycle stage separately and names the exact blocker for the matched base-model comparison. A repository-level `supported` label no longer implies corpus audit, evaluation, or packaging completeness.

A capability marked `supported` means the referenced repository evidence reaches the stated engineering boundary. It does not prove perceptual quality, accent fidelity, commercial suitability, or equivalence across untested runtimes. `unverified_for_adapter` keeps an upstream or community runtime visible without implying that this repository's adapted artifact works there.

The common evaluation pack separates deterministic audio diagnostics and objective proxies from blinded human listening. It intentionally defines no universal composite score.

For a reference and candidate runtime, generate the same frozen prompt with recorded settings and run `instavar-voice-eval compare-audio reference.wav candidate.wav`. The result exposes format and signal-level deltas while explicitly refusing to claim runtime equivalence. Establish intelligibility, speaker identity, accent, cadence, and naturalness separately through objective proxies and the blind listening pack.

Before training, use the contract's `audit-corpus` command with explicit train, validation, and test manifests. Supply a parent recording or source identifier through `--group-field` so the audit can reject leakage across splits. File presence and manifest integrity do not prove transcript accuracy or audio quality, which remain separate checks.

`scripts/run_lora_train.sh` can enforce that gate before standalone LoRA work. The registered LoRA and full-SFT lifecycle wrappers always audit all three splits before training. Set `CORPUS_GROUP_FIELD` when the JSONL rows carry a recording or source identifier.

Validate locally with a checkout of the pinned contract:

```bash
python /path/to/instavar-voice-evaluation/main.py validate-repository .
python /path/to/instavar-voice-evaluation/main.py \
  validate-backend-registry instavar-voice-backend-registry.json
```
