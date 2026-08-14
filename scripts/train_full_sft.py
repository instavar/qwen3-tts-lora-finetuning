#!/usr/bin/env python3
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Train a full Qwen3-TTS checkpoint with the companion's known fixes.

This is derived from QwenLM/Qwen3-TTS finetuning/sft_12hz.py under
Apache-2.0. It intentionally supports one process only until distributed
training has its own reproduced checkpoint and reload evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--init-model-path", required=True)
    parser.add_argument("--output-model-path", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path)
    parser.add_argument(
        "--train-row-limit",
        type=int,
        default=0,
        help="Use only the first N training rows; 0 consumes the full manifest.",
    )
    parser.add_argument(
        "--validation-row-limit",
        type=int,
        default=0,
        help="Use only the first N validation rows; 0 consumes the full manifest.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--speaker-name", default="speaker")
    parser.add_argument("--speaker-id", type=int, default=3000)
    parser.add_argument("--speaker-reference-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--trust-resume-state", action="store_true")
    return parser.parse_args()


def _positive(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _nonnegative(value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be at least 0")
    return value


def _positive_finite(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than 0")
    return value


def _load_rows(path: Path, row_limit: int = 0) -> list[dict]:
    _nonnegative(row_limit, "row limit")
    rows: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
            if row_limit and len(rows) >= row_limit:
                break
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"resume-state root is missing or unsafe: {root}")
    files: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"resume-state tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"resume-state tree contains an unsupported entry: {path}")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    if not files:
        raise ValueError("resume-state tree contains no files")
    encoded = json.dumps(
        files, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return {
        "schema_version": "1.0.0",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _file_manifest(path: Path, *, root: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence file is missing or unsafe: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def evaluator_full_sft_artifact_paths(checkpoint: Path) -> dict[str, Path]:
    """Map a new single-process checkpoint to evaluator 0.45 state roles."""
    unresolved = checkpoint.expanduser()
    if unresolved.is_symlink():
        raise ValueError("evaluator checkpoint must not be a symlink")
    resolved = unresolved.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"evaluator checkpoint is missing: {resolved}")

    metadata_path = resolved / "instavar-full-sft-metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise FileNotFoundError(f"resume metadata is missing or unsafe: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "1.2.0":
        raise ValueError("evaluator mapping requires metadata schema 1.2.0")
    if metadata.get("adaptation_mode") != "full_sft":
        raise ValueError("evaluator mapping requires full_sft metadata")

    state_root = resolved / "resume-state"
    if metadata.get("resume_state") != _tree_manifest(state_root):
        raise ValueError("resume-state content does not match checkpoint metadata")
    trainer_state = resolved / "trainer-state.json"
    if metadata.get("trainer_state") != _file_manifest(trainer_state, root=resolved):
        raise ValueError("trainer state does not match checkpoint metadata")

    candidates = {
        "model_state": sorted(state_root.glob("model*.safetensors")),
        "optimizer_state": sorted(state_root.glob("optimizer*.bin")),
        "scheduler_state": sorted(state_root.glob("scheduler*.bin")),
        "rng_state": sorted(state_root.glob("random_states*.pkl")),
    }
    for role, paths in candidates.items():
        if len(paths) != 1:
            raise ValueError(f"evaluator mapping needs exactly one {role} file")
    artifacts = {role: paths[0] for role, paths in candidates.items()}
    artifacts["trainer_state"] = trainer_state
    identities = [(path.stat().st_dev, path.stat().st_ino) for path in artifacts.values()]
    if len(identities) != len(set(identities)):
        raise ValueError("evaluator artifact roles must not share hardlinks")
    return artifacts


def _training_contract(args: argparse.Namespace) -> dict:
    train_jsonl = args.train_jsonl.resolve()
    val_jsonl = args.val_jsonl.resolve() if args.val_jsonl else None
    qwen_dir = args.qwen_dir.resolve()
    dataset_source = qwen_dir / "finetuning" / "dataset.py"
    model_loader_source = qwen_dir / "qwen_tts" / "inference" / "qwen3_tts_model.py"
    init_model_path = Path(args.init_model_path).expanduser()
    if init_model_path.exists():
        resolved_model = init_model_path.resolve()
        init_model_artifact = {
            "kind": "local_directory",
            "path": str(resolved_model),
            "manifest": _tree_manifest(resolved_model),
        }
    else:
        init_model_artifact = {
            "kind": "model_identifier",
            "identifier": str(args.init_model_path),
            "boundary": (
                "The identifier is recorded but its remotely resolved bytes are not "
                "content-bound by this checkpoint."
            ),
        }
    return {
        "schema_version": "1.0.0",
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "qwen_sources": {
            "dataset_py_sha256": _sha256(dataset_source),
            "model_loader_py_sha256": _sha256(model_loader_source),
        },
        "init_model_path": str(args.init_model_path),
        "init_model_artifact": init_model_artifact,
        "train_jsonl": {"path": str(train_jsonl), "sha256": _sha256(train_jsonl)},
        "val_jsonl": (
            {"path": str(val_jsonl), "sha256": _sha256(val_jsonl)}
            if val_jsonl is not None
            else None
        ),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "train_row_limit": getattr(args, "train_row_limit", 0),
        "validation_row_limit": getattr(args, "validation_row_limit", 0),
        "learning_rate": args.learning_rate,
        "scheduler": {
            "type": "constant_lambda",
            "step_interval": "optimizer_step",
        },
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "mixed_precision": args.mixed_precision,
        "attention": args.attention,
        "speaker_name": args.speaker_name,
        "speaker_id": args.speaker_id,
        "speaker_reference_index": args.speaker_reference_index,
        "seed": args.seed,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
    }


def _runtime_contract(
    *, torch, accelerate_version: str, transformers_version: str
) -> dict:
    return {
        "schema_version": "1.0.0",
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda)
        if torch.version.cuda is not None
        else None,
        "accelerate": accelerate_version,
        "transformers": transformers_version,
    }


def _checkpoint_config(source_config: dict, current_config) -> dict:
    result = json.loads(json.dumps(source_config))
    result["tts_model_type"] = "custom_voice"
    source_talker = result.get("talker_config")
    if not isinstance(source_talker, dict):
        raise ValueError("source config is missing talker_config")
    source_talker["spk_id"] = dict(current_config.talker_config.spk_id)
    source_talker["spk_is_dialect"] = dict(
        current_config.talker_config.spk_is_dialect
    )
    return result


def _save_pretrained_with_compatible_config(
    model,
    output_dir: Path,
    state_dict: dict,
    checkpoint_config: dict,
) -> None:
    """Work around Transformers 4.57 nested Qwen config diff serialization."""
    config = model.config
    try:
        config.to_diff_dict()
    except KeyError as error:
        if error.args != ("dtype",):
            raise
        config_type = type(config)
        original_to_diff_dict = config_type.to_diff_dict
        config_type.to_diff_dict = lambda self: checkpoint_config
        try:
            model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=True,
            )
        finally:
            config_type.to_diff_dict = original_to_diff_dict
        return
    model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=True,
    )


def _copy_speech_tokenizer(source: Path, target: Path) -> dict:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"speech tokenizer source is missing or unsafe: {source}")
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"speech tokenizer source contains a symlink: {path}")
        if not path.is_dir() and not path.is_file():
            raise ValueError(
                f"speech tokenizer source contains an unsupported entry: {path}"
            )
    shutil.copytree(source, target)
    return _tree_manifest(target)


def _resume_state(
    checkpoint: Path | None,
    expected_contract: dict,
    *,
    num_epochs: int,
    trust_resume_state: bool,
    expected_runtime_contract: dict | None = None,
) -> tuple[int, Path | None]:
    if checkpoint is None:
        return 0, None
    if not trust_resume_state:
        raise ValueError(
            "resume requires --trust-resume-state because optimizer state may use "
            "PyTorch serialization"
        )
    unresolved = checkpoint.expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"resume checkpoint must not be a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"resume checkpoint is missing: {resolved}")
    metadata_path = resolved / "instavar-full-sft-metadata.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"resume metadata is missing or unsafe: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") not in {"1.1.0", "1.2.0"}:
        raise ValueError("resume checkpoint requires metadata schema 1.1.0 or 1.2.0")
    if metadata.get("adaptation_mode") != "full_sft":
        raise ValueError("resume checkpoint adaptation_mode must equal full_sft")
    if metadata.get("training_contract") != expected_contract:
        raise ValueError("resume checkpoint training contract does not match this run")
    if (
        expected_runtime_contract is not None
        and metadata.get("runtime_contract") != expected_runtime_contract
    ):
        raise ValueError(
            "resume checkpoint runtime contract does not match this environment"
        )
    completed_epochs = metadata.get("completed_epochs")
    if (
        not isinstance(completed_epochs, int)
        or isinstance(completed_epochs, bool)
        or completed_epochs < 1
    ):
        raise ValueError(
            "resume checkpoint completed_epochs must be a positive integer"
        )
    if completed_epochs >= num_epochs:
        raise ValueError(
            "num epochs must exceed the resume checkpoint's completed epochs"
        )
    state_dir = resolved / "resume-state"
    actual_state = _tree_manifest(state_dir)
    if metadata.get("resume_state") != actual_state:
        raise ValueError("resume-state content does not match checkpoint metadata")
    return completed_epochs, state_dir


def _compute_loss(model, batch, torch):
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    with torch.no_grad():
        parameter = next(model.parameters())
        speaker_embedding = model.speaker_encoder(
            ref_mels.to(device=parameter.device, dtype=parameter.dtype)
        ).detach()

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]
    input_text_embedding = model.talker.model.text_embedding(input_text_ids)
    if hasattr(model.talker, "text_projection"):
        input_text_embedding = model.talker.text_projection(input_text_embedding)
    input_text_embedding = input_text_embedding * text_embedding_mask
    input_codec_embedding = (
        model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    )
    input_codec_embedding[:, 6, :] = speaker_embedding
    input_embeddings = input_text_embedding + input_codec_embedding

    for index in range(1, 16):
        codec_embedding = model.talker.code_predictor.get_input_embeddings()[index - 1](
            codec_ids[:, :, index]
        )
        input_embeddings = input_embeddings + codec_embedding * codec_mask.unsqueeze(-1)

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=None,
        output_hidden_states=True,
    )
    targets = codec_0_labels[:, 1:]
    codec_0_loss = torch.nn.functional.cross_entropy(
        outputs.logits.reshape(-1, outputs.logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]
    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
        talker_codec_ids,
        talker_hidden_states,
    )
    total_loss = codec_0_loss + sub_talker_loss
    if not torch.isfinite(total_loss).all():
        raise FloatingPointError("non-finite training or validation loss")
    return total_loss


def _evaluate(model, dataloader, accelerator, torch) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            loss = _compute_loss(model, batch, torch)
            gathered = accelerator.gather_for_metrics(loss.detach())
            losses.append(gathered.reshape(-1))
    if not losses:
        raise ValueError("validation dataloader contains no batches")
    return float(torch.cat(losses).mean().item())


def _canonical_speaker_embedding(
    model,
    dataset,
    reference_index: int,
    accelerator,
    torch,
    DataLoader,
):
    if reference_index >= len(dataset):
        raise ValueError(
            f"speaker reference index {reference_index} is outside {len(dataset)} rows"
        )
    dataloader = DataLoader(
        torch.utils.data.Subset(dataset, [reference_index]),
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    try:
        batch = next(iter(dataloader))
    except StopIteration as error:
        raise ValueError("training dataloader contains no batches") from error
    core_model = accelerator.unwrap_model(model)
    parameter = next(core_model.parameters())
    with torch.no_grad():
        return core_model.speaker_encoder(
            batch["ref_mels"][:1].to(device=parameter.device, dtype=parameter.dtype)
        )[0].detach()


def _save_checkpoint(
    model,
    processor,
    output_dir: Path,
    *,
    speaker_name: str,
    speaker_id: int,
    speaker_embedding,
    speaker_reference_index: int,
    training_seed: int,
    completed_epochs: int,
    training_contract: dict,
    runtime_contract: dict,
    training_observation: dict,
    resume_provenance: dict | None,
    source_config: dict,
    speech_tokenizer_source: Path,
    accelerator,
    torch,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    core_model = accelerator.unwrap_model(model)
    core_model.config.tts_model_type = "custom_voice"
    talker_config = core_model.config.talker_config
    talker_config.spk_id = dict(getattr(talker_config, "spk_id", {}))
    talker_config.spk_is_dialect = dict(getattr(talker_config, "spk_is_dialect", {}))
    existing_names = [
        name
        for name, identifier in talker_config.spk_id.items()
        if identifier == speaker_id and name != speaker_name
    ]
    if existing_names:
        raise ValueError(
            f"speaker id {speaker_id} is already assigned to {sorted(existing_names)!r}"
        )
    state_dict = accelerator.get_state_dict(model)
    codec_weight_key = "talker.model.codec_embedding.weight"
    if codec_weight_key not in state_dict:
        raise KeyError(f"checkpoint state is missing {codec_weight_key}")
    weight = state_dict[codec_weight_key].clone()
    if speaker_id >= weight.shape[0]:
        raise ValueError(
            f"speaker id {speaker_id} is outside codec embedding rows {weight.shape[0]}"
        )
    with torch.no_grad():
        weight[speaker_id].copy_(
            speaker_embedding.to(device=weight.device, dtype=weight.dtype)
        )
    state_dict[codec_weight_key] = weight
    talker_config.spk_id[speaker_name] = speaker_id
    talker_config.spk_is_dialect[speaker_name] = False

    output_dir.mkdir(parents=True, exist_ok=False)
    accelerator.save_state(output_dir / "resume-state", safe_serialization=True)
    resume_state_manifest = _tree_manifest(output_dir / "resume-state")
    trainer_state = {
        "schema_version": "1.0.0",
        "completed_epochs": completed_epochs,
        "epoch_index": training_observation["epoch_index"],
        "microbatches": training_observation["microbatches"],
        "optimizer_steps": training_observation["optimizer_steps"],
        "training_seed": training_seed,
    }
    trainer_state_path = output_dir / "trainer-state.json"
    trainer_state_path.write_text(
        json.dumps(trainer_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    speech_tokenizer_manifest = _copy_speech_tokenizer(
        speech_tokenizer_source, output_dir / "speech_tokenizer"
    )
    _save_pretrained_with_compatible_config(
        core_model,
        output_dir,
        state_dict,
        _checkpoint_config(source_config, core_model.config),
    )
    processor.save_pretrained(output_dir)
    metadata = {
        "schema_version": "1.2.0",
        "adaptation_mode": "full_sft",
        "completed_epochs": completed_epochs,
        "speaker_name": speaker_name,
        "speaker_id": speaker_id,
        "speaker_reference_index": speaker_reference_index,
        "training_seed": training_seed,
        "training_contract": training_contract,
        "runtime_contract": runtime_contract,
        "training_observation": training_observation,
        "resume_provenance": resume_provenance,
        "resume_state": resume_state_manifest,
        "trainer_state": _file_manifest(trainer_state_path, root=output_dir),
        "speech_tokenizer": speech_tokenizer_manifest,
        "distributed_processes": accelerator.num_processes,
        "evidence_boundary": (
            "The nested Accelerator state supports same-contract epoch-boundary "
            "resume in one process. Checkpoint creation does not prove perceptual "
            "quality, cross-version resume, or distribution rights."
        ),
    }
    (output_dir / "instavar-full-sft-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    for value, name in (
        (args.batch_size, "batch size"),
        (args.num_epochs, "num epochs"),
        (args.gradient_accumulation_steps, "gradient accumulation steps"),
        (args.save_every, "save every"),
        (args.eval_every, "eval every"),
    ):
        _positive(value, name)
    _nonnegative(args.speaker_id, "speaker id")
    _nonnegative(args.speaker_reference_index, "speaker reference index")
    _nonnegative(args.train_row_limit, "train row limit")
    _nonnegative(args.validation_row_limit, "validation row limit")
    _positive_finite(args.learning_rate, "learning rate")
    if args.eval_batch_size is not None:
        _positive(args.eval_batch_size, "eval batch size")
    if not args.speaker_name.strip():
        raise ValueError("speaker name must be non-empty")
    training_contract = _training_contract(args)

    qwen_dir = args.qwen_dir.resolve()
    finetuning_dir = qwen_dir / "finetuning"
    if not (finetuning_dir / "dataset.py").is_file():
        raise FileNotFoundError(
            f"Qwen fine-tuning dataset module not found: {finetuning_dir}"
        )
    sys.path.insert(0, str(qwen_dir))
    sys.path.insert(0, str(finetuning_dir))

    import accelerate
    import torch
    import transformers
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from dataset import TTSDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig
    from transformers.utils import cached_file

    runtime_contract = _runtime_contract(
        torch=torch,
        accelerate_version=accelerate.__version__,
        transformers_version=transformers.__version__,
    )
    start_epoch, resume_state = _resume_state(
        args.resume_from_checkpoint,
        training_contract,
        num_epochs=args.num_epochs,
        trust_resume_state=args.trust_resume_state,
        expected_runtime_contract=runtime_contract,
    )
    resume_provenance = None
    if args.resume_from_checkpoint is not None:
        resume_root = args.resume_from_checkpoint.expanduser().resolve()
        resume_provenance = {
            "checkpoint_path": str(resume_root),
            "metadata_sha256": _sha256(
                resume_root / "instavar-full-sft-metadata.json"
            ),
            "resume_state": _tree_manifest(resume_root / "resume-state"),
        }

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
    )
    if accelerator.num_processes != 1:
        raise RuntimeError(
            "full SFT currently requires exactly one process; multi-process "
            "checkpoint semantics are unverified"
        )
    set_seed(args.seed, device_specific=True)

    dtype_map = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=dtype_map[args.mixed_precision],
        attn_implementation=args.attention,
    )
    config_source = cached_file(args.init_model_path, "config.json")
    if config_source is None:
        raise FileNotFoundError("base model config.json could not be resolved")
    source_config = json.loads(Path(config_source).read_text(encoding="utf-8"))
    speech_tokenizer_config = cached_file(
        args.init_model_path, "speech_tokenizer/config.json"
    )
    if speech_tokenizer_config is None:
        raise FileNotFoundError("base model speech_tokenizer/config.json is missing")
    speech_tokenizer_source = Path(speech_tokenizer_config).parent
    config = AutoConfig.from_pretrained(args.init_model_path)
    train_dataset = TTSDataset(
        _load_rows(args.train_jsonl, args.train_row_limit), tts.processor, config
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    validation_loader = None
    if args.val_jsonl:
        validation_dataset = TTSDataset(
            _load_rows(args.val_jsonl, args.validation_row_limit),
            tts.processor,
            config,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            collate_fn=validation_dataset.collate_fn,
        )

    model = tts.model
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )
    if validation_loader is None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    else:
        model, optimizer, train_loader, validation_loader, scheduler = (
            accelerator.prepare(
                model,
                optimizer,
                train_loader,
                validation_loader,
                scheduler,
            )
        )
    speaker_embedding = _canonical_speaker_embedding(
        model,
        train_dataset,
        args.speaker_reference_index,
        accelerator,
        torch,
        DataLoader,
    )
    if resume_state is not None:
        accelerator.load_state(resume_state)

    model.train()
    for epoch in range(start_epoch, args.num_epochs):
        epoch_started = time.perf_counter()
        epoch_losses: list[float] = []
        optimizer_steps = 0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                loss = _compute_loss(model, batch, torch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()
            epoch_losses.append(float(loss.detach().item()))
            if accelerator.sync_gradients:
                optimizer_steps += 1
            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                )

        validation_loss = None
        if validation_loader is not None and (epoch + 1) % args.eval_every == 0:
            validation_loss = _evaluate(model, validation_loader, accelerator, torch)
            accelerator.print(f"Epoch {epoch} | Validation loss: {validation_loss:.4f}")
            model.train()

        if (epoch + 1) % args.save_every == 0:
            training_observation = {
                "epoch_index": epoch,
                "completed_epochs": epoch + 1,
                "microbatches": len(epoch_losses),
                "optimizer_steps": optimizer_steps,
                "mean_training_loss": sum(epoch_losses) / len(epoch_losses),
                "final_training_loss": epoch_losses[-1],
                "validation_loss": validation_loss,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "peak_cuda_memory_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else None
                ),
                "peak_cuda_memory_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved())
                    if torch.cuda.is_available()
                    else None
                ),
            }
            _save_checkpoint(
                model,
                tts.processor,
                args.output_model_path / f"checkpoint-epoch-{epoch}",
                speaker_name=args.speaker_name,
                speaker_id=args.speaker_id,
                speaker_embedding=speaker_embedding,
                speaker_reference_index=args.speaker_reference_index,
                training_seed=args.seed,
                completed_epochs=epoch + 1,
                training_contract=training_contract,
                runtime_contract=runtime_contract,
                training_observation=training_observation,
                resume_provenance=resume_provenance,
                source_config=source_config,
                speech_tokenizer_source=speech_tokenizer_source,
                accelerator=accelerator,
                torch=torch,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
