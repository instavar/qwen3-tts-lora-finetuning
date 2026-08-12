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
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--init-model-path", required=True)
    parser.add_argument("--output-model-path", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path)
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


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    return rows


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
    core_model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=True,
    )
    processor.save_pretrained(output_dir)
    metadata = {
        "schema_version": "1.0.0",
        "adaptation_mode": "full_sft",
        "speaker_name": speaker_name,
        "speaker_id": speaker_id,
        "speaker_reference_index": speaker_reference_index,
        "training_seed": training_seed,
        "distributed_processes": accelerator.num_processes,
        "evidence_boundary": (
            "Checkpoint creation does not prove perceptual quality or "
            "distribution rights."
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
    _positive_finite(args.learning_rate, "learning rate")
    if args.eval_batch_size is not None:
        _positive(args.eval_batch_size, "eval batch size")
    if not args.speaker_name.strip():
        raise ValueError("speaker name must be non-empty")

    qwen_dir = args.qwen_dir.resolve()
    finetuning_dir = qwen_dir / "finetuning"
    if not (finetuning_dir / "dataset.py").is_file():
        raise FileNotFoundError(
            f"Qwen fine-tuning dataset module not found: {finetuning_dir}"
        )
    sys.path.insert(0, str(qwen_dir))
    sys.path.insert(0, str(finetuning_dir))

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from dataset import TTSDataset
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

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
    tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=dtype_map[args.mixed_precision],
        attn_implementation=args.attention,
    )
    config = AutoConfig.from_pretrained(args.init_model_path)
    train_dataset = TTSDataset(_load_rows(args.train_jsonl), tts.processor, config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    validation_loader = None
    if args.val_jsonl:
        validation_dataset = TTSDataset(
            _load_rows(args.val_jsonl), tts.processor, config
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            collate_fn=validation_dataset.collate_fn,
        )

    model = tts.model
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    if validation_loader is None:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )
    else:
        model, optimizer, train_loader, validation_loader = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            validation_loader,
        )
    speaker_embedding = _canonical_speaker_embedding(
        model,
        train_dataset,
        args.speaker_reference_index,
        accelerator,
        torch,
        DataLoader,
    )

    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                loss = _compute_loss(model, batch, torch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}"
                )

        if validation_loader is not None and (epoch + 1) % args.eval_every == 0:
            validation_loss = _evaluate(model, validation_loader, accelerator, torch)
            accelerator.print(f"Epoch {epoch} | Validation loss: {validation_loss:.4f}")
            model.train()

        if (epoch + 1) % args.save_every == 0:
            _save_checkpoint(
                model,
                tts.processor,
                args.output_model_path / f"checkpoint-epoch-{epoch}",
                speaker_name=args.speaker_name,
                speaker_id=args.speaker_id,
                speaker_embedding=speaker_embedding,
                speaker_reference_index=args.speaker_reference_index,
                training_seed=args.seed,
                accelerator=accelerator,
                torch=torch,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
