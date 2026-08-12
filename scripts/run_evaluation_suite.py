#!/usr/bin/env python3
"""Run a frozen Instavar Voice generation plan with one loaded Qwen3-TTS adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel


IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--runtime-id", default="pytorch")
    parser.add_argument("--artifact-set-id")
    parser.add_argument("--artifact-set-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speaker-name", default="female01")
    parser.add_argument("--speaker-id", type=int, default=3000)
    parser.add_argument("--speaker-embedding")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--lora-scale", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--no-merge-lora", action="store_true")
    return parser.parse_args()


def read_plan(path: Path, candidate_id: str) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        plan = json.load(source)
    if plan.get("schema_version") != "1.0.0":
        raise ValueError("generation plan schema_version must equal 1.0.0")
    rows = [row for row in plan.get("samples", []) if row.get("candidate_id") == candidate_id]
    if not rows:
        raise ValueError(f"generation plan has no rows for candidate {candidate_id!r}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_observations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_artifact_fields(args: argparse.Namespace) -> dict[str, str]:
    if not IDENTIFIER_RE.fullmatch(args.runtime_id):
        raise ValueError("runtime id must be a lowercase machine-readable identifier")
    if bool(args.artifact_set_id) != bool(args.artifact_set_sha256):
        raise ValueError("artifact set id and sha256 must be provided together")
    fields = {"runtime_id": args.runtime_id}
    if args.artifact_set_id:
        if not IDENTIFIER_RE.fullmatch(args.artifact_set_id):
            raise ValueError("artifact set id must be a lowercase machine-readable identifier")
        if not re.fullmatch(r"[0-9a-f]{64}", args.artifact_set_sha256):
            raise ValueError("artifact set sha256 must be a lowercase SHA-256 digest")
        fields.update(
            {
                "artifact_set_id": args.artifact_set_id,
                "artifact_set_sha256": args.artifact_set_sha256,
            }
        )
    return fields


def main() -> int:
    args = parse_args()
    artifact_fields = runtime_artifact_fields(args)
    rows = read_plan(args.generation_plan, args.candidate_id)
    finetuning_dir = args.qwen_dir.resolve() / "finetuning"
    if not finetuning_dir.is_dir():
        raise FileNotFoundError(f"Qwen finetuning directory not found: {finetuning_dir}")
    sys.path.insert(0, str(args.qwen_dir.resolve()))
    sys.path.insert(0, str(finetuning_dir))
    from qwen_tts import Qwen3TTSModel

    helper = importlib.import_module("infer_lora_custom_voice")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tts = Qwen3TTSModel.from_pretrained(
        args.base_model,
        device_map=args.device,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attention,
    )
    peft_model = PeftModel.from_pretrained(tts.model, str(args.adapter))
    helper._set_lora_scale(peft_model, args.lora_scale)
    tts.model = peft_model if args.no_merge_lora else peft_model.merge_and_unload()
    core_model = helper._resolve_core_model(tts.model)
    helper._apply_speaker_config(core_model, str(args.adapter), args.speaker_name, args.speaker_id)
    helper._apply_speaker_embedding(core_model, str(args.adapter), args.speaker_name, args.speaker_embedding)
    core_model.eval()

    observations: list[dict] = []
    for row in rows:
        output = args.output_dir / row["expected_audio_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        set_seed(int(row["seed"]))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        observation = {
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "seed": row["seed"],
            "requested_text": row["text"],
            "valid": False,
            "runtime": "qwen3_tts_pytorch_cuda_adapter",
            **artifact_fields,
        }
        try:
            wavs, sample_rate = tts.generate_custom_voice(
                text=row["text"],
                speaker=args.speaker_name,
                language=args.language,
                instruct=row.get("instruction"),
                max_new_tokens=args.max_new_tokens,
            )
            sf.write(output, wavs[0], sample_rate)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            info = sf.info(output)
            observation.update(
                {
                    "valid": info.frames > 0,
                    "audio_path": str(output),
                    "audio_sha256": sha256(output),
                    "audio_duration_seconds": float(info.duration),
                    "generation_seconds": elapsed,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                    "instruction_applied": bool(row.get("instruction")),
                }
            )
        except Exception as error:
            observation.update(
                {
                    "generation_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        observations.append(observation)
        write_observations(args.output_dir / "generation-observations.json", observations)

    return 0 if all(row["valid"] for row in observations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
