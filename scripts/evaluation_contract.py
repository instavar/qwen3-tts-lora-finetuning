"""Dependency-free inference-mode validation for Qwen3-TTS evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _model_type(path: Path, name: str) -> str:
    if not path.is_dir():
        raise ValueError(f"{name} must be an existing directory")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"{name} must contain config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("tts_model_type")
    if model_type not in {"base", "custom_voice"}:
        raise ValueError(f"{name} config must declare tts_model_type base or custom_voice")
    return model_type


def resolve_inference_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Validate one base-clone, adapter, or full-SFT condition."""
    mode = args.inference_mode
    if mode is None:
        if args.adapter and not args.model:
            mode = "adapter"
        elif args.model and not args.adapter:
            mode = "full-sft"
        else:
            parser.error("provide --inference-mode or exactly one of --adapter or --model")

    if mode == "base-clone":
        if args.adapter or args.model:
            parser.error("base-clone mode forbids --adapter and --model")
        if not args.base_model:
            parser.error("base-clone mode requires --base-model")
        if not args.reference_audio or not args.reference_text or not args.reference_text.strip():
            parser.error("base-clone mode requires --reference-audio and --reference-text")
        if args.reference_audio.is_symlink():
            parser.error("base-clone reference audio must not be a symlink")
        if not args.reference_audio.is_file():
            parser.error("base-clone reference audio must be an existing file")
        if _model_type(Path(args.base_model), "base model") != "base":
            parser.error("base-clone mode requires a base tts_model_type")
        return mode

    if args.reference_audio or args.reference_text:
        parser.error("reference audio and text are only valid in base-clone mode")

    if mode == "adapter":
        if not args.adapter or not args.base_model or args.model:
            parser.error("adapter mode requires --base-model and --adapter and forbids --model")
        if not args.adapter.is_dir():
            parser.error("adapter must be an existing directory")
        if _model_type(Path(args.base_model), "base model") != "base":
            parser.error("adapter mode requires a base tts_model_type")
        return mode

    if not args.model or args.base_model or args.adapter:
        parser.error("full-sft mode requires --model and forbids --base-model and --adapter")
    if _model_type(Path(args.model), "full-SFT model") != "custom_voice":
        parser.error("full-sft mode requires a custom_voice tts_model_type")
    return mode


def reject_unsupported_plan_rows(mode: str, rows: list[dict], parser: argparse.ArgumentParser) -> None:
    if mode == "base-clone" and any(row.get("instruction") for row in rows):
        parser.error("base-clone mode does not support instruction-bearing plan rows")
