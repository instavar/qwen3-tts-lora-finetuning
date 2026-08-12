#!/usr/bin/env python3
"""Reload one full Qwen3-TTS checkpoint and synthesize one utterance."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--speaker-name", default="speaker")
    parser.add_argument(
        "--text", default="On a quiet morning, the streets were nearly empty."
    )
    parser.add_argument("--language", default="auto")
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max new tokens must be at least 1")
    qwen_dir = args.qwen_dir.resolve()
    sys.path.insert(0, str(qwen_dir))

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=dtype_map[args.dtype],
        attn_implementation=args.attention,
    )
    tts.model.eval()
    wavs, sample_rate = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker_name,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
    )
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, wavs[0], sample_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
