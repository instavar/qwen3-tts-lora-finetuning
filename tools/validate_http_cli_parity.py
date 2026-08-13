#!/usr/bin/env python3
"""Validate one content-bound Qwen HTTP observation against its CLI row."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--cli-observations", type=Path, required=True)
    parser.add_argument("--http-observation", type=Path, required=True)
    parser.add_argument("--startup-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exactly_one(rows: list[dict], candidate_id: str, sample_id: str, label: str) -> dict:
    matches = [
        row
        for row in rows
        if row.get("candidate_id") == candidate_id and row.get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one selected row")
    return matches[0]


def load_plan_row(path: Path, candidate_id: str, sample_id: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("unsupported generation plan schema")
    return exactly_one(document.get("samples", []), candidate_id, sample_id, "generation plan")


def validate_identity(row: dict, plan: dict, label: str) -> None:
    for field in ("sample_id", "candidate_id", "prompt_id", "category", "seed"):
        if row.get(field) != plan.get(field):
            raise ValueError(f"{label} {field} does not match the generation plan")
    if row.get("requested_text") != plan.get("text"):
        raise ValueError(f"{label} requested text does not match the generation plan")
    if row.get("instruction_applied") != bool(plan.get("instruction")):
        raise ValueError(f"{label} instruction state does not match the generation plan")
    if row.get("valid") is not True or not SHA256_RE.fullmatch(str(row.get("audio_sha256", ""))):
        raise ValueError(f"{label} is not a valid audio observation")


def main() -> int:
    args = parse_args()
    plan = load_plan_row(args.generation_plan, args.candidate_id, args.sample_id)
    cli_document = json.loads(args.cli_observations.read_text(encoding="utf-8"))
    if not isinstance(cli_document, list):
        raise ValueError("CLI observations must be a JSON array")
    cli = exactly_one(cli_document, args.candidate_id, args.sample_id, "CLI observations")
    http = json.loads(args.http_observation.read_text(encoding="utf-8"))
    if not isinstance(http, dict):
        raise ValueError("HTTP observation must be a JSON object")
    receipt = json.loads(args.startup_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "1.0.0":
        raise ValueError("unsupported startup receipt schema")
    if receipt.get("runtime_id") != "qwen3_tts_openai_compatible_http":
        raise ValueError("startup receipt runtime does not match the HTTP bridge")
    if receipt.get("seed") != plan.get("seed"):
        raise ValueError("startup receipt seed does not match the selected plan row")
    expected_artifact_mode = receipt.get("artifact_mode")
    if expected_artifact_mode not in {"adapter", "full_sft"}:
        raise ValueError("startup receipt has an invalid artifact mode")
    if http.get("artifact_mode") != expected_artifact_mode:
        raise ValueError("HTTP observation artifact mode does not match startup receipt")
    receipt_sha256 = sha256_file(args.startup_receipt)
    if http.get("startup_receipt_sha256") != receipt_sha256:
        raise ValueError("HTTP observation is not bound to the supplied startup receipt")
    validate_identity(cli, plan, "CLI observation")
    validate_identity(http, plan, "HTTP observation")
    if cli.get("artifact_mode") != http.get("artifact_mode"):
        raise ValueError("CLI and HTTP observations use different artifact modes")

    exact = cli["audio_sha256"] == http["audio_sha256"]
    report = {
        "schema_version": "1.0.0",
        "candidate_id": args.candidate_id,
        "sample_id": args.sample_id,
        "seed": plan["seed"],
        "instruction_applied": bool(plan.get("instruction")),
        "artifact_mode": expected_artifact_mode,
        "startup_receipt_sha256": receipt_sha256,
        "artifact_set_id": receipt.get("artifact_set_id"),
        "artifact_set_sha256": receipt.get("artifact_set_sha256"),
        "cli_audio_sha256": cli["audio_sha256"],
        "http_audio_sha256": http["audio_sha256"],
        "exact_wav_equivalent": exact,
        "duration_delta_seconds": float(http["audio_duration_seconds"])
        - float(cli["audio_duration_seconds"]),
        "boundary": (
            "Exact equality establishes one matched deterministic row for the frozen artifacts and settings. "
            "It does not establish quality, throughput, cancellation, load behavior, or production readiness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("parity output already exists")
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
