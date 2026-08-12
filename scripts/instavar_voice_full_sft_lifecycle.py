#!/usr/bin/env python3
"""Execute Qwen3-TTS full SFT through the Instavar Voice lifecycle."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from instavar_voice_lifecycle import (  # noqa: E402
    _archive_directory,
    _capture,
    _extract_archive,
    _git_status_paths,
    _required_path,
    _run,
    _safe_child_name,
    _sha256,
    _stage_result,
    _verify_dataset_lineage,
    _work_dir,
    _write_json,
)

REPO_ROOT = Path(__file__).parents[1]
STAGES = {"preflight", "train", "infer", "evaluate", "package"}


def _verify_source_revisions(qwen_dir: Path) -> dict[str, Any]:
    experiment = json.loads(
        _required_path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST").read_text(encoding="utf-8")
    )
    if experiment.get("adaptation_mode") != "full_sft":
        raise ValueError("experiment adaptation_mode must equal full_sft")
    backend = experiment.get("backend", {})
    companion_revision = _capture(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    companion_dirty = _git_status_paths(REPO_ROOT)
    if companion_dirty:
        raise ValueError(
            "companion repository must be clean: " + ", ".join(sorted(companion_dirty))
        )
    upstream_revision = _capture(["git", "rev-parse", "HEAD"], cwd=qwen_dir)
    upstream_dirty = _git_status_paths(qwen_dir)
    if upstream_dirty:
        raise ValueError(
            "full SFT requires a clean Qwen checkout: "
            + ", ".join(sorted(upstream_dirty))
        )
    if backend.get("instavar_revision") != companion_revision:
        raise ValueError(
            "experiment backend.instavar_revision does not match the companion checkout"
        )
    if backend.get("upstream_revision") != upstream_revision:
        raise ValueError(
            "experiment backend.upstream_revision does not match the Qwen checkout"
        )
    return {
        "companion_revision": companion_revision,
        "upstream_revision": upstream_revision,
        "upstream_checkout": "clean",
        "trainer_sha256": _sha256(REPO_ROOT / "scripts" / "train_full_sft.py"),
    }


def _generation_rows() -> tuple[Path, str, list[dict[str, Any]]]:
    plan_path = _required_path("GENERATION_PLAN")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    candidate_id = os.environ["CANDIDATE_ID"]
    rows = [
        row
        for row in plan.get("samples", [])
        if row.get("candidate_id") == candidate_id
    ]
    if plan.get("schema_version") not in {"1.0.0", "1.1.0"} or not rows:
        raise ValueError(
            "GENERATION_PLAN must be schema 1.0.0 or 1.1.0 and contain "
            "CANDIDATE_ID rows"
        )
    return plan_path, candidate_id, rows


def _preflight() -> None:
    from instavar_voice_lab.corpus import audit_corpus

    qwen_dir = _required_path("QWEN_DIR", directory=True)
    required_upstream = [
        qwen_dir / "finetuning" / "dataset.py",
        qwen_dir / "qwen_tts" / "inference" / "qwen3_tts_model.py",
    ]
    missing = [str(path) for path in required_upstream if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required Qwen source files are missing: " + ", ".join(missing)
        )
    source_revisions = _verify_source_revisions(qwen_dir)
    lineage = _verify_dataset_lineage()
    splits = {
        "train": _required_path("TRAIN_JSONL"),
        "validation": _required_path("VAL_JSONL"),
        "test": _required_path("TEST_JSONL"),
    }
    audit = audit_corpus(
        splits, group_field=os.environ.get("CORPUS_GROUP_FIELD") or None
    )
    if audit["status"] != "passed":
        raise ValueError("corpus audit failed: " + "; ".join(audit["errors"]))
    _, candidate_id, rows = _generation_rows()
    selected_name = _safe_child_name(
        os.environ["SELECTED_CHECKPOINT_NAME"],
        environment_name="SELECTED_CHECKPOINT_NAME",
    )
    _write_json(
        _work_dir() / "preflight" / "preflight.json",
        {
            "schema_version": "1.0.0",
            "status": "passed",
            "adaptation_mode": "full_sft",
            "candidate_id": candidate_id,
            "generation_rows": len(rows),
            "selected_checkpoint_name": selected_name,
            "corpus_audit": audit,
            "qwen_dir": str(qwen_dir),
            "base_model": os.environ["BASE_MODEL"],
            "source_revisions": source_revisions,
            "dataset_lineage": lineage,
            "evidence_boundary": (
                "Preflight does not prove that training fits available GPU memory."
            ),
        },
    )


def _train() -> None:
    _verify_dataset_lineage()
    work_dir = _work_dir()
    output_dir = work_dir / "train" / "output"
    environment = os.environ.copy()
    environment.update(
        {
            "OUTPUT_DIR": str(output_dir),
            "INIT_MODEL_PATH": os.environ["BASE_MODEL"],
            "PYTHON": sys.executable,
        }
    )
    _run(["bash", "scripts/run_full_sft_train.sh"], environment=environment)
    selected = output_dir / _safe_child_name(
        os.environ["SELECTED_CHECKPOINT_NAME"],
        environment_name="SELECTED_CHECKPOINT_NAME",
    )
    _archive_directory(
        selected, work_dir / "train" / "selected-full-model.tar", arcname="model"
    )


def _infer() -> None:
    work_dir = _work_dir()
    model = _extract_archive(
        work_dir / "train" / "selected-full-model.tar",
        work_dir / "infer" / "reload",
        arcname="model",
    )
    output = work_dir / "infer" / "candidate.wav"
    _run(
        [
            sys.executable,
            "scripts/run_full_sft_infer.py",
            "--qwen-dir",
            os.environ["QWEN_DIR"],
            "--model",
            str(model),
            "--speaker-name",
            os.environ.get("SPEAKER_NAME", "speaker"),
            "--output-wav",
            str(output),
            "--seed",
            os.environ.get("SEED", "42"),
            "--max-new-tokens",
            os.environ.get("MAX_NEW_TOKENS", "4096"),
            "--device",
            os.environ.get("DEVICE", "cuda:0"),
            "--dtype",
            os.environ.get("DTYPE", "bf16"),
            "--attention",
            os.environ.get("ATTN_IMPL", "flash_attention_2"),
        ]
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("fresh-process full-model inference did not produce audio")


def _evaluate() -> None:
    work_dir = _work_dir()
    model = _extract_archive(
        work_dir / "train" / "selected-full-model.tar",
        work_dir / "evaluate" / "reload",
        arcname="model",
    )
    output_dir = work_dir / "evaluate" / "output"
    plan, candidate_id, _ = _generation_rows()
    command = [
        sys.executable,
        "scripts/run_evaluation_suite.py",
        "--qwen-dir",
        os.environ["QWEN_DIR"],
        "--model",
        str(model),
        "--generation-plan",
        str(plan),
        "--candidate-id",
        candidate_id,
        "--runtime-id",
        "pytorch_full_sft",
        "--output-dir",
        str(output_dir),
        "--speaker-name",
        os.environ.get("SPEAKER_NAME", "speaker"),
        "--device",
        os.environ.get("DEVICE", "cuda:0"),
        "--dtype",
        os.environ.get("DTYPE", "bf16"),
        "--attention",
        os.environ.get("ATTN_IMPL", "flash_attention_2"),
        "--max-new-tokens",
        os.environ.get("MAX_NEW_TOKENS", "4096"),
        "--allow-invalid-output",
    ]
    _run(command)
    raw_observations = output_dir / "generation-observations.json"
    receipt = output_dir / "generation-attempt-receipt.json"
    bound_observations = output_dir / "objective-observations.json"
    producer_revision = _capture(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    _run(
        [
            sys.executable,
            "-m",
            "instavar_voice_lab.cli",
            "build-generation-attempt-receipt",
            str(raw_observations),
            "--plan",
            str(plan),
            "--audio-base-dir",
            str(output_dir),
            "--producer-name",
            "qwen3-full-sft-evaluation-runner",
            "--producer-revision",
            producer_revision,
            "--output",
            str(receipt),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "instavar_voice_lab.cli",
            "apply-generation-attempt-receipt",
            str(raw_observations),
            str(receipt),
            "--plan",
            str(plan),
            "--audio-base-dir",
            str(output_dir),
            "--output",
            str(bound_observations),
        ]
    )
    _archive_directory(
        output_dir,
        work_dir / "evaluate" / "evaluation-bundle.tar",
        arcname="evaluation",
    )


def _package() -> None:
    work_dir = _work_dir()
    staging = work_dir / "package" / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    sources = {
        "selected-full-model.tar": work_dir / "train" / "selected-full-model.tar",
        "evaluation-bundle.tar": work_dir / "evaluate" / "evaluation-bundle.tar",
        "experiment-manifest.json": _required_path(
            "INSTAVAR_VOICE_EXPERIMENT_MANIFEST"
        ),
        "generation-plan.json": _required_path("GENERATION_PLAN"),
        "dataset-lineage.json": _required_path("DATASET_LINEAGE"),
    }
    records: list[dict[str, Any]] = []
    for name, source in sources.items():
        target = staging / name
        shutil.copyfile(source, target)
        if _sha256(source) != _sha256(target):
            raise ValueError(f"packaged evidence does not match its source: {name}")
        records.append(
            {"path": name, "sha256": _sha256(target), "bytes": target.stat().st_size}
        )
    _write_json(
        staging / "package-manifest.json",
        {
            "schema_version": "1.0.0",
            "backend_id": "qwen3-tts-full-sft-pytorch",
            "candidate_id": os.environ["CANDIDATE_ID"],
            "files": records,
            "evidence_boundary": (
                "This research bundle is not a distribution approval or perceptual "
                "quality claim."
            ),
        },
    )
    _archive_directory(
        staging, work_dir / "package" / "full-sft-package.tar", arcname="package"
    )


def run(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown lifecycle stage: {stage}")
    {
        "preflight": _preflight,
        "train": _train,
        "infer": _infer,
        "evaluate": _evaluate,
        "package": _package,
    }[stage]()
    if stage in {"preflight", "train"}:
        _verify_dataset_lineage()
    _write_json(
        _stage_result(), {"schema_version": "1.0.0", "stage": stage, "status": "passed"}
    )


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        print("usage: instavar_voice_full_sft_lifecycle.py STAGE", file=sys.stderr)
        return 2
    try:
        run(values[0])
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        tarfile.TarError,
    ) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
