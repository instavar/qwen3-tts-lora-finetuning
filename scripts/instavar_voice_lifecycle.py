#!/usr/bin/env python3
"""Execute the Qwen3-TTS LoRA path through the Instavar Voice lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).parents[1]
STAGES = {"preflight", "train", "infer", "evaluate", "package"}


def _required_path(name: str, *, directory: bool = False) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    path = Path(value).expanduser().resolve()
    if directory and not path.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"{name} file not found: {path}")
    return path


def _work_dir() -> Path:
    return _required_path("INSTAVAR_VOICE_WORK_DIR", directory=True)


def _stage_result() -> Path:
    value = os.environ.get("INSTAVAR_VOICE_STAGE_RESULT", "").strip()
    if not value:
        raise ValueError("INSTAVAR_VOICE_STAGE_RESULT is required")
    return Path(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_child_name(value: str) -> str:
    path = Path(value)
    if not value or value in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise ValueError("SELECTED_ADAPTER_NAME must be one safe child directory name")
    return value


def _check_archive_source(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"adapter source must be a non-symlink directory: {root}")
    files = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"adapter source must not contain symlinks: {path}")
        if path.is_file():
            files += 1
        elif not path.is_dir():
            raise ValueError(f"adapter source contains an unsupported entry: {path}")
    if files == 0:
        raise ValueError(f"adapter source contains no files: {root}")


def _archive_directory(source: Path, destination: Path, *, arcname: str) -> None:
    _check_archive_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as archive:
        archive.add(source, arcname=arcname, recursive=True)


def _extract_archive(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("adapter archive is empty")
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise ValueError(f"unsafe adapter archive member: {member.name}")
        archive.extractall(destination, members=members, filter="data")
    adapter = destination / "adapter"
    _check_archive_source(adapter)
    return adapter


def _run(command: list[str], *, environment: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command[0]}")


def _capture(command: list[str], *, cwd: Path, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command[0]}: {detail}")
    return result.stdout.strip()


def _git_status_paths(repository: Path) -> set[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    output = result.stdout
    records = [record for record in output.split("\0") if record]
    paths: set[str] = set()
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 4:
            raise ValueError("unexpected git status record")
        status = record[:2]
        paths.add(record[3:])
        index += 2 if "R" in status or "C" in status else 1
    return paths


def _patch_paths(patch: Path) -> set[str]:
    paths: set[str] = set()
    for line in patch.read_text(encoding="utf-8").splitlines():
        for prefix in ("+++ b/", "--- a/"):
            if line.startswith(prefix):
                paths.add(line[len(prefix) :])
    if not paths:
        raise ValueError(f"patch contains no repository paths: {patch}")
    return paths


def _verify_patched_upstream(qwen_dir: Path, patch: Path) -> dict[str, Any]:
    touched = _patch_paths(patch)
    dirty = _git_status_paths(qwen_dir)
    unexpected = sorted(dirty - touched)
    if unexpected:
        raise ValueError("Qwen checkout has changes outside the pinned companion patch: " + ", ".join(unexpected))
    with tempfile.TemporaryDirectory() as temporary:
        index_path = Path(temporary) / "index"
        environment = os.environ.copy()
        environment["GIT_INDEX_FILE"] = str(index_path)
        _capture(["git", "read-tree", "HEAD"], cwd=qwen_dir, environment=environment)
        _capture(["git", "apply", "--cached", str(patch)], cwd=qwen_dir, environment=environment)
        for relative in sorted(touched):
            expected = _capture(["git", "ls-files", "--stage", "--", relative], cwd=qwen_dir, environment=environment)
            path = qwen_dir / relative
            if not expected:
                if path.exists() or path.is_symlink():
                    raise ValueError(f"patched checkout should delete {relative}")
                continue
            expected_blob = expected.split()[1]
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"patched checkout file is missing or unsafe: {relative}")
            observed_blob = _capture(["git", "hash-object", "--", relative], cwd=qwen_dir)
            if observed_blob != expected_blob:
                raise ValueError(f"Qwen checkout does not match the pinned companion patch: {relative}")
    return {
        "upstream_revision": _capture(["git", "rev-parse", "HEAD"], cwd=qwen_dir),
        "patch_sha256": _sha256(patch),
        "patched_paths": sorted(touched),
    }


def _verify_source_revisions(qwen_dir: Path) -> dict[str, Any]:
    experiment_path = _required_path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    backend = experiment.get("backend", {})
    companion_revision = _capture(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    companion_dirty = _git_status_paths(REPO_ROOT)
    if companion_dirty:
        raise ValueError("companion repository must be clean: " + ", ".join(sorted(companion_dirty)))
    patch_evidence = _verify_patched_upstream(qwen_dir, REPO_ROOT / "patches" / "0001-qwen3-tts-lora.patch")
    if backend.get("instavar_revision") != companion_revision:
        raise ValueError("experiment backend.instavar_revision does not match the companion checkout")
    if backend.get("upstream_revision") != patch_evidence["upstream_revision"]:
        raise ValueError("experiment backend.upstream_revision does not match the Qwen checkout")
    return {"companion_revision": companion_revision, **patch_evidence}


def _preflight() -> None:
    from instavar_voice_lab.corpus import audit_corpus

    qwen_dir = _required_path("QWEN_DIR", directory=True)
    required_upstream = [
        qwen_dir / "finetuning" / "sft_12hz_lora.py",
        qwen_dir / "finetuning" / "infer_lora_custom_voice.py",
    ]
    missing = [str(path) for path in required_upstream if not path.is_file()]
    if missing:
        raise FileNotFoundError("Qwen companion patches are missing: " + ", ".join(missing))
    source_revisions = _verify_source_revisions(qwen_dir)
    splits = {
        "train": _required_path("TRAIN_JSONL"),
        "validation": _required_path("VAL_JSONL"),
        "test": _required_path("TEST_JSONL"),
    }
    audit = audit_corpus(splits, group_field=os.environ.get("CORPUS_GROUP_FIELD") or None)
    if audit["status"] != "passed":
        raise ValueError("corpus audit failed: " + "; ".join(audit["errors"]))
    plan_path = _required_path("GENERATION_PLAN")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    candidate_id = os.environ["CANDIDATE_ID"]
    rows = [row for row in plan.get("samples", []) if row.get("candidate_id") == candidate_id]
    if plan.get("schema_version") != "1.0.0" or not rows:
        raise ValueError("GENERATION_PLAN must be schema 1.0.0 and contain CANDIDATE_ID rows")
    selected_name = _safe_child_name(os.environ["SELECTED_ADAPTER_NAME"])
    _write_json(
        _work_dir() / "preflight" / "preflight.json",
        {
            "schema_version": "1.0.0",
            "status": "passed",
            "candidate_id": candidate_id,
            "generation_rows": len(rows),
            "selected_adapter_name": selected_name,
            "corpus_audit": audit,
            "qwen_dir": str(qwen_dir),
            "base_model": os.environ["BASE_MODEL"],
            "source_revisions": source_revisions,
        },
    )


def _train() -> None:
    work_dir = _work_dir()
    output_dir = work_dir / "train" / "output"
    environment = os.environ.copy()
    environment.update(
        {
            "OUTPUT_DIR": str(output_dir),
            "INIT_MODEL_PATH": os.environ["BASE_MODEL"],
            "AUDIT_CORPUS": "0",
            "PYTHON": sys.executable,
        }
    )
    _run(["bash", "scripts/run_lora_train.sh"], environment=environment)
    selected = output_dir / _safe_child_name(os.environ["SELECTED_ADAPTER_NAME"])
    _archive_directory(selected, work_dir / "train" / "selected-adapter.tar", arcname="adapter")


def _infer() -> None:
    work_dir = _work_dir()
    adapter = _extract_archive(work_dir / "train" / "selected-adapter.tar", work_dir / "infer" / "reload")
    output = work_dir / "infer" / "candidate.wav"
    environment = os.environ.copy()
    environment.update(
        {
            "ADAPTER_DIR": str(adapter),
            "OUT_WAV": str(output),
            "PYTHON": sys.executable,
        }
    )
    _run(["bash", "scripts/run_lora_infer.sh"], environment=environment)
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("fresh-process adapter inference did not produce audio")


def _evaluate() -> None:
    work_dir = _work_dir()
    adapter = _extract_archive(work_dir / "train" / "selected-adapter.tar", work_dir / "evaluate" / "reload")
    output_dir = work_dir / "evaluate" / "output"
    command = [
        sys.executable,
        "scripts/run_evaluation_suite.py",
        "--qwen-dir",
        os.environ["QWEN_DIR"],
        "--base-model",
        os.environ["BASE_MODEL"],
        "--adapter",
        str(adapter),
        "--generation-plan",
        os.environ["GENERATION_PLAN"],
        "--candidate-id",
        os.environ["CANDIDATE_ID"],
        "--output-dir",
        str(output_dir),
        "--speaker-name",
        os.environ.get("SPEAKER_NAME", "speaker"),
        "--lora-scale",
        os.environ.get("LORA_SCALE", "0.3"),
    ]
    _run(command)
    _archive_directory(output_dir, work_dir / "evaluate" / "evaluation-bundle.tar", arcname="evaluation")


def _package() -> None:
    work_dir = _work_dir()
    staging = work_dir / "package" / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    sources = {
        "selected-adapter.tar": work_dir / "train" / "selected-adapter.tar",
        "evaluation-bundle.tar": work_dir / "evaluate" / "evaluation-bundle.tar",
        "experiment-manifest.json": _required_path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST"),
        "generation-plan.json": _required_path("GENERATION_PLAN"),
    }
    records: list[dict[str, Any]] = []
    for name, source in sources.items():
        target = staging / name
        shutil.copyfile(source, target)
        if _sha256(source) != _sha256(target):
            raise ValueError(f"packaged evidence does not match its source: {name}")
        records.append({"path": name, "sha256": _sha256(target), "bytes": target.stat().st_size})
    _write_json(
        staging / "package-manifest.json",
        {
            "schema_version": "1.0.0",
            "backend_id": "qwen3-tts-lora-pytorch",
            "candidate_id": os.environ["CANDIDATE_ID"],
            "files": records,
            "evidence_boundary": "This bundle preserves lifecycle artifacts and provenance; it does not prove perceptual quality or distribution rights.",
        },
    )
    destination = work_dir / "package" / "adapter-package.tar"
    _archive_directory(staging, destination, arcname="package")


def run(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown lifecycle stage: {stage}")
    {"preflight": _preflight, "train": _train, "infer": _infer, "evaluate": _evaluate, "package": _package}[stage]()
    _write_json(_stage_result(), {"schema_version": "1.0.0", "stage": stage, "status": "passed"})


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        print("usage: instavar_voice_lifecycle.py STAGE", file=sys.stderr)
        return 2
    try:
        run(values[0])
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError, tarfile.TarError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
