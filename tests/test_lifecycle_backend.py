from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from instavar_voice_lab.lineage import build_dataset_lineage


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("qwen_lifecycle", ROOT / "scripts" / "instavar_voice_lifecycle.py")
assert SPEC and SPEC.loader
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)


class LifecycleBackendTests(unittest.TestCase):
    def test_backend_commands_route_every_stage_through_wrapper(self) -> None:
        spec = json.loads((ROOT / "instavar-voice-backend.json").read_text())
        self.assertEqual(spec["schema_version"], "1.2.0")
        self.assertEqual(spec["capability_binding"]["adaptation"], "lora")
        for stage in ("preflight", "train", "infer", "evaluate", "package"):
            self.assertEqual(spec["commands"][stage][-1], stage)

    def test_selected_adapter_must_be_one_safe_child(self) -> None:
        self.assertEqual(LIFECYCLE._safe_child_name("checkpoint-epoch-3"), "checkpoint-epoch-3")
        for unsafe in ("", ".", "..", "../checkpoint", "nested/checkpoint", "/absolute"):
            with self.assertRaises(ValueError):
                LIFECYCLE._safe_child_name(unsafe)

    def test_archive_rejects_symlinked_adapter_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "adapter"
            root.mkdir()
            target = Path(temporary) / "outside.bin"
            target.write_bytes(b"outside")
            (root / "adapter_model.bin").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                LIFECYCLE._archive_directory(root, Path(temporary) / "adapter.tar", arcname="adapter")

    def test_patched_upstream_must_equal_head_plus_exact_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary) / "upstream"
            repository.mkdir()
            for command in (
                ["git", "init", "-q"],
                ["git", "config", "user.name", "Fixture"],
                ["git", "config", "user.email", "fixture@example.invalid"],
            ):
                subprocess.run(command, cwd=repository, check=True)
            tracked = repository / "tracked.txt"
            tracked.write_text("old\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
            patch_file = Path(temporary) / "change.patch"
            patch_file.write_text(
                "diff --git a/tracked.txt b/tracked.txt\n"
                "--- a/tracked.txt\n"
                "+++ b/tracked.txt\n"
                "@@ -1 +1 @@\n"
                "-old\n"
                "+new\n"
            )
            subprocess.run(["git", "apply", str(patch_file)], cwd=repository, check=True)
            evidence = LIFECYCLE._verify_patched_upstream(repository, patch_file)
            self.assertEqual(evidence["patched_paths"], ["tracked.txt"])
            (repository / "unexpected.txt").write_text("unexpected\n")
            with self.assertRaisesRegex(ValueError, "outside the pinned companion patch"):
                LIFECYCLE._verify_patched_upstream(repository, patch_file)

    def test_preflight_audits_three_splits_and_frozen_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qwen_dir = root / "qwen" / "finetuning"
            qwen_dir.mkdir(parents=True)
            (qwen_dir / "sft_12hz_lora.py").write_text("# fixture\n")
            (qwen_dir / "infer_lora_custom_voice.py").write_text("# fixture\n")
            manifests: dict[str, str] = {}
            for split in ("train", "validation", "test"):
                audio = root / f"{split}.wav"
                audio.write_bytes(b"fixture")
                manifest = root / f"{split}.jsonl"
                manifest.write_text(json.dumps({"audio": str(audio), "text": f"{split} text"}) + "\n")
                manifests[split] = str(manifest)
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "samples": [{"candidate_id": "candidate", "sample_id": "one"}],
                    }
                )
            )
            work_dir = root / "work"
            (work_dir / "preflight").mkdir(parents=True)
            result = work_dir / "preflight" / "stage-result.json"
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
            lineage = root / "dataset-lineage.json"
            lineage.write_text(
                json.dumps(
                    build_dataset_lineage(
                        lineage_id="qwen-fixture-v1",
                        producer_repository="instavar/qwen3-tts-lora-finetuning",
                        producer_revision=revision,
                        inputs={
                            "raw_train": (Path(manifests["train"]), "file"),
                            "raw_validation": (Path(manifests["validation"]), "file"),
                            "raw_test": (Path(manifests["test"]), "file"),
                        },
                        outputs={
                            "training_train": (Path(manifests["train"]), "file"),
                            "training_validation": (Path(manifests["validation"]), "file"),
                            "training_test": (Path(manifests["test"]), "file"),
                        },
                    )
                )
            )
            environment = {
                "QWEN_DIR": str(qwen_dir.parent),
                "BASE_MODEL": str(root / "base-model"),
                "TRAIN_JSONL": manifests["train"],
                "VAL_JSONL": manifests["validation"],
                "TEST_JSONL": manifests["test"],
                "DATASET_LINEAGE": str(lineage),
                "SELECTED_ADAPTER_NAME": "checkpoint-epoch-3",
                "GENERATION_PLAN": str(plan),
                "CANDIDATE_ID": "candidate",
                "INSTAVAR_VOICE_WORK_DIR": str(work_dir),
                "INSTAVAR_VOICE_STAGE_RESULT": str(result),
            }
            revision_evidence = {
                "companion_revision": revision,
                "upstream_revision": "b" * 40,
                "patch_sha256": "c" * 64,
                "patched_paths": ["finetuning/sft_12hz_lora.py"],
            }
            with (
                patch.dict(os.environ, environment, clear=False),
                patch.object(LIFECYCLE, "_verify_source_revisions", return_value=revision_evidence),
            ):
                LIFECYCLE.run("preflight")
            report = json.loads((work_dir / "preflight" / "preflight.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["generation_rows"], 1)
            self.assertEqual(report["source_revisions"], revision_evidence)
            self.assertEqual(report["dataset_lineage"]["lineage_id"], "qwen-fixture-v1")
            self.assertEqual(json.loads(result.read_text())["stage"], "preflight")

            Path(manifests["train"]).write_text(
                json.dumps({"audio": str(root / "train.wav"), "text": "changed"}) + "\n"
            )
            with (
                patch.dict(os.environ, environment, clear=False),
                self.assertRaisesRegex(ValueError, "raw_train"),
            ):
                LIFECYCLE._verify_dataset_lineage()

    def test_package_bundles_adapter_evaluation_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_dir = root / "work"
            (work_dir / "train").mkdir(parents=True)
            (work_dir / "evaluate").mkdir(parents=True)
            (work_dir / "train" / "selected-adapter.tar").write_bytes(b"adapter")
            (work_dir / "evaluate" / "evaluation-bundle.tar").write_bytes(b"evaluation")
            experiment = root / "experiment.json"
            experiment.write_text("{}\n")
            plan = root / "plan.json"
            plan.write_text("{}\n")
            lineage = root / "dataset-lineage.json"
            lineage.write_text("{}\n")
            result = work_dir / "package" / "stage-result.json"
            environment = {
                "CANDIDATE_ID": "candidate",
                "GENERATION_PLAN": str(plan),
                "DATASET_LINEAGE": str(lineage),
                "INSTAVAR_VOICE_EXPERIMENT_MANIFEST": str(experiment),
                "INSTAVAR_VOICE_WORK_DIR": str(work_dir),
                "INSTAVAR_VOICE_STAGE_RESULT": str(result),
            }
            with patch.dict(os.environ, environment, clear=False):
                LIFECYCLE.run("package")
            with tarfile.open(work_dir / "package" / "adapter-package.tar", "r") as archive:
                names = set(archive.getnames())
            self.assertIn("package/selected-adapter.tar", names)
            self.assertIn("package/evaluation-bundle.tar", names)
            self.assertIn("package/experiment-manifest.json", names)
            self.assertIn("package/dataset-lineage.json", names)
            self.assertIn("package/generation-plan.json", names)
            self.assertIn("package/package-manifest.json", names)


if __name__ == "__main__":
    unittest.main()
