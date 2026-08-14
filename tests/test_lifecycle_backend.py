from __future__ import annotations

import argparse
import io
import importlib.util
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from instavar_voice_lab.lifecycle import resolve_backend_spec
from instavar_voice_lab.lineage import build_dataset_lineage


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("qwen_lifecycle", ROOT / "scripts" / "instavar_voice_lifecycle.py")
assert SPEC and SPEC.loader
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)

FULL_SPEC = importlib.util.spec_from_file_location(
    "qwen_full_sft_lifecycle",
    ROOT / "scripts" / "instavar_voice_full_sft_lifecycle.py",
)
assert FULL_SPEC and FULL_SPEC.loader
FULL_LIFECYCLE = importlib.util.module_from_spec(FULL_SPEC)
FULL_SPEC.loader.exec_module(FULL_LIFECYCLE)

TRAIN_SPEC = importlib.util.spec_from_file_location(
    "qwen_full_sft_trainer",
    ROOT / "scripts" / "train_full_sft.py",
)
assert TRAIN_SPEC and TRAIN_SPEC.loader
FULL_TRAINER = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(FULL_TRAINER)


class LifecycleBackendTests(unittest.TestCase):
    def test_backend_commands_route_every_stage_through_wrapper(self) -> None:
        spec = json.loads((ROOT / "instavar-voice-backend.json").read_text())
        self.assertEqual(spec["schema_version"], "1.2.0")
        self.assertEqual(spec["capability_binding"]["adaptation"], "lora")
        required = {item["name"] for item in spec["required_environment"]}
        self.assertIn("PERSISTED_PACKAGE_ROOT", required)
        self.assertIn("package/persisted-package.json", spec["expected_artifacts"]["package"])
        for stage in ("preflight", "train", "infer", "evaluate", "package"):
            self.assertEqual(spec["commands"][stage][-1], stage)

    def test_backend_registry_separates_lora_and_full_sft(self) -> None:
        registry = json.loads((ROOT / "instavar-voice-backend-registry.json").read_text())
        self.assertEqual(registry["schema_version"], "1.0.0")
        self.assertEqual(
            {entry["backend_id"] for entry in registry["backends"]},
            {"qwen3-tts-lora-pytorch", "qwen3-tts-full-sft-pytorch"},
        )
        spec = json.loads((ROOT / "instavar-voice-backend-full-sft.json").read_text())
        self.assertEqual(spec["capability_binding"]["adaptation"], "full_sft")
        self.assertEqual(spec["capability_binding"]["runtime_ids"], ["pytorch_full_sft"])
        required = {item["name"] for item in spec["required_environment"]}
        self.assertIn("PERSISTED_PACKAGE_ROOT", required)
        self.assertIn("package/persisted-package.json", spec["expected_artifacts"]["package"])
        self.assertEqual(
            spec["environment"],
            {
                "DEVICE": "cuda:0",
                "DTYPE": "bf16",
                "MIXED_PRECISION": "bf16",
                "ATTN_IMPL": "flash_attention_2",
            },
        )
        for stage in ("preflight", "train", "infer", "evaluate", "package"):
            self.assertEqual(spec["commands"][stage][-1], stage)

    def test_registry_selects_full_sft_from_experiment_mode(self) -> None:
        import instavar_voice_lab

        evaluator_root = Path(instavar_voice_lab.__file__).parents[1]
        experiment = json.loads(
            (evaluator_root / "examples" / "experiment-manifest.json").read_text()
        )
        experiment["adaptation_mode"] = "full_sft"
        with tempfile.TemporaryDirectory() as temporary:
            experiment_path = Path(temporary) / "experiment.json"
            experiment_path.write_text(json.dumps(experiment))
            selected = resolve_backend_spec(
                ROOT / "instavar-voice-backend-registry.json",
                experiment_path,
            )
        self.assertEqual(selected.name, "instavar-voice-backend-full-sft.json")

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

    def test_archive_round_trip_supports_full_model_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "source"
            model.mkdir()
            (model / "config.json").write_text("{}\n")
            (model / "model.safetensors").write_bytes(b"weights")
            archive = root / "model.tar"
            LIFECYCLE._archive_directory(model, archive, arcname="model")
            extracted = LIFECYCLE._extract_archive(archive, root / "reload", arcname="model")
            self.assertEqual((extracted / "model.safetensors").read_bytes(), b"weights")

    def test_full_model_archive_excludes_optimizer_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model-source"
            (model / "resume-state").mkdir(parents=True)
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "resume-state" / "optimizer.bin").write_bytes(b"optimizer")
            archive = root / "model.tar"
            LIFECYCLE._archive_directory(
                model,
                archive,
                arcname="model",
                exclude_top_level=frozenset({"resume-state"}),
            )
            with tarfile.open(archive, "r") as handle:
                names = handle.getnames()
            self.assertIn("model/model.safetensors", names)
            self.assertFalse(any(name.startswith("model/resume-state") for name in names))

    def test_archive_rejects_members_outside_declared_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "outside.txt"
            source.write_text("outside\n")
            archive = root / "mixed.tar"
            with tarfile.open(archive, "w") as handle:
                handle.add(source, arcname="unexpected/outside.txt")
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                LIFECYCLE._extract_archive(archive, root / "reload", arcname="model")

    def test_archive_rejects_traversal_duplicates_and_special_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = {
                "traversal": [("model/../escape.bin", b"escape", tarfile.REGTYPE)],
                "duplicate": [
                    ("model/weights.bin", b"first", tarfile.REGTYPE),
                    ("model/weights.bin", b"second", tarfile.REGTYPE),
                ],
                "special": [("model/device", b"", tarfile.CHRTYPE)],
            }
            for name, entries in cases.items():
                archive_path = root / f"{name}.tar"
                with tarfile.open(archive_path, "w") as archive:
                    for member_name, payload, member_type in entries:
                        member = tarfile.TarInfo(member_name)
                        member.type = member_type
                        member.size = len(payload)
                        archive.addfile(member, io.BytesIO(payload))
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(ValueError, "unsafe archive member"),
                ):
                    LIFECYCLE._extract_archive(
                        archive_path, root / f"reload-{name}", arcname="model"
                    )

    def test_persist_package_is_mode_bound_content_addressed_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "package.tar"
            source.write_bytes(b"immutable package")
            store = root / "store"
            store.mkdir()
            first = LIFECYCLE._persist_package(source, store, "lora")
            destination = Path(first["persisted_path"])
            self.assertTrue(destination.name.startswith("qwen3-tts-lora-package-sha256-"))
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(first["reused_existing"])
            second = LIFECYCLE._persist_package(source, store, "lora")
            self.assertTrue(second["reused_existing"])
            full = LIFECYCLE._persist_package(source, store, "full_sft")
            self.assertNotEqual(full["persisted_path"], first["persisted_path"])
            destination.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                LIFECYCLE._persist_package(source, store, "lora")

    def test_package_root_is_bound_to_path_device_and_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            qwen = root / "qwen"
            base = root / "base"
            store = root / "store"
            other = root / "other"
            for path in (work, qwen, base, store, other):
                path.mkdir()
            identity = store.stat()
            environment = {
                "INSTAVAR_VOICE_WORK_DIR": str(work),
                "QWEN_DIR": str(qwen),
                "BASE_MODEL": str(base),
                "PERSISTED_PACKAGE_ROOT": str(store),
            }
            preflight = {
                "persistent_package_root": str(store.resolve()),
                "persistence_probe": {"device": identity.st_dev, "inode": identity.st_ino},
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertEqual(LIFECYCLE._locked_persistent_package_root(preflight), store.resolve())
                for changed in (
                    {**preflight, "persistent_package_root": str(other)},
                    {**preflight, "persistence_probe": {"device": identity.st_dev + 1, "inode": identity.st_ino}},
                    {**preflight, "persistence_probe": {"device": identity.st_dev, "inode": identity.st_ino + 1}},
                ):
                    with self.assertRaisesRegex(ValueError, "changed after preflight"):
                        LIFECYCLE._locked_persistent_package_root(changed)
                store.rename(root / "retired-store")
                store.mkdir()
                self.assertEqual(store.stat().st_dev, identity.st_dev)
                with self.assertRaisesRegex(ValueError, "changed after preflight"):
                    LIFECYCLE._locked_persistent_package_root(preflight)

    def test_persistent_package_root_rejects_work_source_and_local_model_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            qwen = root / "qwen"
            base = root / "base"
            for path in (work, qwen, base):
                path.mkdir()
                (path / "packages").mkdir()
            environment = {
                "INSTAVAR_VOICE_WORK_DIR": str(work),
                "QWEN_DIR": str(qwen),
                "BASE_MODEL": str(base),
            }
            for package_root, message in (
                (work / "packages", "outside the lifecycle work directory"),
                (qwen / "packages", "outside the Qwen checkout"),
                (base / "packages", "outside the local base-model input"),
            ):
                with (
                    patch.dict(
                        os.environ,
                        {**environment, "PERSISTED_PACKAGE_ROOT": str(package_root)},
                        clear=False,
                    ),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    LIFECYCLE._persistent_package_root()

    def test_persistence_probe_records_identity_and_owns_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = LIFECYCLE._probe_persistent_package_root(root)
            identity = root.stat()
            self.assertEqual(result["device"], identity.st_dev)
            self.assertEqual(result["inode"], identity.st_ino)
            self.assertEqual(list(root.iterdir()), [])
            with (
                patch.object(LIFECYCLE.os, "link", side_effect=FileExistsError("collision")),
                patch.object(Path, "unlink", autospec=True) as unlink,
                self.assertRaisesRegex(ValueError, "cannot publish an atomic package"),
            ):
                LIFECYCLE._probe_persistent_package_root(root)
            unlinked = [call.args[0] for call in unlink.call_args_list]
            self.assertEqual(len(unlinked), 1)
            self.assertTrue(str(unlinked[0]).endswith(".partial"))

    def test_full_sft_numeric_guards_reject_ood_values(self) -> None:
        for value in (float("nan"), float("inf"), 0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "finite and greater than 0"):
                FULL_TRAINER._positive_finite(value, "learning rate")
        for value in (-1, -100):
            with self.assertRaisesRegex(ValueError, "at least 0"):
                FULL_TRAINER._nonnegative(value, "speaker reference index")

    def test_full_sft_manifest_loader_rejects_empty_and_non_object_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "train.jsonl"
            manifest.write_text("\n")
            with self.assertRaisesRegex(ValueError, "contains no rows"):
                FULL_TRAINER._load_rows(manifest)
            manifest.write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
                FULL_TRAINER._load_rows(manifest)
            manifest.write_text('{"audio": "one.wav", "text": "one"}\n')
            self.assertEqual(FULL_TRAINER._load_rows(manifest)[0]["text"], "one")
            manifest.write_text(
                '{"audio": "one.wav", "text": "one"}\n'
                '{"audio": "two.wav", "text": "two"}\n'
            )
            self.assertEqual(len(FULL_TRAINER._load_rows(manifest, 1)), 1)
            with self.assertRaisesRegex(ValueError, "row limit"):
                FULL_TRAINER._load_rows(manifest, -1)

    def test_full_sft_resume_binds_contract_state_and_epoch_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            qwen = root / "qwen"
            (qwen / "finetuning").mkdir(parents=True)
            (qwen / "qwen_tts" / "inference").mkdir(parents=True)
            (qwen / "finetuning" / "dataset.py").write_text("# dataset\n")
            (qwen / "qwen_tts" / "inference" / "qwen3_tts_model.py").write_text(
                "# model loader\n"
            )
            train.write_text('{"audio": "train.wav", "text": "train"}\n')
            validation.write_text('{"audio": "validation.wav", "text": "validation"}\n')
            args = argparse.Namespace(
                init_model_path="Qwen/Qwen3-TTS-fixture",
                qwen_dir=qwen,
                train_jsonl=train,
                val_jsonl=validation,
                batch_size=2,
                eval_batch_size=1,
                learning_rate=2e-6,
                gradient_accumulation_steps=4,
                mixed_precision="bf16",
                attention="flash_attention_2",
                speaker_name="fixture",
                speaker_id=3000,
                speaker_reference_index=0,
                seed=42,
                save_every=1,
                eval_every=1,
                train_row_limit=0,
                validation_row_limit=0,
            )
            contract = FULL_TRAINER._training_contract(args)
            runtime = {
                "schema_version": "1.0.0",
                "python": "3.11.0",
                "torch": "2.8.0",
                "torch_cuda": "12.8",
                "accelerate": "1.10.0",
                "transformers": "4.57.0",
            }
            checkpoint = root / "checkpoint-epoch-0"
            state = checkpoint / "resume-state"
            state.mkdir(parents=True)
            (state / "model.safetensors").write_bytes(b"model")
            (state / "optimizer.bin").write_bytes(b"optimizer")
            metadata = {
                "schema_version": "1.1.0",
                "adaptation_mode": "full_sft",
                "completed_epochs": 1,
                "training_contract": contract,
                "runtime_contract": runtime,
                "resume_state": FULL_TRAINER._tree_manifest(state),
            }
            (checkpoint / "instavar-full-sft-metadata.json").write_text(
                json.dumps(metadata) + "\n"
            )

            with self.assertRaisesRegex(ValueError, "trust-resume-state"):
                FULL_TRAINER._resume_state(
                    checkpoint, contract, num_epochs=3, trust_resume_state=False
                )
            completed, state_path = FULL_TRAINER._resume_state(
                checkpoint,
                contract,
                num_epochs=3,
                trust_resume_state=True,
                expected_runtime_contract=runtime,
            )
            self.assertEqual(completed, 1)
            self.assertEqual(state_path, state.resolve())

            (state / "optimizer.bin").write_bytes(b"tampered")
            with self.assertRaisesRegex(
                ValueError, "does not match checkpoint metadata"
            ):
                FULL_TRAINER._resume_state(
                    checkpoint,
                    contract,
                    num_epochs=3,
                    trust_resume_state=True,
                    expected_runtime_contract=runtime,
                )

    def test_full_sft_evaluator_roles_are_live_bound_and_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def checkpoint(name: str, *, extra_rng: bool = False, hardlink: bool = False):
                candidate = root / name
                state = candidate / "resume-state"
                state.mkdir(parents=True)
                (state / "model.safetensors").write_bytes(b"model")
                (state / "optimizer.bin").write_bytes(b"optimizer")
                if hardlink:
                    os.link(state / "optimizer.bin", state / "scheduler.bin")
                else:
                    (state / "scheduler.bin").write_bytes(b"scheduler")
                (state / "random_states_0.pkl").write_bytes(b"rng")
                if extra_rng:
                    (state / "random_states_1.pkl").write_bytes(b"other-rng")
                trainer = candidate / "trainer-state.json"
                trainer.write_text('{"completed_epochs":2}\n')
                metadata = {
                    "schema_version": "1.2.0",
                    "adaptation_mode": "full_sft",
                    "resume_state": FULL_TRAINER._tree_manifest(state),
                    "trainer_state": FULL_TRAINER._file_manifest(
                        trainer, root=candidate
                    ),
                }
                (candidate / "instavar-full-sft-metadata.json").write_text(
                    json.dumps(metadata) + "\n"
                )
                return candidate

            complete = checkpoint("complete")
            artifacts = FULL_TRAINER.evaluator_full_sft_artifact_paths(complete)
            self.assertEqual(
                {role: path.name for role, path in artifacts.items()},
                {
                    "model_state": "model.safetensors",
                    "optimizer_state": "optimizer.bin",
                    "scheduler_state": "scheduler.bin",
                    "rng_state": "random_states_0.pkl",
                    "trainer_state": "trainer-state.json",
                },
            )

            with self.assertRaisesRegex(ValueError, "exactly one rng_state"):
                FULL_TRAINER.evaluator_full_sft_artifact_paths(
                    checkpoint("ambiguous", extra_rng=True)
                )
            with self.assertRaisesRegex(ValueError, "must not share hardlinks"):
                FULL_TRAINER.evaluator_full_sft_artifact_paths(
                    checkpoint("hardlinked", hardlink=True)
                )

    def test_full_sft_checkpoint_writes_bound_trainer_state(self) -> None:
        source = (ROOT / "scripts" / "train_full_sft.py").read_text()
        self.assertIn('trainer_state_path = output_dir / "trainer-state.json"', source)
        self.assertIn('"trainer_state": _file_manifest(', source)

    def test_full_sft_registers_and_steps_constant_scheduler(self) -> None:
        source = (ROOT / "scripts" / "train_full_sft.py").read_text()
        self.assertIn('"type": "constant_lambda"', source)
        self.assertIn("torch.optim.lr_scheduler.LambdaLR(", source)
        self.assertIn(
            "model, optimizer, train_loader, scheduler = accelerator.prepare(",
            source,
        )
        self.assertIn("validation_loader, scheduler = (", source)
        self.assertIn(
            "if accelerator.sync_gradients:\n                    scheduler.step()",
            source,
        )

    def test_full_sft_resume_rejects_contract_drift_and_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-epoch-0"
            state = checkpoint / "resume-state"
            state.mkdir(parents=True)
            (state / "state.bin").write_bytes(b"state")
            contract = {"schema_version": "1.0.0", "seed": 42}
            runtime = {"schema_version": "1.0.0", "python": "3.11.0"}
            metadata = {
                "schema_version": "1.1.0",
                "adaptation_mode": "full_sft",
                "completed_epochs": 1,
                "training_contract": contract,
                "runtime_contract": runtime,
                "resume_state": FULL_TRAINER._tree_manifest(state),
            }
            metadata_path = checkpoint / "instavar-full-sft-metadata.json"
            metadata_path.write_text(json.dumps(metadata) + "\n")

            with self.assertRaisesRegex(ValueError, "training contract"):
                FULL_TRAINER._resume_state(
                    checkpoint,
                    {**contract, "seed": 7},
                    num_epochs=3,
                    trust_resume_state=True,
                    expected_runtime_contract=runtime,
                )
            with self.assertRaisesRegex(ValueError, "must exceed"):
                FULL_TRAINER._resume_state(
                    checkpoint,
                    contract,
                    num_epochs=1,
                    trust_resume_state=True,
                    expected_runtime_contract=runtime,
                )
            with self.assertRaisesRegex(ValueError, "runtime contract"):
                FULL_TRAINER._resume_state(
                    checkpoint,
                    contract,
                    num_epochs=3,
                    trust_resume_state=True,
                    expected_runtime_contract={**runtime, "python": "3.14.0"},
                )

    def test_full_sft_training_contract_rejects_manifest_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            qwen = root / "qwen"
            (qwen / "finetuning").mkdir(parents=True)
            (qwen / "qwen_tts" / "inference").mkdir(parents=True)
            (qwen / "finetuning" / "dataset.py").write_text("# dataset\n")
            (qwen / "qwen_tts" / "inference" / "qwen3_tts_model.py").write_text(
                "# model loader\n"
            )
            train.write_text('{"audio": "train.wav", "text": "one"}\n')
            args = argparse.Namespace(
                init_model_path="Qwen/Qwen3-TTS-fixture",
                qwen_dir=qwen,
                train_jsonl=train,
                val_jsonl=None,
                batch_size=2,
                eval_batch_size=None,
                learning_rate=2e-6,
                gradient_accumulation_steps=4,
                mixed_precision="bf16",
                attention="flash_attention_2",
                speaker_name="fixture",
                speaker_id=3000,
                speaker_reference_index=0,
                seed=42,
                save_every=1,
                eval_every=1,
                train_row_limit=0,
                validation_row_limit=0,
            )
            before = FULL_TRAINER._training_contract(args)
            train.write_text('{"audio": "train.wav", "text": "two"}\n')
            after = FULL_TRAINER._training_contract(args)
            self.assertNotEqual(
                before["train_jsonl"]["sha256"], after["train_jsonl"]["sha256"]
            )

    def test_full_sft_training_contract_binds_local_base_model_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.jsonl"
            model = root / "model"
            qwen = root / "qwen"
            model.mkdir()
            (model / "config.json").write_text('{"model": "fixture"}\n')
            (qwen / "finetuning").mkdir(parents=True)
            (qwen / "qwen_tts" / "inference").mkdir(parents=True)
            (qwen / "finetuning" / "dataset.py").write_text("# dataset\n")
            (qwen / "qwen_tts" / "inference" / "qwen3_tts_model.py").write_text(
                "# model loader\n"
            )
            train.write_text('{"audio": "train.wav", "text": "one"}\n')
            args = argparse.Namespace(
                init_model_path=str(model),
                qwen_dir=qwen,
                train_jsonl=train,
                val_jsonl=None,
                batch_size=1,
                eval_batch_size=None,
                train_row_limit=1,
                validation_row_limit=0,
                learning_rate=2e-6,
                gradient_accumulation_steps=1,
                mixed_precision="bf16",
                attention="flash_attention_2",
                speaker_name="fixture",
                speaker_id=3000,
                speaker_reference_index=0,
                seed=42,
                save_every=1,
                eval_every=1,
            )
            before = FULL_TRAINER._training_contract(args)
            (model / "config.json").write_text('{"model": "changed"}\n')
            after = FULL_TRAINER._training_contract(args)
            self.assertEqual(before["init_model_artifact"]["kind"], "local_directory")
            self.assertNotEqual(
                before["init_model_artifact"]["manifest"]["sha256"],
                after["init_model_artifact"]["manifest"]["sha256"],
            )

    def test_full_sft_checkpoint_uses_source_compatible_config_for_dtype_bug(self) -> None:
        class Config:
            def __init__(self, key: str) -> None:
                self.key = key

            def to_diff_dict(self) -> dict:
                raise KeyError(self.key)

        class Model:
            def __init__(self, key: str) -> None:
                self.config = Config(key)
                self.saved = None

            def save_pretrained(self, output_dir, **kwargs) -> None:
                self.saved = (output_dir, self.config.to_diff_dict(), kwargs)

        model = Model("dtype")
        compatible = {
            "model_type": "qwen3_tts",
            "talker_config": {"hidden_size": 1024},
        }
        FULL_TRAINER._save_pretrained_with_compatible_config(
            model,
            Path("checkpoint"),
            {"weight": "fixture"},
            compatible,
        )
        self.assertEqual(model.saved[1]["model_type"], "qwen3_tts")
        self.assertEqual(model.saved[1]["talker_config"]["hidden_size"], 1024)
        with self.assertRaisesRegex(KeyError, "dtype"):
            model.config.to_diff_dict()
        with self.assertRaisesRegex(KeyError, "unexpected"):
            FULL_TRAINER._save_pretrained_with_compatible_config(
                Model("unexpected"), Path("checkpoint"), {}, compatible
            )

    def test_full_sft_checkpoint_config_changes_only_voice_registration(self) -> None:
        source = {
            "model_type": "qwen3_tts",
            "tts_model_type": "base",
            "speaker_encoder_config": {"hidden_size": 256},
            "talker_config": {
                "hidden_size": 1024,
                "spk_id": {},
                "spk_is_dialect": {},
            },
        }
        current = argparse.Namespace(
            talker_config=argparse.Namespace(
                spk_id={"female01": 3000},
                spk_is_dialect={"female01": False},
            )
        )
        result = FULL_TRAINER._checkpoint_config(source, current)
        self.assertEqual(result["tts_model_type"], "custom_voice")
        self.assertEqual(result["talker_config"]["spk_id"], {"female01": 3000})
        self.assertEqual(result["speaker_encoder_config"], {"hidden_size": 256})
        self.assertEqual(source["tts_model_type"], "base")

    def test_full_sft_checkpoint_copies_content_bound_speech_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text('{"codec": "fixture"}\n')
            (source / "model.safetensors").write_bytes(b"weights")
            target = root / "checkpoint" / "speech_tokenizer"
            manifest = FULL_TRAINER._copy_speech_tokenizer(source, target)
            self.assertEqual(
                (target / "model.safetensors").read_bytes(), b"weights"
            )
            self.assertEqual(manifest, FULL_TRAINER._tree_manifest(target))
            unsafe = root / "unsafe"
            unsafe.mkdir()
            (unsafe / "linked.json").symlink_to(source / "config.json")
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                FULL_TRAINER._copy_speech_tokenizer(
                    unsafe, root / "unsafe-checkpoint"
                )

    def test_full_sft_preflight_rejects_dirty_upstream_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "experiment.json"
            experiment.write_text(
                json.dumps(
                    {
                        "adaptation_mode": "full_sft",
                        "backend": {
                            "instavar_revision": "a" * 40,
                            "upstream_revision": "b" * 40,
                        },
                    }
                )
            )
            with (
                patch.object(FULL_LIFECYCLE, "_required_path", return_value=experiment),
                patch.object(
                    FULL_LIFECYCLE,
                    "_capture",
                    side_effect=["a" * 40, "b" * 40],
                ),
                patch.object(
                    FULL_LIFECYCLE,
                    "_git_status_paths",
                    side_effect=[set(), {"finetuning/sft_12hz.py"}],
                ),
                self.assertRaisesRegex(ValueError, "requires a clean Qwen checkout"),
            ):
                FULL_LIFECYCLE._verify_source_revisions(Path(temporary))

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
                audio.write_bytes(f"fixture:{split}".encode())
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
            store = root / "store"
            store.mkdir()
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
                "PERSISTED_PACKAGE_ROOT": str(store),
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
            self.assertEqual(report["persistent_package_root"], str(store.resolve()))
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
            (work_dir / "preflight").mkdir(parents=True)
            qwen = root / "qwen"
            store = root / "store"
            qwen.mkdir()
            store.mkdir()
            identity = store.stat()
            (work_dir / "preflight" / "preflight.json").write_text(
                json.dumps({"persistent_package_root": str(store.resolve()), "persistence_probe": {"device": identity.st_dev, "inode": identity.st_ino}})
            )
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
                "QWEN_DIR": str(qwen),
                "BASE_MODEL": "Qwen/Qwen3-TTS-fixture",
                "PERSISTED_PACKAGE_ROOT": str(store),
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
            receipt = json.loads((work_dir / "package" / "persisted-package.json").read_text())
            self.assertEqual(Path(receipt["persisted_path"]).read_bytes(), (work_dir / "package" / "adapter-package.tar").read_bytes())

    def test_full_sft_package_binds_checkpoint_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_dir = root / "work"
            (work_dir / "train").mkdir(parents=True)
            (work_dir / "evaluate").mkdir(parents=True)
            (work_dir / "preflight").mkdir(parents=True)
            qwen = root / "qwen"
            store = root / "store"
            qwen.mkdir()
            store.mkdir()
            identity = store.stat()
            (work_dir / "preflight" / "preflight.json").write_text(
                json.dumps({"persistent_package_root": str(store.resolve()), "persistence_probe": {"device": identity.st_dev, "inode": identity.st_ino}})
            )
            (work_dir / "train" / "selected-full-model.tar").write_bytes(b"checkpoint")
            (work_dir / "evaluate" / "evaluation-bundle.tar").write_bytes(b"evaluation")
            controls = {}
            for name in ("experiment", "plan", "lineage"):
                path = root / f"{name}.json"
                path.write_text("{}\n")
                controls[name] = path
            result = work_dir / "package" / "stage-result.json"
            environment = {
                "CANDIDATE_ID": "full-sft-candidate",
                "GENERATION_PLAN": str(controls["plan"]),
                "DATASET_LINEAGE": str(controls["lineage"]),
                "INSTAVAR_VOICE_EXPERIMENT_MANIFEST": str(controls["experiment"]),
                "INSTAVAR_VOICE_WORK_DIR": str(work_dir),
                "INSTAVAR_VOICE_STAGE_RESULT": str(result),
                "QWEN_DIR": str(qwen),
                "BASE_MODEL": "Qwen/Qwen3-TTS-fixture",
                "PERSISTED_PACKAGE_ROOT": str(store),
            }
            with patch.dict(os.environ, environment, clear=False):
                FULL_LIFECYCLE.run("package")
            package = work_dir / "package" / "full-sft-package.tar"
            with tarfile.open(package, "r") as archive:
                names = set(archive.getnames())
                manifest = json.load(archive.extractfile("package/package-manifest.json"))
            self.assertIn("package/selected-full-model.tar", names)
            self.assertIn("package/evaluation-bundle.tar", names)
            self.assertEqual(manifest["backend_id"], "qwen3-tts-full-sft-pytorch")
            receipt = json.loads((work_dir / "package" / "persisted-package.json").read_text())
            self.assertEqual(Path(receipt["persisted_path"]).read_bytes(), package.read_bytes())
            self.assertEqual(json.loads(result.read_text())["stage"], "package")


if __name__ == "__main__":
    unittest.main()
