import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import ser_protocol


class SerProtocolTest(unittest.TestCase):
    def make_split(self, data_root: Path, dataset: str, fold: int, split: str, samples: int) -> Path:
        directory = data_root / "manifests" / dataset / ser_protocol.fold_dir_name(fold)
        directory.mkdir(parents=True, exist_ok=True)
        rows = ["/root/songhao/datasets/SER_DATASET"]
        labels = []
        for index in range(samples):
            rows.append(f"{dataset}/sample_{index}.wav\t16000")
            labels.append(f"sample_{index}\tneutral")
        (directory / f"{split}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (directory / f"{split}.lbl").write_text("\n".join(labels) + "\n", encoding="utf-8")
        return directory

    def make_earlystop_splits(self, data_root: Path, dataset: str, fold: int) -> Path:
        self.make_split(data_root, dataset, fold, "inner_train", 33)
        self.make_split(data_root, dataset, fold, "inner_valid", 4)
        return data_root / "manifests" / dataset / ser_protocol.fold_dir_name(fold)

    def make_stats(self, data_root: Path, dataset: str, fold: int, stage: str = "selection") -> Path:
        directory = data_root / "norm_stats" / dataset / stage
        directory.mkdir(parents=True, exist_ok=True)
        split = "inner_train" if stage == "selection" else "train"
        manifest = (
            data_root / "manifests" / dataset
            / ser_protocol.fold_dir_name(fold) / f"{split}.tsv"
        )
        stats = {
            "stats_schema": "eat2emo-ser-fbank-v1",
            "dataset": dataset,
            "fold": fold,
            "stage": stage,
            "split": split,
            "target_length": ser_protocol.DATASETS[dataset].target_length,
            "mel_bins": 128,
            "sample_rate": 16000,
            "normalization_scope": "valid_frames",
            "padding_included": False,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "mel": {"mean": -5.0, "std": 4.0},
        }
        path = directory / f"{ser_protocol.fold_dir_name(fold)}.json"
        path.write_text(json.dumps(stats), encoding="utf-8")
        return path

    def run_main(self, arguments):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = ser_protocol.main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_frozen_dataset_defaults(self):
        expected = {
            "emodb": (7, 640, 1, 10, 80),
            "iemocap": (4, 1024, 1, 5, 20),
            "ravdess-speech": (8, 512, 0, 4, 80),
            "ravdess-song": (6, 640, 0, 4, 80),
        }
        actual = {
            name: (
                spec.num_classes,
                spec.target_length,
                spec.first_fold,
                spec.last_fold,
                spec.max_epochs,
            )
            for name, spec in ser_protocol.DATASETS.items()
        }
        self.assertEqual(actual, expected)

    def test_update_schedule_uses_manifest_batches_and_ten_percent_warmup(self):
        self.assertEqual(ser_protocol.compute_schedule(33, 60), (2, 120, 12))
        self.assertEqual(ser_protocol.compute_schedule(1, 1), (1, 1, 0))

    def test_manifest_count_rejects_extra_columns_and_nonpositive_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "train.tsv"
            for entry in ("clip.wav\t0", "clip.wav\t16000\textra"):
                manifest.write_text(f"/audio\n{entry}\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Malformed manifest line 2"):
                    ser_protocol.count_manifest_samples(manifest)

    def test_earlystop_dry_run_uses_inner_splits_and_best_checkpoint_eval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_earlystop_splits(data_root, "emodb", 1)
            code, stdout, stderr = self.run_main(
                [
                    "--dataset", "emodb", "--fold", "1",
                    "--data-root", str(data_root), "--output-root", str(output_root), "--dry-run",
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn('"protocol": "earlystop"', stdout)
            self.assertIn('"max_epochs": 80', stdout)
            self.assertIn('"train_subset": "inner_train"', stdout)
            self.assertIn('"valid_subset": "inner_valid"', stdout)
            self.assertIn('"max_updates": 160', stdout)
            self.assertIn("dataset.disable_validation=false", stdout)
            self.assertIn("dataset.valid_subset=inner_valid", stdout)
            self.assertIn("optimization.max_epoch=80", stdout)
            self.assertIn("checkpoint_best.pt", stdout)
            self.assertIn("<best_epoch>", stdout)
            self.assertIn("--selected-epoch", stdout)
            self.assertIn(
                f'"output_dir": "{output_root.resolve()}/emodb/earlystop/{ser_protocol.DEFAULT_CONFIG_NAME}/seed_3949/fold_01"',
                stdout,
            )
            self.assertFalse(output_root.exists())

    def test_fixed_dry_run_uses_full_train_and_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_split(data_root, "emodb", 1, "train", 40)
            self.make_split(data_root, "emodb", 1, "valid", 7)
            self.make_stats(data_root, "emodb", 1, stage="fixed")
            code, stdout, stderr = self.run_main(
                [
                    "--dataset", "emodb", "--fold", "1",
                    "--protocol", "fixed", "--fixed-epochs", "60",
                    "--norm-mode", "fold",
                    "--data-root", str(data_root), "--output-root", str(output_root),
                    "--dry-run",
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn('"protocol": "fixed"', stdout)
            self.assertIn('"fixed_epochs": 60', stdout)
            self.assertIn('"max_epochs": null', stdout)
            self.assertIn('"train_subset": "train"', stdout)
            self.assertIn('"valid_subset": "valid"', stdout)
            self.assertIn("dataset.disable_validation=true", stdout)
            self.assertIn("dataset.valid_subset=train", stdout)
            self.assertIn("optimization.max_epoch=60", stdout)
            self.assertIn("checkpoint_final.pt", stdout)
            self.assertIn("--fixed-epochs 60", stdout)
            self.assertNotIn("--selected-epoch", stdout)
            self.assertIn("norm_stats/emodb/fixed/fold_01.json", stdout)
            self.assertIn(
                f'"output_dir": "{output_root.resolve()}/emodb/fixed/{ser_protocol.DEFAULT_CONFIG_NAME}_foldnorm/seed_3949/fold_01"',
                stdout,
            )
            self.assertFalse(output_root.exists())

    def test_fixed_nondefault_epochs_get_an_isolated_config(self):
        parser = ser_protocol.make_parser()
        args = parser.parse_args(
            ["--dataset", "emodb", "--fold", "1", "--protocol", "fixed",
             "--fixed-epochs", "2"]
        )
        self.assertEqual(
            ser_protocol.resolve_config_name(args),
            f"{ser_protocol.DEFAULT_CONFIG_NAME}_fixedep2",
        )

    def test_fixed_rejects_max_epochs(self):
        code, _, stderr = self.run_main(
            [
                "--dataset", "emodb", "--fold", "1", "--protocol", "fixed",
                "--max-epochs", "1", "--dry-run",
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("--max-epochs applies only", stderr)

    def test_nondefault_max_epochs_get_an_isolated_config(self):
        parser = ser_protocol.make_parser()
        args = parser.parse_args(
            ["--dataset", "emodb", "--fold", "1", "--max-epochs", "1"]
        )
        self.assertEqual(
            ser_protocol.resolve_config_name(args),
            f"{ser_protocol.DEFAULT_CONFIG_NAME}_maxep1",
        )
        args.config_name = "named_smoke"
        args.auto_config_name = False
        self.assertEqual(ser_protocol.resolve_config_name(args), "named_smoke")

    def test_nondefault_batch_size_changes_schedule_and_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            self.make_earlystop_splits(data_root, "emodb", 1)
            code, stdout, stderr = self.run_main(
                [
                    "--dataset", "emodb", "--fold", "1", "--batch-size", "12",
                    "--data-root", str(data_root), "--output-root", str(root / "runs"),
                    "--dry-run",
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn('"batch_size": 12', stdout)
            self.assertIn('"max_updates": 240', stdout)
            self.assertIn("dataset.batch_size=12", stdout)
            self.assertIn(
                f'"config_name": "{ser_protocol.DEFAULT_CONFIG_NAME}_bs12"',
                stdout,
            )

    def test_fold_norm_mode_requires_selection_stats_and_passes_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            self.make_earlystop_splits(data_root, "emodb", 1)
            self.make_stats(data_root, "emodb", 1)
            code, stdout, stderr = self.run_main(
                [
                    "--dataset", "emodb", "--fold", "1", "--max-epochs", "1",
                    "--norm-mode", "fold",
                    "--data-root", str(data_root), "--output-root", str(root / "runs"),
                    "--dry-run",
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn('"norm_mode": "fold"', stdout)
            self.assertIn("task.norm_mode=fold", stdout)
            self.assertIn("task.norm_stats_path=", stdout)
            self.assertIn("norm_stats/emodb/selection/fold_01.json", stdout)
            self.assertIn("_foldnorm_maxep1", stdout)
            self.assertIn("--norm-mode fold", stdout)
            self.assertIn("--norm-stats-path", stdout)

    def test_wandb_project_and_run_name_are_forwarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            self.make_earlystop_splits(data_root, "emodb", 1)
            code, stdout, stderr = self.run_main(
                [
                    "--dataset", "emodb", "--fold", "1",
                    "--data-root", str(data_root), "--output-root", str(root / "runs"),
                    "--wandb-project", "eat2emo-test", "--dry-run",
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn('"wandb_project": "eat2emo-test"', stdout)
            self.assertIn("common.wandb_project=eat2emo-test", stdout)
            self.assertIn("emodb-earlystop-fold01-seed3949-EAT-base_epoch30_pt", stdout)

    def test_fold_ranges_are_dataset_specific(self):
        code, _, stderr = self.run_main(
            ["--dataset", "ravdess-speech", "--fold", "5", "--dry-run"]
        )
        self.assertEqual(code, 2)
        self.assertIn("fold must be 0..4", stderr)

    def test_dry_run_still_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_earlystop_splits(data_root, "emodb", 1)
            output = (
                output_root / "emodb" / "earlystop" / ser_protocol.DEFAULT_CONFIG_NAME
                / "seed_3949" / "fold_01"
            )
            output.mkdir(parents=True)
            code, _, stderr = self.run_main(
                [
                    "--dataset", "emodb", "--fold", "1",
                    "--data-root", str(data_root), "--output-root", str(output_root), "--dry-run",
                ]
            )
            self.assertEqual(code, 2)
            self.assertIn("Refusing to overwrite", stderr)

    def test_validation_metrics_history_records_all_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "train.log"
            log.write_text(
                'inner_valid | {"epoch": 1, "inner_valid_loss": "1.25", '
                '"inner_valid_accuracy": "50.0", "inner_valid_war": "52.0", '
                '"inner_valid_uar": "40.0", "inner_valid_macro_f1": "45.0"}\n'
                'inner_valid | {"epoch": 2, "inner_valid_loss": 0.75, '
                '"inner_valid_accuracy": 75.0, "inner_valid_war": 76.0, '
                '"inner_valid_uar": 70.0, "inner_valid_macro_f1": 72.0}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                ser_protocol.validation_metrics_history(log),
                [
                    {
                        "epoch": 1,
                        "loss": 1.25,
                        "accuracy": 50.0,
                        "war": 52.0,
                        "uar": 40.0,
                        "macro_f1": 45.0,
                    },
                    {
                        "epoch": 2,
                        "loss": 0.75,
                        "accuracy": 75.0,
                        "war": 76.0,
                        "uar": 70.0,
                        "macro_f1": 72.0,
                    },
                ],
            )

    def test_select_best_epoch_uses_earliest_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "checkpoint_best.pt").touch()
            (output / "train.log").write_text(
                'inner_valid | {"epoch": 1, "inner_valid_uar": "80.0"}\n'
                'inner_valid | {"epoch": 2, "inner_valid_uar": "80.0"}\n'
                'inner_valid | {"epoch": 3, "inner_valid_uar": "70.0"}\n',
                encoding="utf-8",
            )
            best_epoch, history, best_row = ser_protocol.select_best_epoch(output, 3)
            self.assertEqual(best_epoch, 1)
            self.assertEqual(len(history), 3)
            self.assertEqual(best_row["uar"], 80.0)

    def test_select_best_epoch_rejects_missing_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "checkpoint_best.pt").touch()
            (output / "train.log").write_text(
                'inner_valid | {"epoch": 1, "inner_valid_uar": "80.0"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Expected inner-valid metrics"):
                ser_protocol.select_best_epoch(output, 2)

    def test_execute_run_trains_then_evaluates_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output_root = root / "runs"
            fairseq_root = root / "fairseq"
            pretrained = root / "eat_pretrained.pt"
            (fairseq_root / "fairseq_cli").mkdir(parents=True)
            (fairseq_root / "fairseq_cli" / "hydra_train.py").touch()
            pretrained.write_bytes(b"pretrained")
            self.make_earlystop_splits(data_root, "emodb", 1)

            commands = []

            def fake_run(command, log_path, **kwargs):
                commands.append(list(command))
                if str(command[1]).endswith("hydra_train.py"):
                    overrides = {
                        token.split("=", 1)[0]: token.split("=", 1)[1]
                        for token in command
                        if "=" in token and not token.startswith("--")
                    }
                    save_dir = Path(overrides["checkpoint.save_dir"])
                    (save_dir / "checkpoint_best.pt").write_bytes(b"best")
                    log_path.write_text(
                        'inner_valid | {"epoch": 1, "inner_valid_loss": "1.2", '
                        '"inner_valid_accuracy": "50", "inner_valid_war": "50", '
                        '"inner_valid_uar": "40", "inner_valid_macro_f1": "45"}\n'
                        'inner_valid | {"epoch": 2, "inner_valid_loss": "0.9", '
                        '"inner_valid_accuracy": "75", "inner_valid_war": "75", '
                        '"inner_valid_uar": "70", "inner_valid_macro_f1": "72"}\n',
                        encoding="utf-8",
                    )
                else:
                    log_path.write_text("eval complete\n", encoding="utf-8")

            with mock.patch.object(ser_protocol, "run_and_tee", side_effect=fake_run):
                code, _, stderr = self.run_main(
                    [
                        "--dataset", "emodb", "--fold", "1", "--seed", "1",
                        "--max-epochs", "2",
                        "--data-root", str(data_root), "--output-root", str(output_root),
                        "--fairseq-root", str(fairseq_root), "--pretrained", str(pretrained),
                    ]
                )

            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(len(commands), 2)
            train_command, eval_command = commands
            self.assertIn("model.model_path=" + str(pretrained.resolve()), train_command)
            self.assertIn("dataset.train_subset=inner_train", train_command)
            self.assertIn("dataset.valid_subset=inner_valid", train_command)
            self.assertIn("dataset.disable_validation=false", train_command)
            self.assertIn("optimization.max_epoch=2", train_command)
            self.assertIn("optimization.max_update=4", train_command)
            self.assertIn("checkpoint_best.pt", " ".join(eval_command))
            self.assertIn("--selected-epoch", eval_command)
            self.assertIn("2", eval_command)
            self.assertIn("--max-epochs", eval_command)

            run_dir = (
                output_root / "emodb" / "earlystop"
                / f"{ser_protocol.DEFAULT_CONFIG_NAME}_maxep2"
                / "seed_1" / "fold_01"
            )
            summary = json.loads(
                (run_dir / "earlystop_summary.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["best_epoch"], 2)
            self.assertEqual(summary["best_uar"], 70.0)
            self.assertEqual(len(summary["history"]), 2)
            self.assertEqual(manifest["status"], "evaluated")
            self.assertEqual(manifest["best_epoch"], 2)

    def test_execute_fixed_trains_then_evaluates_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output_root = root / "runs"
            fairseq_root = root / "fairseq"
            pretrained = root / "eat_pretrained.pt"
            (fairseq_root / "fairseq_cli").mkdir(parents=True)
            (fairseq_root / "fairseq_cli" / "hydra_train.py").touch()
            pretrained.write_bytes(b"pretrained")
            self.make_split(data_root, "emodb", 1, "train", 40)
            self.make_split(data_root, "emodb", 1, "valid", 7)
            self.make_stats(data_root, "emodb", 1, stage="fixed")

            commands = []

            def fake_run(command, log_path, **kwargs):
                commands.append(list(command))
                if str(command[1]).endswith("hydra_train.py"):
                    overrides = {
                        token.split("=", 1)[0]: token.split("=", 1)[1]
                        for token in command
                        if "=" in token and not token.startswith("--")
                    }
                    save_dir = Path(overrides["checkpoint.save_dir"])
                    (save_dir / "checkpoint_last.pt").write_bytes(b"last")
                else:
                    log_path.write_text("eval complete\n", encoding="utf-8")

            with mock.patch.object(ser_protocol, "run_and_tee", side_effect=fake_run):
                code, _, stderr = self.run_main(
                    [
                        "--dataset", "emodb", "--fold", "1", "--seed", "1",
                        "--protocol", "fixed", "--fixed-epochs", "2",
                        "--norm-mode", "fold",
                        "--data-root", str(data_root), "--output-root", str(output_root),
                        "--fairseq-root", str(fairseq_root), "--pretrained", str(pretrained),
                    ]
                )

            self.assertEqual((code, stderr), (0, ""))
            self.assertEqual(len(commands), 2)
            train_command, eval_command = commands
            self.assertIn("dataset.train_subset=train", train_command)
            self.assertIn("dataset.valid_subset=train", train_command)
            self.assertIn("dataset.disable_validation=true", train_command)
            self.assertIn("optimization.max_epoch=2", train_command)
            self.assertIn("checkpoint_final.pt", " ".join(eval_command))
            self.assertIn("--fixed-epochs", eval_command)
            self.assertIn("2", eval_command)
            self.assertNotIn("--selected-epoch", eval_command)

            run_dir = (
                output_root / "emodb" / "fixed"
                / f"{ser_protocol.DEFAULT_CONFIG_NAME}_foldnorm_fixedep2"
                / "seed_1" / "fold_01"
            )
            self.assertEqual((run_dir / "checkpoint_final.pt").read_bytes(), b"last")
            manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "evaluated")
            self.assertIsNone(manifest["best_epoch"])
            self.assertFalse((run_dir / "earlystop_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
