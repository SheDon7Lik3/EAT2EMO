import contextlib
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

    def run_main(self, arguments):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = ser_protocol.main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_frozen_dataset_defaults(self):
        expected = {
            "emodb": (7, 512, 1, 10, 80),
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
                spec.selection_epochs,
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

    def test_fixed_dry_run_defaults_to_sixty_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_split(data_root, "emodb", 1, "train", 33)
            self.make_split(data_root, "emodb", 1, "valid", 4)
            code, stdout, stderr = self.run_main(
                [
                    "--dataset", "emodb", "--fold", "1", "--protocol", "fixed",
                    "--data-root", str(data_root), "--output-root", str(output_root), "--dry-run",
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn('"stage": "fixed"', stdout)
            self.assertIn('"epochs": 60', stdout)
            self.assertIn('"max_updates": 120', stdout)
            self.assertIn("dataset.disable_validation=true", stdout)
            self.assertIn("dataset.valid_subset=train", stdout)
            self.assertIn("--fixed-epochs 60", stdout)
            self.assertIn("--norm-mean -4.268 --norm-std 4.569", stdout)
            self.assertFalse(output_root.exists())

    def test_nested_selection_uses_inner_splits_and_smoke_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            for split, count in (("inner_train", 65), ("inner_valid", 5)):
                self.make_split(data_root, "ravdess-song", 0, split, count)
            code, stdout, stderr = self.run_main(
                [
                    "--dataset", "ravdess-song", "--fold", "0", "--stage", "select",
                    "--selection-max-epochs", "1", "--data-root", str(data_root),
                    "--output-root", str(root / "runs"), "--dry-run",
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn('"train_subset": "inner_train"', stdout)
            self.assertIn('"target_length": 640', stdout)
            self.assertIn('"max_updates": 3', stdout)
            self.assertIn("dataset.disable_validation=false", stdout)
            self.assertNotIn("EVAL:", stdout)

    def test_one_epoch_nested_all_dry_run_prints_both_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            for split in ("inner_train", "inner_valid", "train", "valid"):
                self.make_split(data_root, "emodb", 1, split, 2)
            code, stdout, stderr = self.run_main(
                [
                    "--dataset", "emodb", "--fold", "1", "--selection-max-epochs", "1",
                    "--data-root", str(data_root), "--output-root", str(root / "runs"), "--dry-run",
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn('"stage": "selection"', stdout)
            self.assertIn('"stage": "final"', stdout)
            self.assertIn("--selected-epoch 1", stdout)
            self.assertIn(
                f'"config_name": "{ser_protocol.DEFAULT_CONFIG_NAME}_selectmaxep1"',
                stdout,
            )

    def test_nondefault_fixed_epochs_get_an_isolated_default_config(self):
        parser = ser_protocol.make_parser()
        args = parser.parse_args(
            ["--dataset", "emodb", "--fold", "1", "--protocol", "fixed", "--fixed-epochs", "1"]
        )
        self.assertEqual(
            ser_protocol.resolve_config_name(args),
            f"{ser_protocol.DEFAULT_CONFIG_NAME}_fixedep1",
        )
        args.config_name = "named_smoke"
        args.auto_config_name = False
        self.assertEqual(ser_protocol.resolve_config_name(args), "named_smoke")

    def test_nested_final_reads_and_forwards_selected_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output_root = root / "runs"
            for split, count in (("train", 40), ("valid", 7)):
                self.make_split(data_root, "iemocap", 1, split, count)
            selection = (
                output_root / "iemocap" / "nested"
                / f"{ser_protocol.DEFAULT_CONFIG_NAME}_selectmaxep1"
                / "seed_1" / "fold_01" / "selection"
            )
            selection.mkdir(parents=True)
            pretrained = root / "eat_pretrained.pt"
            (selection / "selection_summary.json").write_text(
                json.dumps({
                    "dataset": "iemocap", "protocol": "nested",
                    "config_name": f"{ser_protocol.DEFAULT_CONFIG_NAME}_selectmaxep1",
                    "pretrained": str(pretrained.resolve()),
                    "fold": 1, "seed": 1,
                    "max_epochs": 1, "best_epoch": 1,
                }),
                encoding="utf-8",
            )
            code, stdout, stderr = self.run_main(
                [
                    "--dataset", "iemocap", "--fold", "1", "--stage", "final",
                    "--selection-max-epochs", "1", "--data-root", str(data_root),
                    "--output-root", str(output_root), "--pretrained", str(pretrained),
                    "--dry-run",
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            self.assertIn('"selected_epoch": 1', stdout)
            self.assertIn("dataset.disable_validation=true", stdout)
            self.assertIn("--selected-epoch 1", stdout)
            self.assertIn(
                f"--config {ser_protocol.DEFAULT_CONFIG_NAME}_selectmaxep1",
                stdout,
            )
            self.assertIn("/seed_1/fold_01/final", stdout)

    def test_nested_final_rejects_selection_from_other_pretrained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = ser_protocol.make_parser().parse_args(
                [
                    "--dataset", "emodb", "--fold", "1", "--stage", "final",
                    "--selection-max-epochs", "1",
                    "--pretrained", str(root / "expected.pt"),
                ]
            )
            args.config_name = ser_protocol.resolve_config_name(args)
            args.pretrained = args.pretrained.resolve()
            summary = root / "selection_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "dataset": "emodb",
                        "protocol": "nested",
                        "config_name": args.config_name,
                        "pretrained": str((root / "different.pt").resolve()),
                        "fold": 1,
                        "seed": 1,
                        "max_epochs": 1,
                        "best_epoch": 1,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Selection summary pretrained"):
                ser_protocol.read_selection_summary(summary, args)

    def test_fixed_rejects_nested_stage(self):
        code, _, stderr = self.run_main(
            ["--dataset", "emodb", "--fold", "1", "--protocol", "fixed", "--stage", "final", "--dry-run"]
        )
        self.assertEqual(code, 2)
        self.assertIn("--stage applies only", stderr)

    def test_fold_ranges_are_dataset_specific(self):
        code, _, stderr = self.run_main(
            ["--dataset", "ravdess-speech", "--fold", "5", "--stage", "select", "--dry-run"]
        )
        self.assertEqual(code, 2)
        self.assertIn("fold must be 0..4", stderr)

    def test_dry_run_still_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_split(data_root, "emodb", 1, "train", 2)
            self.make_split(data_root, "emodb", 1, "valid", 1)
            fixed = (
                output_root / "emodb" / "fixed" / ser_protocol.DEFAULT_CONFIG_NAME
                / "seed_1" / "fold_01" / "fixed"
            )
            fixed.mkdir(parents=True)
            code, _, stderr = self.run_main(
                [
                    "--dataset", "emodb", "--fold", "1", "--protocol", "fixed",
                    "--data-root", str(data_root), "--output-root", str(output_root), "--dry-run",
                ]
            )
            self.assertEqual(code, 2)
            self.assertIn("Refusing to overwrite", stderr)

    def test_validation_history_preserves_epoch_curve(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "train.log"
            log.write_text(
                'prefix {"epoch": 1, "valid_uar": "0.5"}\n'
                'prefix {"epoch": 2, "valid_uar": 0.75}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                ser_protocol.validation_history(log),
                [{"epoch": 1, "uar": 0.5}, {"epoch": 2, "uar": 0.75}],
            )

    def test_validation_history_accepts_fairseq_valid_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "checkpoint_best.pt").touch()
            (output / "train.log").write_text(
                'inner_valid | {"epoch": 1, "inner_valid_uar": "0.8"}\n'
                'inner_valid | {"epoch": 2, "inner_valid_uar": "0.8"}\n'
                'inner_valid | {"epoch": 3, "inner_valid_uar": "0.7"}\n',
                encoding="utf-8",
            )
            epoch, history = ser_protocol.select_best_epoch(output, 3)
            self.assertEqual(epoch, 1)
            self.assertEqual(len(history), 3)

    def test_final_checkpoint_is_promoted_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            source = output / "checkpoint_last.pt"
            source.write_bytes(b"checkpoint")
            target = ser_protocol.promote_final_checkpoint(output)
            self.assertEqual(target.read_bytes(), b"checkpoint")
            with self.assertRaises(FileExistsError):
                ser_protocol.promote_final_checkpoint(output)

    def test_select_then_final_restarts_from_pretrained_and_evaluates_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            output_root = root / "runs"
            fairseq_root = root / "fairseq"
            pretrained = root / "eat_pretrained.pt"
            (fairseq_root / "fairseq_cli").mkdir(parents=True)
            (fairseq_root / "fairseq_cli" / "hydra_train.py").touch()
            pretrained.write_bytes(b"pretrained")
            for split in ("inner_train", "inner_valid", "train", "valid"):
                self.make_split(data_root, "emodb", 1, split, 33)

            commands = []

            def fake_run(command, log_path):
                commands.append(list(command))
                if str(command[1]).endswith("hydra_train.py"):
                    overrides = {
                        token.split("=", 1)[0]: token.split("=", 1)[1]
                        for token in command
                        if "=" in token and not token.startswith("--")
                    }
                    save_dir = Path(overrides["checkpoint.save_dir"])
                    (save_dir / "checkpoint_last.pt").write_bytes(b"trained")
                    if overrides["dataset.train_subset"] == "inner_train":
                        (save_dir / "checkpoint_best.pt").write_bytes(b"best")
                        log_path.write_text(
                            'inner_valid | {"epoch": 1, "inner_valid_uar": "72.5"}\n',
                            encoding="utf-8",
                        )
                    else:
                        log_path.write_text("train complete\n", encoding="utf-8")
                else:
                    log_path.write_text("eval complete\n", encoding="utf-8")

            common = [
                "--dataset", "emodb", "--fold", "1", "--selection-max-epochs", "1",
                "--data-root", str(data_root), "--output-root", str(output_root),
                "--fairseq-root", str(fairseq_root), "--pretrained", str(pretrained),
            ]
            with mock.patch.object(ser_protocol, "run_and_tee", side_effect=fake_run):
                select_code, _, select_error = self.run_main([*common, "--stage", "select"])
                final_code, _, final_error = self.run_main([*common, "--stage", "final"])

            self.assertEqual((select_code, select_error), (0, ""))
            self.assertEqual((final_code, final_error), (0, ""))
            self.assertEqual(len(commands), 3)  # selection train, final train, outer-valid eval
            selection_train, final_train, evaluation = commands
            self.assertIn(f"model.model_path={pretrained.resolve()}", selection_train)
            self.assertIn(f"model.model_path={pretrained.resolve()}", final_train)
            self.assertIn("dataset.disable_validation=false", selection_train)
            self.assertIn("dataset.valid_subset=inner_valid", selection_train)
            self.assertIn("dataset.disable_validation=true", final_train)
            self.assertIn("dataset.valid_subset=train", final_train)
            self.assertIn("optimization.max_epoch=1", final_train)
            self.assertIn("optimization.max_update=2", final_train)
            self.assertIn("optimizer.groups.default.lr_scheduler.warmup_updates=0", final_train)
            self.assertFalse(any("checkpoint_best.pt" in token for token in final_train))
            self.assertIn("--selected-epoch", evaluation)
            final_dir = (
                output_root / "emodb" / "nested"
                / f"{ser_protocol.DEFAULT_CONFIG_NAME}_selectmaxep1"
                / "seed_1" / "fold_01" / "final"
            )
            self.assertEqual((final_dir / "checkpoint_final.pt").read_bytes(), b"trained")


if __name__ == "__main__":
    unittest.main()
