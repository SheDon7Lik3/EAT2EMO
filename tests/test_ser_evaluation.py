import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.eval_ser import (
    build_parser,
    compute_ser_metrics,
    read_manifest,
    save_evaluation_artifacts,
)
from evaluation.summarize_ser import (
    DEFAULT_SEEDS,
    _load_fold_result,
    aggregate_results,
    discover_fold_results,
)


LABELS = {
    "emodb": tuple(f"class_{index}" for index in range(7)),
    "iemocap": tuple(f"class_{index}" for index in range(4)),
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def probabilities_for(targets, num_classes, confidence=0.9):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.full(
        (len(targets), num_classes),
        (1.0 - confidence) / (num_classes - 1),
        dtype=np.float64,
    )
    probabilities[np.arange(len(targets)), targets] = confidence
    return probabilities


def save_fold(
    root,
    *,
    dataset,
    fold,
    seed,
    targets,
    paths,
    probabilities=None,
    config="test_config",
    protocol="earlystop",
    selected_epoch=3,
    max_epochs=5,
    fixed_epochs=None,
    norm_mode="audioset",
    norm_mean=-4.268,
    norm_std=4.569,
    norm_stats_path=None,
    norm_stats_sha256="",
):
    labels = LABELS[dataset]
    if probabilities is None:
        probabilities = probabilities_for(targets, len(labels))
    if protocol == "fixed":
        selected_epoch = None
        max_epochs = None
        if fixed_epochs is None:
            fixed_epochs = 60
    output = (
        root / dataset / protocol / config / f"seed_{seed}"
        / f"fold_{fold:02d}" / "results"
    )
    metadata = {
        "dataset": dataset,
        "protocol": protocol,
        "config": config,
        "fold": fold,
        "seed": seed,
        "split": "valid",
        "selected_epoch": selected_epoch,
        "max_epochs": max_epochs,
        "fixed_epochs": fixed_epochs,
        "trained_epochs": selected_epoch if protocol == "earlystop" else fixed_epochs,
        "target_length": 640 if dataset == "emodb" else 1024,
        "norm_mode": norm_mode,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_stats_path": norm_stats_path,
        "norm_stats_sha256": norm_stats_sha256,
    }
    save_evaluation_artifacts(
        output,
        probabilities,
        np.asarray(targets),
        paths,
        labels,
        metadata,
    )
    return output


class SerMetricTest(unittest.TestCase):
    def test_valid_is_default_and_manifest_order_is_not_resorted(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--data", "/tmp/data", "--checkpoint", "/tmp/model.pt",
                "--model-dir", "/tmp/model", "--output", "/tmp/output",
                "--dataset", "iemocap", "--protocol", "earlystop",
                "--fold", "1", "--seed", "1",
                "--selected-epoch", "3", "--max-epochs", "5",
            ]
        )
        self.assertEqual(args.split, "valid")
        self.assertEqual(args.selected_epoch, 3)
        self.assertEqual(args.max_epochs, 5)
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "valid.tsv").write_text(
                "/server/audio\nSession1/z.wav\t10\nSession1/a.wav\t20\n"
            )
            (data / "valid.lbl").write_text(
                "Session1/z.wav\thappy\nSession1/a.wav\tangry\n"
            )
            (data / "label_descriptors.csv").write_text("0,angry\n1,happy\n")
            manifest = read_manifest(data, verify_files=False)
            self.assertEqual(
                manifest.identities,
                ("Session1/z.wav", "Session1/a.wav"),
            )
            np.testing.assert_array_equal(manifest.targets, [1, 0])

    def test_missing_classes_are_excluded_from_single_fold_macro_average(self):
        targets = np.asarray([0, 0, 1])
        probabilities = np.asarray(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.8, 0.1],
            ]
        )
        result, confusion = compute_ser_metrics(probabilities, targets)
        self.assertEqual(result["present_classes"], [0, 1])
        self.assertEqual(result["metric_average_classes"], [0, 1])
        self.assertAlmostEqual(result["uar"], 0.75)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 1.0) / 2)
        self.assertEqual(confusion.shape, (3, 3))

        pooled, _ = compute_ser_metrics(
            probabilities, targets, average_over_present_classes=False
        )
        self.assertEqual(pooled["metric_average_classes"], [0, 1, 2])
        self.assertAlmostEqual(pooled["uar"], 0.5)

    def test_manifest_rejects_reordered_label_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "valid.tsv").write_text(
                "/server/audio\nSession1/a.wav\t10\nSession1/b.wav\t20\n"
            )
            (data / "valid.lbl").write_text(
                "Session1/b.wav\tangry\nSession1/a.wav\thappy\n"
            )
            (data / "label_descriptors.csv").write_text("0,angry\n1,happy\n")
            with self.assertRaisesRegex(ValueError, "TSV/LBL identity mismatch"):
                read_manifest(data, verify_files=False)

    def test_artifacts_preserve_order_and_refuse_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            targets = np.asarray([1, 0])
            probabilities = probabilities_for(targets, 4)
            paths = ["Session1/second.wav", "Session1/first.wav"]
            metadata = {
                "dataset": "iemocap",
                "protocol": "earlystop",
                "config": "test_config",
                "fold": 1,
                "seed": 2,
                "split": "valid",
                "selected_epoch": 3,
                "max_epochs": 5,
                "trained_epochs": 3,
            }
            payload = save_evaluation_artifacts(
                output,
                probabilities,
                targets,
                paths,
                LABELS["iemocap"],
                metadata,
            )
            self.assertEqual(payload["accuracy"], 1.0)
            self.assertEqual(payload["war"], 1.0)
            self.assertEqual(payload["selected_epoch"], 3)
            self.assertEqual((output / "paths.txt").read_text().splitlines(), paths)
            with np.load(output / "predictions.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["targets"], targets)
                np.testing.assert_allclose(archive["probabilities"], probabilities)
            with self.assertRaises(FileExistsError):
                save_evaluation_artifacts(
                    output,
                    probabilities,
                    targets,
                    paths,
                    LABELS["iemocap"],
                    metadata,
                )

    def test_fold_norm_metrics_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_fold(
                root,
                dataset="emodb",
                fold=1,
                seed=1,
                targets=[0, 1],
                paths=["a.wav", "b.wav"],
                norm_mode="fold",
                norm_mean=-5.0,
                norm_std=4.0,
                norm_stats_path="/tmp/norm_stats.json",
                norm_stats_sha256="a" * 64,
            )
            metrics_path = (
                root / "emodb" / "earlystop" / "test_config" / "seed_1"
                / "fold_01" / "results" / "metrics.json"
            )
            result = _load_fold_result(metrics_path)
            self.assertEqual(result.payload["norm_mode"], "fold")
            self.assertAlmostEqual(result.payload["norm_mean"], -5.0)

    def test_selected_epoch_greater_than_max_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = save_fold(
                root,
                dataset="emodb",
                fold=1,
                seed=1,
                targets=[0, 1],
                paths=["a.wav", "b.wav"],
                selected_epoch=6,
                max_epochs=5,
            )
            with self.assertRaisesRegex(ValueError, "selected_epoch exceeds max_epochs"):
                _load_fold_result(output / "metrics.json")

    def test_fixed_metrics_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = save_fold(
                root,
                dataset="emodb",
                fold=1,
                seed=1,
                targets=[0, 1],
                paths=["a.wav", "b.wav"],
                protocol="fixed",
                fixed_epochs=60,
            )
            result = _load_fold_result(output / "metrics.json")
            self.assertEqual(result.payload["protocol"], "fixed")
            self.assertEqual(result.payload["fixed_epochs"], 60)
            self.assertIsNone(result.payload["selected_epoch"])

    def test_wrong_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = save_fold(
                root,
                dataset="emodb",
                fold=1,
                seed=1,
                targets=[0, 1],
                paths=["a.wav", "b.wav"],
            )
            metrics_path = output / "metrics.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["protocol"] = "nested"
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid protocol"):
                _load_fold_result(metrics_path)


class SerAggregationTest(unittest.TestCase):
    def test_default_seeds_match_reference_protocol(self):
        self.assertEqual(DEFAULT_SEEDS, (3949, 42, 2026, 1024, 666))

    def test_non_emodb_uses_fold_mean_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fold_targets = {1: [0, 1], 2: [2, 3]}
            for seed in (1, 2):
                for fold, targets in fold_targets.items():
                    probabilities = probabilities_for(targets, 4)
                    if seed == 2 and fold == 2:
                        probabilities = probabilities[[1, 0]]
                    save_fold(
                        root,
                        dataset="iemocap",
                        fold=fold,
                        seed=seed,
                        targets=targets,
                        probabilities=probabilities,
                        paths=[f"fold{fold}/sample{index}.wav" for index in range(2)],
                    )
            results = discover_fold_results(root, "iemocap", "earlystop")
            self.assertEqual(len(results), 4)
            summary = aggregate_results(
                results,
                dataset="iemocap",
                protocol="earlystop",
                expected_folds=(1, 2),
                expected_seeds=(1, 2),
                expected_samples=4,
            )
            self.assertEqual(summary["protocol"], "earlystop")
            self.assertEqual(summary["max_epochs"], 5)
            self.assertEqual(summary["primary"]["method"], "mean_across_folds_per_seed")
            self.assertEqual(summary["primary"]["per_seed"]["1"]["accuracy"], 1.0)
            self.assertEqual(summary["primary"]["per_seed"]["2"]["accuracy"], 0.5)
            self.assertAlmostEqual(
                summary["primary"]["across_seeds"]["accuracy"]["mean"], 0.75
            )
            self.assertEqual(len(summary["fold_results"]), 4)
            self.assertEqual(summary["fold_results"][0]["selected_epoch"], 3)

    def test_missing_grid_entry_and_duplicate_outer_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = save_fold(
                root,
                dataset="iemocap",
                fold=1,
                seed=1,
                targets=[0, 1],
                paths=["shared.wav", "one.wav"],
            )
            results = discover_fold_results(root, "iemocap", "earlystop")
            with self.assertRaisesRegex(ValueError, "missing seed/fold"):
                aggregate_results(
                    results,
                    dataset="iemocap",
                    protocol="earlystop",
                    expected_folds=(1, 2),
                    expected_seeds=(1,),
                    expected_samples=4,
                )

            save_fold(
                root,
                dataset="iemocap",
                fold=2,
                seed=1,
                targets=[2, 3],
                paths=["shared.wav", "two.wav"],
            )
            results = discover_fold_results(root, "iemocap", "earlystop")
            with self.assertRaisesRegex(ValueError, "unique paths"):
                aggregate_results(
                    results,
                    dataset="iemocap",
                    protocol="earlystop",
                    expected_folds=(1, 2),
                    expected_seeds=(1,),
                    expected_samples=4,
                )
            self.assertTrue((first / "metrics.json").is_file())

    def test_fixed_results_aggregate_with_fixed_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold, targets in ((1, [0, 1]), (2, [2, 3])):
                save_fold(
                    root,
                    dataset="iemocap",
                    fold=fold,
                    seed=1,
                    targets=targets,
                    paths=[f"fold{fold}/{index}.wav" for index in range(2)],
                    protocol="fixed",
                    fixed_epochs=60,
                )
            results = discover_fold_results(root, "iemocap", "fixed")
            summary = aggregate_results(
                results,
                dataset="iemocap",
                protocol="fixed",
                expected_folds=(1, 2),
                expected_seeds=(1,),
                expected_samples=4,
                fixed_epochs=60,
            )
            self.assertEqual(summary["fixed_epochs"], 60)
            self.assertIsNone(summary["max_epochs"])
            self.assertEqual(summary["fold_results"][0]["fixed_epochs"], 60)
            self.assertIsNone(summary["fold_results"][0]["selected_epoch"])

    def test_inconsistent_max_epochs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_fold(
                root,
                dataset="iemocap",
                fold=1,
                seed=1,
                targets=[0, 1],
                paths=["a.wav", "b.wav"],
                max_epochs=5,
            )
            save_fold(
                root,
                dataset="iemocap",
                fold=2,
                seed=1,
                targets=[2, 3],
                paths=["c.wav", "d.wav"],
                max_epochs=6,
            )
            results = discover_fold_results(root, "iemocap", "earlystop")
            with self.assertRaisesRegex(ValueError, "different max_epochs"):
                aggregate_results(
                    results,
                    dataset="iemocap",
                    protocol="earlystop",
                    expected_folds=(1, 2),
                    expected_seeds=(1,),
                    expected_samples=4,
                )

    def test_emodb_primary_is_complete_535_sample_oof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(1, 6):
                offset = 0
                for fold in range(1, 11):
                    count = 58 if fold == 10 else 53
                    targets = np.arange(offset, offset + count) % 7
                    save_fold(
                        root,
                        dataset="emodb",
                        fold=fold,
                        seed=seed,
                        targets=targets,
                        paths=[f"speaker_fold_{fold}/{index}.wav" for index in range(count)],
                    )
                    offset += count
                self.assertEqual(offset, 535)
            results = discover_fold_results(root, "emodb", "earlystop")
            summary = aggregate_results(
                results,
                dataset="emodb",
                protocol="earlystop",
                expected_folds=tuple(range(1, 11)),
                expected_seeds=(1, 2, 3, 4, 5),
                expected_samples=535,
            )
            primary = summary["primary"]
            self.assertEqual(primary["method"], "pooled_10_fold_oof")
            self.assertEqual(primary["per_seed"]["1"]["samples"], 535)
            self.assertEqual(primary["per_seed"]["1"]["folds"], 10)
            self.assertEqual(primary["per_seed"]["1"]["accuracy"], 1.0)
            self.assertEqual(primary["per_seed"]["1"]["metric_average_classes"], list(range(7)))
            self.assertEqual(primary["across_seeds"]["accuracy"]["count"], 5)
            self.assertEqual(primary["across_seeds"]["accuracy"]["std"], 0.0)


class SerBatchScriptTest(unittest.TestCase):
    @staticmethod
    def make_protocol_manifests(data_root, dataset, folds):
        for fold in folds:
            directory = data_root / "manifests" / dataset / f"fold_{fold:02d}"
            directory.mkdir(parents=True)
            for split in ("train", "valid", "inner_train", "inner_valid"):
                (directory / f"{split}.tsv").write_text(
                    "/root/songhao/datasets/SER_DATASET\n"
                    f"{dataset}/fold{fold}/{split}.wav\t16000\n"
                )
                (directory / f"{split}.lbl").write_text(
                    f"{split}.wav\tneutral\n"
                )

    def test_batch_dry_run_prints_automatic_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_protocol_manifests(data_root, "ravdess-song", range(5))
            result = subprocess.run(
                [
                    "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                    "--datasets", "ravdess-song", "--seeds", "7",
                    "--max-epochs", "3",
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('"protocol": "earlystop"'), 5)
            self.assertIn("/fold_00", result.stdout)
            self.assertIn("/fold_04", result.stdout)
            self.assertIn("SUMMARY:", result.stdout)
            self.assertIn("--expected-seeds 7", result.stdout)
            self.assertIn("--max-epochs 3", result.stdout)
            config = "eat_base_sr16k_audiosetnorm_bs32_maxep3"
            self.assertIn(f'"config_name": "{config}"', result.stdout)
            self.assertIn(
                str(output_root / f"ravdess-song/earlystop/{config}/summary.json"),
                result.stdout,
            )
            self.assertFalse(output_root.exists())

    def test_batch_fixed_dry_run_uses_full_train_and_fixed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_protocol_manifests(data_root, "ravdess-song", range(5))
            result = subprocess.run(
                [
                    "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                    "--datasets", "ravdess-song", "--seeds", "7",
                    "--protocol", "fixed", "--fixed-epochs", "3",
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('"protocol": "fixed"'), 5)
            self.assertIn("dataset.disable_validation=true", result.stdout)
            self.assertIn("dataset.valid_subset=train", result.stdout)
            self.assertIn("--fixed-epochs 3", result.stdout)
            config = "eat_base_sr16k_audiosetnorm_bs32_fixedep3"
            self.assertIn(f'"config_name": "{config}"', result.stdout)
            self.assertIn(
                str(output_root / f"ravdess-song/fixed/{config}/summary.json"),
                result.stdout,
            )
            self.assertIn("--protocol fixed", result.stdout.replace("\\", ""))
            self.assertIn("--fixed-epochs 3", result.stdout.replace("\\", ""))
            self.assertFalse(output_root.exists())

    def test_batch_rejects_max_epochs_for_fixed(self):
        result = subprocess.run(
            [
                "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                "--protocol", "fixed", "--max-epochs", "3", "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--max-epochs applies only", result.stderr)

    def test_batch_default_seeds_match_reference_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_protocol_manifests(data_root, "ravdess-song", range(5))
            result = subprocess.run(
                [
                    "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                    "--datasets", "ravdess-song",
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for seed in (3949, 42, 2026, 1024, 666):
                self.assertIn(f'"seed": {seed}', result.stdout)
            self.assertIn(
                "--expected-seeds 3949,42,2026,1024,666",
                result.stdout.replace("\\,", ","),
            )
            self.assertFalse(output_root.exists())

    def test_batch_can_select_single_fold_with_all_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_protocol_manifests(data_root, "ravdess-song", range(5))
            result = subprocess.run(
                [
                    "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                    "--datasets", "ravdess-song", "--folds", "2",
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('"fold": 2'), 5)
            self.assertNotIn('"fold": 0', result.stdout)
            self.assertIn("Partial folds selected", result.stderr)
            self.assertNotIn("SUMMARY:", result.stdout)
            self.assertFalse(output_root.exists())

    def test_batch_can_select_single_seed_and_multiple_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_protocol_manifests(data_root, "ravdess-song", range(5))
            result = subprocess.run(
                [
                    "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                    "--datasets", "ravdess-song", "--seeds", "3949",
                    "--folds", "0,2",
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('"seed": 3949'), 2)
            self.assertIn('"fold": 0', result.stdout)
            self.assertIn('"fold": 2', result.stdout)
            self.assertNotIn('"seed": 42', result.stdout)
            self.assertIn("Partial folds selected", result.stderr)
            self.assertNotIn("SUMMARY:", result.stdout)

    def test_batch_single_seed_all_folds_still_summarizes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_protocol_manifests(data_root, "ravdess-song", range(5))
            result = subprocess.run(
                [
                    "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                    "--datasets", "ravdess-song", "--seeds", "3949", "--folds", "all",
                    "--data-root", str(data_root),
                    "--output-root", str(output_root),
                    "--dry-run",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count('"seed": 3949'), 5)
            self.assertIn("SUMMARY:", result.stdout)
            self.assertIn("--expected-seeds 3949", result.stdout.replace("\\,", ","))

    def test_batch_rejects_fold_invalid_for_dataset(self):
        result = subprocess.run(
            [
                "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                "--datasets", "emodb", "--folds", "0", "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Fold 0 is invalid for emodb", result.stderr)

    def test_batch_rejects_duplicate_seeds_before_launch(self):
        result = subprocess.run(
            [
                "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                "--datasets", "emodb", "--seeds", "1,1", "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Duplicate seed: 1", result.stderr)

    def test_batch_rejects_invalid_max_epochs(self):
        result = subprocess.run(
            [
                "bash", str(PROJECT_ROOT / "scripts/run_all_SER.sh"),
                "--datasets", "emodb", "--max-epochs", "0", "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--max-epochs must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
