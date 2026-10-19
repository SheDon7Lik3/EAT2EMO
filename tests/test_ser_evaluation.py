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
    protocol,
    fold,
    seed,
    targets,
    paths,
    probabilities=None,
    config="test_config",
    fixed_epochs=60,
):
    labels = LABELS[dataset]
    if probabilities is None:
        probabilities = probabilities_for(targets, len(labels))
    stage = "fixed" if protocol == "fixed" else "final"
    output = root / dataset / protocol / config / f"seed_{seed}" / f"fold_{fold:02d}" / stage / "results"
    metadata = {
        "dataset": dataset,
        "protocol": protocol,
        "config": config,
        "fold": fold,
        "seed": seed,
        "stage": stage,
        "split": "valid",
        "fixed_epochs": fixed_epochs if protocol == "fixed" else None,
        "selected_epoch": 3 if protocol == "nested" else None,
        "trained_epochs": fixed_epochs if protocol == "fixed" else 3,
        "target_length": 512 if dataset == "emodb" else 1024,
        "norm_mean": -4.268,
        "norm_std": 4.569,
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
                "--dataset", "iemocap", "--protocol", "nested",
                "--fold", "1", "--seed", "1", "--selected-epoch", "3",
            ]
        )
        self.assertEqual(args.split, "valid")
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
                "protocol": "fixed",
                "config": "test_config",
                "fold": 1,
                "seed": 2,
                "stage": "fixed",
                "split": "valid",
                "fixed_epochs": 60,
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
            self.assertEqual(payload["fixed_epochs"], 60)
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


class SerAggregationTest(unittest.TestCase):
    def test_default_seeds_match_reference_protocol(self):
        self.assertEqual(DEFAULT_SEEDS, (3949, 42, 2026, 1024, 666))

    def test_non_emodb_uses_fold_mean_per_seed_and_keeps_protocols_separate(self):
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
                        protocol="fixed",
                        fold=fold,
                        seed=seed,
                        targets=targets,
                        probabilities=probabilities,
                        paths=[f"fold{fold}/sample{index}.wav" for index in range(2)],
                    )
            # A different protocol under the same scan root must never be mixed in.
            save_fold(
                root,
                dataset="iemocap",
                protocol="nested",
                fold=1,
                seed=1,
                targets=[0, 1],
                paths=["nested/0.wav", "nested/1.wav"],
            )
            results = discover_fold_results(root, "iemocap", "fixed")
            self.assertEqual(len(results), 4)
            summary = aggregate_results(
                results,
                dataset="iemocap",
                protocol="fixed",
                expected_folds=(1, 2),
                expected_seeds=(1, 2),
                expected_samples=4,
                fixed_epochs=60,
            )
            self.assertEqual(summary["primary"]["method"], "mean_across_folds_per_seed")
            self.assertEqual(summary["primary"]["per_seed"]["1"]["accuracy"], 1.0)
            self.assertEqual(summary["primary"]["per_seed"]["2"]["accuracy"], 0.5)
            self.assertAlmostEqual(
                summary["primary"]["across_seeds"]["accuracy"]["mean"], 0.75
            )
            self.assertEqual(len(summary["fold_results"]), 4)

    def test_missing_grid_entry_and_duplicate_outer_path_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = save_fold(
                root,
                dataset="iemocap",
                protocol="nested",
                fold=1,
                seed=1,
                targets=[0, 1],
                paths=["shared.wav", "one.wav"],
            )
            results = discover_fold_results(root, "iemocap", "nested")
            with self.assertRaisesRegex(ValueError, "missing seed/fold"):
                aggregate_results(
                    results,
                    dataset="iemocap",
                    protocol="nested",
                    expected_folds=(1, 2),
                    expected_seeds=(1,),
                    expected_samples=4,
                )

            save_fold(
                root,
                dataset="iemocap",
                protocol="nested",
                fold=2,
                seed=1,
                targets=[2, 3],
                paths=["shared.wav", "two.wav"],
            )
            results = discover_fold_results(root, "iemocap", "nested")
            with self.assertRaisesRegex(ValueError, "unique paths"):
                aggregate_results(
                    results,
                    dataset="iemocap",
                    protocol="nested",
                    expected_folds=(1, 2),
                    expected_seeds=(1,),
                    expected_samples=4,
                )
            self.assertTrue((first / "metrics.json").is_file())

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
                        protocol="nested",
                        fold=fold,
                        seed=seed,
                        targets=targets,
                        paths=[f"speaker_fold_{fold}/{index}.wav" for index in range(count)],
                    )
                    offset += count
                self.assertEqual(offset, 535)
            results = discover_fold_results(root, "emodb", "nested")
            summary = aggregate_results(
                results,
                dataset="emodb",
                protocol="nested",
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
            for split in ("train", "valid"):
                (directory / f"{split}.tsv").write_text(
                    "/root/songhao/datasets/SER_DATASET\n"
                    f"{dataset}/fold{fold}/{split}.wav\t16000\n"
                )
                (directory / f"{split}.lbl").write_text(
                    f"{split}.wav\tneutral\n"
                )

    def test_batch_dry_run_prints_strict_automatic_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "runs"
            self.make_protocol_manifests(
                data_root, "ravdess-song", range(5)
            )
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
            self.assertEqual(result.stdout.count('"stage": "fixed"'), 5)
            self.assertIn("/fold_00/fixed", result.stdout)
            self.assertIn("/fold_04/fixed", result.stdout)
            self.assertIn("SUMMARY:", result.stdout)
            self.assertIn("--expected-seeds 7", result.stdout)
            self.assertIn("--fixed-epochs 3", result.stdout)
            config = "eat_base_sr16k_audiosetnorm_bs32_fixedep3"
            self.assertIn(f'"config_name": "{config}"', result.stdout)
            self.assertIn(str(output_root / f"ravdess-song/fixed/{config}/summary.json"), result.stdout)
            self.assertFalse(output_root.exists())

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
                    "--protocol", "fixed", "--fixed-epochs", "60",
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
            # The dry-run summary is shell-quoted with %q, so commas arrive as "\,".
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
                    "--datasets", "ravdess-song",
                    "--protocol", "fixed", "--folds", "2",
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
                    "--protocol", "fixed", "--folds", "0,2",
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
                    "--protocol", "fixed",
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


if __name__ == "__main__":
    unittest.main()
