#!/usr/bin/env python3
"""Validate and aggregate completed SER fold evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from evaluation.eval_ser import (
        AUDIOSET_NORM_MEAN,
        AUDIOSET_NORM_STD,
        DEFAULT_TARGET_LENGTHS,
        EXPECTED_CLASSES,
        canonical_dataset,
        compute_ser_metrics,
        validate_prediction_arrays,
    )
except ModuleNotFoundError:  # Support ``python evaluation/summarize_ser.py``.
    from eval_ser import (  # type: ignore
        AUDIOSET_NORM_MEAN,
        AUDIOSET_NORM_STD,
        DEFAULT_TARGET_LENGTHS,
        EXPECTED_CLASSES,
        canonical_dataset,
        compute_ser_metrics,
        validate_prediction_arrays,
    )


METRIC_NAMES = ("accuracy", "war", "uar", "macro_f1", "loss")
FORMAL_STAGES = {"nested": "final", "fixed": "fixed"}
DEFAULT_FOLDS = {
    "emodb": tuple(range(1, 11)),
    "iemocap": tuple(range(1, 6)),
    "ravdess-speech": tuple(range(5)),
    "ravdess-song": tuple(range(5)),
}
EXPECTED_SAMPLES = {
    "emodb": 535,
    "iemocap": 5531,
    "ravdess-speech": 1440,
    "ravdess-song": 1012,
}
DEFAULT_SEEDS = (3949, 42, 2026, 1024, 666)


@dataclass(frozen=True)
class FoldResult:
    metrics_path: Path
    payload: dict[str, object]
    probabilities: np.ndarray
    targets: np.ndarray
    paths: tuple[str, ...]
    label_names: tuple[str, ...]

    @property
    def fold(self) -> int:
        return int(self.payload["fold"])

    @property
    def seed(self) -> int:
        return int(self.payload["seed"])


def mean_std(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize empty or non-finite metric values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "count": int(array.size),
    }


def _require_int(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _load_fold_result(metrics_path: Path) -> FoldResult:
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read metrics: {metrics_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"metrics payload must be an object: {metrics_path}")
    required_text = ("dataset", "protocol", "config", "stage", "split")
    for name in required_text:
        if not isinstance(payload.get(name), str) or not str(payload[name]).strip():
            raise ValueError(f"missing or invalid {name}: {metrics_path}")
    dataset = canonical_dataset(str(payload["dataset"]))
    protocol = str(payload["protocol"])
    if protocol not in FORMAL_STAGES:
        raise ValueError(f"invalid protocol {protocol!r}: {metrics_path}")
    if payload["stage"] != FORMAL_STAGES[protocol]:
        raise ValueError(
            f"non-formal stage {payload['stage']!r} for {protocol}: {metrics_path}"
        )
    if payload["split"] != "valid":
        raise ValueError(f"formal evaluation must use split 'valid': {metrics_path}")
    if payload.get("diagnostic_outer_validation") is True:
        raise ValueError(
            "diagnostic repeated-outer-valid evaluation is not a formal result: "
            f"{metrics_path}"
        )
    fold = _require_int(payload, "fold")
    seed = _require_int(payload, "seed")
    if fold < 0 or seed < 0:
        raise ValueError(f"fold and seed must be non-negative: {metrics_path}")
    if protocol == "fixed":
        fixed_epochs = _require_int(payload, "fixed_epochs")
        if fixed_epochs <= 0:
            raise ValueError(f"fixed_epochs must be positive: {metrics_path}")
        trained_epochs = _require_int(payload, "trained_epochs")
        if trained_epochs != fixed_epochs:
            raise ValueError(
                f"trained_epochs must equal fixed_epochs: {metrics_path}"
            )
    elif payload.get("fixed_epochs") is not None:
        raise ValueError(f"nested result must not set fixed_epochs: {metrics_path}")
    else:
        selected_epoch = _require_int(payload, "selected_epoch")
        trained_epochs = _require_int(payload, "trained_epochs")
        if selected_epoch <= 0 or trained_epochs != selected_epoch:
            raise ValueError(
                f"nested final must train exactly selected_epoch: {metrics_path}"
            )

    for metric in METRIC_NAMES:
        value = payload.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"missing or non-finite metric {metric}: {metrics_path}")

    label_names = payload.get("labels")
    expected_classes = EXPECTED_CLASSES[dataset]
    if (
        not isinstance(label_names, list)
        or len(label_names) != expected_classes
        or not all(isinstance(label, str) and label for label in label_names)
    ):
        raise ValueError(f"invalid labels for {dataset}: {metrics_path}")
    if payload.get("num_classes") != expected_classes:
        raise ValueError(f"invalid num_classes for {dataset}: {metrics_path}")
    if payload.get("target_length") != DEFAULT_TARGET_LENGTHS[dataset]:
        raise ValueError(f"invalid target_length for {dataset}: {metrics_path}")
    norm_mode = payload.get("norm_mode", "audioset")
    if norm_mode not in ("audioset", "fold"):
        raise ValueError(f"invalid norm_mode: {metrics_path}")
    norm_mean = payload.get("norm_mean")
    norm_std = payload.get("norm_std")
    if (
        isinstance(norm_mean, bool)
        or not isinstance(norm_mean, (int, float))
        or not math.isfinite(float(norm_mean))
    ):
        raise ValueError(f"invalid norm_mean: {metrics_path}")
    if (
        isinstance(norm_std, bool)
        or not isinstance(norm_std, (int, float))
        or not math.isfinite(float(norm_std))
        or float(norm_std) <= 0
    ):
        raise ValueError(f"invalid norm_std: {metrics_path}")
    if norm_mode == "audioset":
        if not math.isclose(float(norm_mean), AUDIOSET_NORM_MEAN) or not math.isclose(
            float(norm_std), AUDIOSET_NORM_STD
        ):
            raise ValueError(f"non-AudioSet norm values in audioset mode: {metrics_path}")
    else:
        if not isinstance(payload.get("norm_stats_path"), str) or not payload["norm_stats_path"]:
            raise ValueError(f"fold mode requires norm_stats_path: {metrics_path}")
        norm_stats_sha256 = payload.get("norm_stats_sha256")
        if (
            not isinstance(norm_stats_sha256, str)
            or len(norm_stats_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in norm_stats_sha256)
        ):
            raise ValueError(f"fold mode requires norm_stats_sha256: {metrics_path}")

    prediction_path = metrics_path.with_name("predictions.npz")
    paths_path = metrics_path.with_name("paths.txt")
    confusion_path = metrics_path.with_name("confusion_matrix.csv")
    for path in (prediction_path, paths_path, confusion_path):
        if not path.is_file():
            raise ValueError(f"missing fold artifact: {path}")
    try:
        with np.load(prediction_path, allow_pickle=False) as archive:
            probabilities = np.asarray(archive["probabilities"], dtype=np.float64)
            targets = np.asarray(archive["targets"])
            saved_label_names = tuple(str(value) for value in archive["label_names"].tolist())
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"invalid prediction artifact {prediction_path}: {error}") from error
    probabilities, targets = validate_prediction_arrays(
        probabilities, targets, num_classes=expected_classes
    )
    if saved_label_names != tuple(label_names):
        raise ValueError(f"prediction labels do not match metrics: {metrics_path}")
    paths = tuple(paths_path.read_text(encoding="utf-8").splitlines())
    if len(paths) != len(targets) or any(not path for path in paths):
        raise ValueError(f"paths do not align with predictions: {metrics_path}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"duplicate path within fold: {metrics_path}")
    if payload.get("samples") != len(targets):
        raise ValueError(f"sample count does not match predictions: {metrics_path}")

    recomputed, _ = compute_ser_metrics(
        probabilities,
        targets,
        loss=float(payload["loss"]),
        average_over_present_classes=True,
    )
    for metric in ("accuracy", "war", "uar", "macro_f1"):
        if not math.isclose(
            float(payload[metric]), float(recomputed[metric]), rel_tol=1e-6, abs_tol=1e-7
        ):
            raise ValueError(f"saved {metric} does not match predictions: {metrics_path}")
    if payload.get("present_classes") != recomputed["present_classes"]:
        raise ValueError(f"present_classes does not match targets: {metrics_path}")
    if payload.get("metric_average_classes") != recomputed["metric_average_classes"]:
        raise ValueError(f"metric_average_classes has the wrong policy: {metrics_path}")

    return FoldResult(
        metrics_path=metrics_path,
        payload=payload,
        probabilities=probabilities,
        targets=targets,
        paths=paths,
        label_names=tuple(label_names),
    )


def discover_fold_results(
    root: Path,
    dataset: str,
    protocol: str,
    *,
    config: str | None = None,
) -> list[FoldResult]:
    """Find only formal results for one dataset/protocol/config namespace."""

    selected_paths: list[Path] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot inspect metrics {metrics_path}: {error}") from error
        if not isinstance(payload, dict) or "dataset" not in payload:
            continue
        try:
            payload_dataset = canonical_dataset(str(payload["dataset"]))
        except argparse.ArgumentTypeError as error:
            raise ValueError(f"invalid dataset metadata: {metrics_path}") from error
        if payload_dataset != dataset or payload.get("protocol") != protocol:
            continue
        if payload.get("stage") == "selection":
            continue
        if payload.get("diagnostic_outer_validation") is True:
            continue
        if config is not None and payload.get("config") != config:
            continue
        selected_paths.append(metrics_path)
    if not selected_paths:
        suffix = f" and config {config!r}" if config else ""
        raise ValueError(
            f"no formal {dataset}/{protocol} metrics found under {root}{suffix}"
        )
    results = [_load_fold_result(path) for path in selected_paths]
    configs = {str(result.payload["config"]) for result in results}
    if len(configs) != 1:
        raise ValueError(
            "multiple configs matched; pass --config to keep experiments isolated: "
            + ", ".join(sorted(configs))
        )
    return results


def _validate_run_grid(
    results: Sequence[FoldResult],
    *,
    dataset: str,
    protocol: str,
    expected_folds: Sequence[int],
    expected_seeds: Sequence[int],
    expected_samples: int,
    fixed_epochs: int | None,
) -> dict[tuple[int, int], FoldResult]:
    expected_fold_set = set(expected_folds)
    expected_seed_set = set(expected_seeds)
    if len(expected_fold_set) != len(expected_folds) or not expected_folds:
        raise ValueError("expected folds must be non-empty and unique")
    if len(expected_seed_set) != len(expected_seeds) or not expected_seeds:
        raise ValueError("expected seeds must be non-empty and unique")
    grid: dict[tuple[int, int], FoldResult] = {}
    for result in results:
        payload = result.payload
        if canonical_dataset(str(payload["dataset"])) != dataset:
            raise ValueError(f"dataset mismatch: {result.metrics_path}")
        if payload["protocol"] != protocol:
            raise ValueError(f"protocol mismatch: {result.metrics_path}")
        if result.fold not in expected_fold_set:
            raise ValueError(f"unexpected fold {result.fold}: {result.metrics_path}")
        if result.seed not in expected_seed_set:
            raise ValueError(f"unexpected seed {result.seed}: {result.metrics_path}")
        key = (result.seed, result.fold)
        if key in grid:
            raise ValueError(
                f"duplicate seed/fold result {key}: {grid[key].metrics_path}, "
                f"{result.metrics_path}"
            )
        grid[key] = result
    missing = [
        (seed, fold)
        for seed in expected_seeds
        for fold in expected_folds
        if (seed, fold) not in grid
    ]
    if missing:
        raise ValueError(
            "missing seed/fold results: "
            + ", ".join(f"seed={seed}/fold={fold}" for seed, fold in missing)
        )

    configs = {str(result.payload["config"]) for result in results}
    if len(configs) != 1:
        raise ValueError("results from different configs cannot be aggregated")
    norm_modes = {str(result.payload.get("norm_mode", "audioset")) for result in results}
    if len(norm_modes) != 1:
        raise ValueError("results from different norm modes cannot be aggregated")
    if protocol == "fixed":
        epochs = {int(result.payload["fixed_epochs"]) for result in results}
        if len(epochs) != 1:
            raise ValueError("fixed results with different fixed_epochs cannot be mixed")
        actual_epochs = next(iter(epochs))
        if fixed_epochs is not None and actual_epochs != fixed_epochs:
            raise ValueError(
                f"expected fixed_epochs={fixed_epochs}, found {actual_epochs}"
            )

    baseline_labels = results[0].label_names
    for result in results:
        if result.label_names != baseline_labels:
            raise ValueError(f"label order differs across folds: {result.metrics_path}")
    for fold in expected_folds:
        baseline = grid[(expected_seeds[0], fold)]
        for seed in expected_seeds[1:]:
            candidate = grid[(seed, fold)]
            if candidate.paths != baseline.paths or not np.array_equal(
                candidate.targets, baseline.targets
            ):
                raise ValueError(
                    f"sample paths/targets differ across seeds for fold {fold}"
                )
    for seed in expected_seeds:
        seed_paths = [
            path for fold in expected_folds for path in grid[(seed, fold)].paths
        ]
        if len(seed_paths) != expected_samples:
            raise ValueError(
                f"seed {seed} has {len(seed_paths)} outer-valid rows; "
                f"expected {expected_samples}"
            )
        if len(set(seed_paths)) != expected_samples:
            raise ValueError(
                f"seed {seed} outer-valid folds do not contain {expected_samples} "
                "unique paths"
            )
    return grid


def pool_out_of_fold_results(
    fold_results: Sequence[FoldResult],
) -> dict[str, object]:
    """Pool one prediction per sample and average over the complete class set."""

    if not fold_results:
        raise ValueError("cannot pool an empty fold list")
    probabilities = np.concatenate(
        [result.probabilities for result in fold_results], axis=0
    )
    targets = np.concatenate([result.targets for result in fold_results], axis=0)
    paths = [path for result in fold_results for path in result.paths]
    if len(paths) != len(set(paths)):
        raise ValueError("OOF folds contain duplicate sample paths")
    weighted_loss = sum(
        float(result.payload["loss"]) * len(result.targets)
        for result in fold_results
    ) / len(targets)
    metrics, _ = compute_ser_metrics(
        probabilities,
        targets,
        loss=weighted_loss,
        average_over_present_classes=False,
    )
    metrics["folds"] = len(fold_results)
    return metrics


def aggregate_results(
    results: Sequence[FoldResult],
    *,
    dataset: str,
    protocol: str,
    expected_folds: Sequence[int],
    expected_seeds: Sequence[int],
    expected_samples: int,
    fixed_epochs: int | None = None,
) -> dict[str, object]:
    grid = _validate_run_grid(
        results,
        dataset=dataset,
        protocol=protocol,
        expected_folds=expected_folds,
        expected_seeds=expected_seeds,
        expected_samples=expected_samples,
        fixed_epochs=fixed_epochs,
    )
    sorted_results = [
        grid[(seed, fold)] for seed in expected_seeds for fold in expected_folds
    ]
    run_rows = [
        {
            "seed": result.seed,
            "fold": result.fold,
            **{metric: float(result.payload[metric]) for metric in METRIC_NAMES},
            "samples": int(result.payload["samples"]),
            "metrics_path": str(result.metrics_path),
        }
        for result in sorted_results
    ]
    fold_statistics = {
        "per_fold_across_seeds": {
            str(fold): {
                metric: mean_std(
                    float(grid[(seed, fold)].payload[metric])
                    for seed in expected_seeds
                )
                for metric in METRIC_NAMES
            }
            for fold in expected_folds
        },
        "per_seed_across_folds": {
            str(seed): {
                metric: mean_std(
                    float(grid[(seed, fold)].payload[metric])
                    for fold in expected_folds
                )
                for metric in METRIC_NAMES
            }
            for seed in expected_seeds
        },
        "all_fold_runs": {
            metric: mean_std(float(row[metric]) for row in run_rows)
            for metric in METRIC_NAMES
        },
    }

    if dataset == "emodb":
        per_seed = {
            str(seed): pool_out_of_fold_results(
                [grid[(seed, fold)] for fold in expected_folds]
            )
            for seed in expected_seeds
        }
        primary_description = "pooled_10_fold_oof"
    else:
        per_seed = {
            str(seed): {
                metric: float(fold_statistics["per_seed_across_folds"][str(seed)][metric]["mean"])
                for metric in METRIC_NAMES
            }
            for seed in expected_seeds
        }
        primary_description = "mean_across_folds_per_seed"
    primary_across_seeds = {
        metric: mean_std(float(per_seed[str(seed)][metric]) for seed in expected_seeds)
        for metric in METRIC_NAMES
    }
    fixed_epoch_value = (
        int(sorted_results[0].payload["fixed_epochs"])
        if protocol == "fixed"
        else None
    )
    return {
        "dataset": dataset,
        "protocol": protocol,
        "config": str(sorted_results[0].payload["config"]),
        "norm_mode": str(sorted_results[0].payload.get("norm_mode", "audioset")),
        "fixed_epochs": fixed_epoch_value,
        "expected_folds": list(expected_folds),
        "expected_seeds": list(expected_seeds),
        "expected_samples_per_seed": expected_samples,
        "complete": True,
        "primary": {
            "method": primary_description,
            "per_seed": per_seed,
            "across_seeds": primary_across_seeds,
        },
        "fold_statistics_supplementary": fold_statistics,
        "fold_results": run_rows,
    }


def write_summary(summary: dict[str, object], output: Path) -> tuple[Path, Path]:
    if output.suffix.lower() != ".json":
        output = output / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = output.with_name(f"{output.stem}_folds.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["seed", "fold", *METRIC_NAMES, "samples", "metrics_path"],
        )
        writer.writeheader()
        writer.writerows(summary["fold_results"])
    return output, csv_path


def _parse_int_list(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from error
    if not values or len(set(values)) != len(values) or any(number < 0 for number in values):
        raise argparse.ArgumentTypeError(
            f"{name} must contain unique non-negative integers"
        )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly validate and summarize SER fold/seed results"
    )
    parser.add_argument("root", type=Path, help="experiment directory to scan recursively")
    parser.add_argument("--dataset", type=canonical_dataset, required=True)
    parser.add_argument("--protocol", choices=tuple(FORMAL_STAGES), required=True)
    parser.add_argument("--config")
    parser.add_argument("--fixed-epochs", type=int)
    parser.add_argument("--expected-folds")
    parser.add_argument(
        "--expected-seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.fixed_epochs is not None and args.fixed_epochs <= 0:
        parser.error("--fixed-epochs must be positive")
    try:
        expected_folds = (
            _parse_int_list(args.expected_folds, "--expected-folds")
            if args.expected_folds
            else DEFAULT_FOLDS[args.dataset]
        )
        expected_seeds = _parse_int_list(args.expected_seeds, "--expected-seeds")
        results = discover_fold_results(
            args.root, args.dataset, args.protocol, config=args.config
        )
        summary = aggregate_results(
            results,
            dataset=args.dataset,
            protocol=args.protocol,
            expected_folds=expected_folds,
            expected_seeds=expected_seeds,
            expected_samples=EXPECTED_SAMPLES[args.dataset],
            fixed_epochs=args.fixed_epochs,
        )
        output, csv_path = write_summary(
            summary,
            args.output
            or args.root / f"summary_{args.dataset}_{args.protocol}.json",
        )
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(output)
    print(csv_path)


if __name__ == "__main__":
    main()
