#!/usr/bin/env python3
"""Run one EAT SER fold with the earlystop or fixed protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_SIZE = 32
DEFAULT_CONFIG_NAME = "eat_base_sr16k_audiosetnorm_bs32"
DEFAULT_PRETRAINED = Path(
    "/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt"
)
DEFAULT_FIXED_EPOCHS = 60
PROTOCOLS = ("earlystop", "fixed")
VALIDATION_METRIC_NAMES = ("loss", "accuracy", "war", "uar", "macro_f1")
EARLYSTOP_SUMMARY_SCHEMA = "eat2emo-ser-earlystop-v1"
STATS_STAGE = {
    "earlystop": "selection",
    "fixed": "fixed",
}


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    target_length: int
    first_fold: int
    last_fold: int
    max_epochs: int


DATASETS = {
    "emodb": DatasetSpec(7, 640, 1, 10, 80),
    "iemocap": DatasetSpec(4, 1024, 1, 5, 20),
    "ravdess-speech": DatasetSpec(8, 512, 0, 4, 80),
    "ravdess-song": DatasetSpec(6, 640, 0, 4, 80),
}


@dataclass(frozen=True)
class RunPlan:
    dataset: str
    protocol: str
    config_name: str
    fold: int
    seed: int
    train_subset: str
    valid_subset: str
    max_epochs: int | None
    fixed_epochs: int | None
    batch_size: int
    train_samples: int
    updates_per_epoch: int
    max_updates: int
    warmup_updates: int
    num_workers: int
    target_length: int
    num_classes: int
    data_dir: Path
    feature_cache_dir: Path
    norm_mode: str
    norm_stats_path: Path | None
    wandb_project: str | None
    wandb_run_name: str
    output_dir: Path
    pretrained: Path
    fairseq_root: Path
    python: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def count_manifest_samples(path: Path) -> int:
    """Count and minimally validate entries in an EAT TSV manifest."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing manifest: {path}") from exc
    if not lines or not lines[0].strip():
        raise ValueError(f"Manifest has no root line: {path}")
    entries = [line for line in lines[1:] if line.strip()]
    if not entries:
        raise ValueError(f"Manifest has no samples: {path}")
    for line_number, line in enumerate(entries, 2):
        fields = line.split("\t")
        try:
            frames = int(fields[1]) if len(fields) == 2 else 0
        except ValueError:
            frames = 0
        if len(fields) != 2 or not fields[0] or frames <= 0:
            raise ValueError(f"Malformed manifest line {line_number}: {path}")
    return len(entries)


def compute_schedule(samples: int, epochs: int, batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[int, int, int]:
    if samples <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("samples, epochs, and batch_size must all be positive")
    updates_per_epoch = math.ceil(samples / batch_size)
    max_updates = updates_per_epoch * epochs
    # Match the reference protocol exactly: floor(10%); tiny smoke runs may use 0.
    warmup_updates = int(max_updates * 0.1)
    return updates_per_epoch, max_updates, warmup_updates


def fold_dir_name(fold: int) -> str:
    return f"fold_{fold:02d}"


def requested_max_epochs(args: argparse.Namespace) -> int:
    return args.max_epochs or DATASETS[args.dataset].max_epochs


def requested_fixed_epochs(args: argparse.Namespace) -> int:
    return args.fixed_epochs or DEFAULT_FIXED_EPOCHS


def resolve_config_name(args: argparse.Namespace) -> str:
    """Keep smoke/non-default runs isolated without ambiguous protocol text."""
    name = args.config_name
    if not args.auto_config_name:
        return name
    if args.norm_mode == "fold":
        name = f"{name}_foldnorm"
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.protocol == "fixed":
        fixed_epochs = requested_fixed_epochs(args)
        if fixed_epochs != DEFAULT_FIXED_EPOCHS:
            name = f"{name}_fixedep{fixed_epochs}"
    else:
        max_epochs = requested_max_epochs(args)
        if max_epochs != DATASETS[args.dataset].max_epochs:
            name = f"{name}_maxep{max_epochs}"
    if args.batch_size != DEFAULT_BATCH_SIZE:
        name = f"{name}_bs{args.batch_size}"
    return name


def run_output_dir(args: argparse.Namespace) -> Path:
    return (
        args.output_root
        / args.dataset
        / args.protocol
        / args.config_name
        / f"seed_{args.seed}"
        / fold_dir_name(args.fold)
    )


def require_split(data_dir: Path, split: str) -> None:
    for suffix in ("tsv", "lbl"):
        path = data_dir / f"{split}.{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} file: {path}")


def norm_stats_path_for(args: argparse.Namespace) -> Path:
    return (
        args.data_root
        / "norm_stats"
        / args.dataset
        / STATS_STAGE[args.protocol]
        / f"{fold_dir_name(args.fold)}.json"
    )


def validate_norm_stats_available(args: argparse.Namespace) -> None:
    if args.norm_mode != "fold":
        return
    path = norm_stats_path_for(args)
    if not path.is_file():
        raise FileNotFoundError(f"Missing norm stats: {path}")


def build_run_plan(args: argparse.Namespace) -> RunPlan:
    spec = DATASETS[args.dataset]
    if args.protocol == "earlystop":
        train_subset = "inner_train"
        valid_subset = "inner_valid"
        max_epochs: int | None = requested_max_epochs(args)
        fixed_epochs: int | None = None
        epochs = max_epochs
    else:
        train_subset = "train"
        valid_subset = "valid"
        max_epochs = None
        fixed_epochs = requested_fixed_epochs(args)
        epochs = fixed_epochs

    data_dir = args.data_root / "manifests" / args.dataset / fold_dir_name(args.fold)
    require_split(data_dir, train_subset)
    require_split(data_dir, valid_subset)
    train_samples = count_manifest_samples(data_dir / f"{train_subset}.tsv")
    updates_per_epoch, max_updates, warmup_updates = compute_schedule(
        train_samples, epochs, args.batch_size
    )
    norm_stats_path = None
    if args.norm_mode == "fold":
        norm_stats_path = norm_stats_path_for(args)
        if not norm_stats_path.is_file():
            raise FileNotFoundError(f"Missing norm stats: {norm_stats_path}")
    return RunPlan(
        dataset=args.dataset,
        protocol=args.protocol,
        config_name=args.config_name,
        fold=args.fold,
        seed=args.seed,
        train_subset=train_subset,
        valid_subset=valid_subset,
        max_epochs=max_epochs,
        fixed_epochs=fixed_epochs,
        batch_size=args.batch_size,
        train_samples=train_samples,
        updates_per_epoch=updates_per_epoch,
        max_updates=max_updates,
        warmup_updates=warmup_updates,
        num_workers=args.num_workers,
        target_length=spec.target_length,
        num_classes=spec.num_classes,
        data_dir=data_dir,
        feature_cache_dir=args.data_root / "feature_cache" / args.dataset,
        norm_mode=args.norm_mode,
        norm_stats_path=norm_stats_path,
        wandb_project=args.wandb_project or None,
        wandb_run_name=args.wandb_run_name
        or (
            f"{args.dataset}-{args.protocol}-fold{args.fold:02d}-seed{args.seed}"
            f"-{args.pretrained.stem}"
        ),
        output_dir=run_output_dir(args),
        pretrained=args.pretrained,
        fairseq_root=args.fairseq_root,
        python=args.python,
    )


def hydra_train_command(plan: RunPlan) -> list[str]:
    output = plan.output_dir
    if plan.protocol == "earlystop":
        fairseq_valid_subset = plan.valid_subset
        disable_validation = "false"
    else:
        # Fixed training does not select a checkpoint; avoid loading outer-valid.
        fairseq_valid_subset = plan.train_subset
        disable_validation = "true"
    epochs = plan.max_epochs if plan.protocol == "earlystop" else plan.fixed_epochs
    assert epochs is not None
    command = [
        plan.python,
        str(plan.fairseq_root / "fairseq_cli" / "hydra_train.py"),
        "--config-dir",
        str(PROJECT_ROOT / "config"),
        "--config-name",
        "finetuning_SER",
        f"hydra.run.dir={output / 'hydra'}",
        f"common.user_dir={PROJECT_ROOT}",
        f"common.seed={plan.seed}",
        f"common.tensorboard_logdir={output / 'tb'}",
        f"task.data={plan.data_dir}",
        f"task.target_length={plan.target_length}",
        f"task.feature_cache_root={plan.feature_cache_dir}",
        f"task.norm_mode={plan.norm_mode}",
        f"model.model_path={plan.pretrained}",
        f"model.num_classes={plan.num_classes}",
        f"model.target_length={plan.target_length}",
        f"dataset.train_subset={plan.train_subset}",
        f"dataset.valid_subset={fairseq_valid_subset}",
        f"dataset.disable_validation={disable_validation}",
        f"dataset.batch_size={plan.batch_size}",
        f"dataset.num_workers={plan.num_workers}",
        "dataset.validate_interval=1",
        f"optimization.max_epoch={epochs}",
        f"optimization.max_update={plan.max_updates}",
        f"optimizer.groups.default.lr_scheduler.warmup_updates={plan.warmup_updates}",
        f"checkpoint.save_dir={output}",
        f"checkpoint.restore_file={output / '__fresh_start__.pt'}",
        "checkpoint.patience=-1",
        "checkpoint.no_last_checkpoints=true",
        "checkpoint.best_checkpoint_metric=uar",
        "checkpoint.maximize_best_checkpoint_metric=true",
    ]
    if plan.protocol == "fixed":
        command.extend(
            [
                "checkpoint.no_epoch_checkpoints=false",
                f"checkpoint.save_interval={epochs}",
            ]
        )
    else:
        command.extend(
            [
                "checkpoint.no_epoch_checkpoints=true",
                "checkpoint.save_interval=1",
            ]
        )
    if plan.norm_stats_path is not None:
        command.append(f"task.norm_stats_path={plan.norm_stats_path}")
    if plan.wandb_project:
        command.append(f"common.wandb_project={plan.wandb_project}")
    return command


def eval_command(plan: RunPlan, checkpoint: Path, selected_epoch: int | str | None) -> list[str]:
    command = [
        plan.python,
        str(PROJECT_ROOT / "evaluation" / "eval_ser.py"),
        "--data",
        str(plan.data_dir),
        "--checkpoint",
        str(checkpoint),
        "--model-dir",
        str(PROJECT_ROOT),
        "--output",
        str(plan.output_dir / "results"),
        "--dataset",
        plan.dataset,
        "--protocol",
        plan.protocol,
        "--config",
        plan.config_name,
        "--fold",
        str(plan.fold),
        "--seed",
        str(plan.seed),
        "--split",
        "valid",
        "--target-length",
        str(plan.target_length),
        "--norm-mode",
        plan.norm_mode,
    ]
    if plan.protocol == "earlystop":
        if selected_epoch is None or plan.max_epochs is None:
            raise ValueError("earlystop evaluation requires selected_epoch and max_epochs")
        command.extend(
            [
                "--selected-epoch",
                str(selected_epoch),
                "--max-epochs",
                str(plan.max_epochs),
            ]
        )
    else:
        if plan.fixed_epochs is None:
            raise ValueError("fixed evaluation requires fixed_epochs")
        command.extend(["--fixed-epochs", str(plan.fixed_epochs)])
    if plan.norm_mode == "fold":
        command.extend(["--norm-stats-path", str(plan.norm_stats_path)])
    else:
        command.extend(["--norm-mean", "-4.268", "--norm-std", "4.569"])
    return command


def _json_records(lines: Iterable[str]) -> Iterable[tuple[str, dict]]:
    decoder = json.JSONDecoder()
    for line in lines:
        start = line.find("{")
        if start < 0:
            continue
        try:
            record, _ = decoder.raw_decode(line[start:])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(record, dict):
            yield line[:start], record


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _valid_metric(record: dict, metric: str) -> float | None:
    for key, value in record.items():
        lowered = str(key).lower()
        if lowered == f"valid_{metric}" or lowered.endswith(f"_valid_{metric}"):
            parsed = _finite_float(value)
            if parsed is not None:
                return parsed
    return _finite_float(record.get(metric))


def validation_metrics_history(log_path: Path) -> list[dict[str, float | int]]:
    """Collect inner-valid metrics per epoch from a Fairseq JSON log."""
    if not log_path.is_file():
        return []
    by_epoch: dict[int, dict[str, float | int]] = {}
    for prefix, record in _json_records(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        lowered_keys = [str(key).lower() for key in record]
        is_validation = (
            "valid" in prefix.lower()
            or "valid" in str(record.get("split", "")).lower()
            or any("valid" in key for key in lowered_keys)
        )
        if not is_validation:
            continue
        try:
            epoch = int(float(record["epoch"]))
        except (KeyError, TypeError, ValueError):
            continue
        row = by_epoch.setdefault(epoch, {"epoch": epoch})
        for metric in VALIDATION_METRIC_NAMES:
            parsed = _valid_metric(record, metric)
            if parsed is not None:
                row[metric] = parsed
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def select_best_epoch(
    output_dir: Path, max_epochs: int
) -> tuple[int, list[dict[str, float | int]], dict[str, float | int]]:
    checkpoint = output_dir / "checkpoint_best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Training did not produce {checkpoint}")
    history = validation_metrics_history(output_dir / "train.log")
    observed_epochs = [int(row["epoch"]) for row in history]
    expected_epochs = list(range(1, max_epochs + 1))
    if observed_epochs != expected_epochs:
        raise RuntimeError(
            f"Expected inner-valid metrics for epochs 1..{max_epochs}, "
            f"got {observed_epochs}: {output_dir / 'train.log'}"
        )
    if any("uar" not in row for row in history):
        raise RuntimeError(f"Inner-valid UAR missing in {output_dir / 'train.log'}")
    best = min(history, key=lambda row: (-float(row["uar"]), int(row["epoch"])))
    best_epoch = int(best["epoch"])
    return best_epoch, history, best


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_run_manifest(
    plan: RunPlan,
    command: Sequence[str],
    *,
    best_epoch: int | None,
    status: str,
) -> None:
    payload = asdict(plan)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload.update(
        {
            "batch_size": plan.batch_size,
            "audio_normalization": {"source": "AudioSet", "mean": -4.268, "std": 4.569},
            "test_subset": "valid",
            "best_epoch": best_epoch,
            "status": status,
            "command": list(command),
        }
    )
    write_json(plan.output_dir / "run_manifest.json", payload)


def write_earlystop_summary(
    plan: RunPlan,
    history: Sequence[dict[str, float | int]],
    best_epoch: int,
    best_row: dict[str, float | int],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": EARLYSTOP_SUMMARY_SCHEMA,
        "protocol": plan.protocol,
        "dataset": plan.dataset,
        "config_name": plan.config_name,
        "fold": plan.fold,
        "seed": plan.seed,
        "pretrained": str(plan.pretrained),
        "train_subset": plan.train_subset,
        "valid_subset": plan.valid_subset,
        "test_subset": "valid",
        "max_epochs": plan.max_epochs,
        "best_epoch": best_epoch,
        "best_uar": float(best_row["uar"]),
        "metric_scale": {
            "loss": "cross_entropy_nats",
            "accuracy": "percent_0_100",
            "war": "percent_0_100",
            "uar": "percent_0_100",
            "macro_f1": "percent_0_100",
        },
        "tie_break": "earliest_epoch",
        "history": history,
        "best_metrics": best_row,
    }
    write_json(plan.output_dir / "earlystop_summary.json", payload)
    return payload


def run_and_tee(command: Sequence[str], log_path: Path, env: dict | None = None) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))


def promote_final_checkpoint(output_dir: Path, fixed_epochs: int) -> Path:
    candidates = [
        output_dir / f"checkpoint{fixed_epochs}.pt",
        output_dir / "checkpoint_last.pt",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    target = output_dir / "checkpoint_final.pt"
    if source is None:
        raise RuntimeError(
            "Training did not produce a final checkpoint; checked "
            + ", ".join(str(path) for path in candidates)
        )
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)
    return target


def ensure_fresh_output(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing run directory: {path}")


def validate_runtime(args: argparse.Namespace) -> None:
    spec = DATASETS[args.dataset]
    if not spec.first_fold <= args.fold <= spec.last_fold:
        raise ValueError(
            f"{args.dataset} fold must be {spec.first_fold}..{spec.last_fold}, got {args.fold}"
        )
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.config_name):
        raise ValueError("config-name may contain only letters, digits, '.', '_' and '-'")
    if args.protocol == "fixed" and args.max_epochs is not None:
        raise ValueError("--max-epochs applies only to --protocol earlystop")
    if not args.dry_run:
        train_entry = args.fairseq_root / "fairseq_cli" / "hydra_train.py"
        if not train_entry.is_file():
            raise FileNotFoundError(f"Fairseq training entrypoint not found: {train_entry}")
        if not args.pretrained.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")


def print_dry_run(plan: RunPlan, command: Sequence[str]) -> None:
    payload = asdict(plan)
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("TRAIN:", shlex.join(command))
    if plan.protocol == "earlystop":
        checkpoint = plan.output_dir / "checkpoint_best.pt"
        print("EVAL:", shlex.join(eval_command(plan, checkpoint, "<best_epoch>")))
    else:
        checkpoint = plan.output_dir / "checkpoint_final.pt"
        print("EVAL:", shlex.join(eval_command(plan, checkpoint, None)))


def execute_run(plan: RunPlan, dry_run: bool) -> None:
    command = hydra_train_command(plan)
    if dry_run:
        print_dry_run(plan, command)
        return

    plan.output_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(plan, command, best_epoch=None, status="training")
    train_env = os.environ.copy()
    if plan.wandb_project:
        train_env["WANDB_NAME"] = plan.wandb_run_name
        train_env["WANDB_RUN_GROUP"] = (
            f"{plan.dataset}-{plan.protocol}-{plan.config_name}"
        )
        train_env["WANDB_JOB_TYPE"] = plan.protocol
    run_and_tee(command, plan.output_dir / "train.log", env=train_env)

    if plan.protocol == "earlystop":
        assert plan.max_epochs is not None
        best_epoch, history, best_row = select_best_epoch(
            plan.output_dir, plan.max_epochs
        )
        write_earlystop_summary(plan, history, best_epoch, best_row)
        write_run_manifest(plan, command, best_epoch=best_epoch, status="trained")
        print(f"Selected epoch {best_epoch} by inner-valid UAR")
        checkpoint = plan.output_dir / "checkpoint_best.pt"
        selected_epoch: int | None = best_epoch
    else:
        assert plan.fixed_epochs is not None
        checkpoint = promote_final_checkpoint(plan.output_dir, plan.fixed_epochs)
        write_run_manifest(plan, command, best_epoch=None, status="trained")
        selected_epoch = None

    evaluation = eval_command(plan, checkpoint, selected_epoch)
    run_and_tee(evaluation, plan.output_dir / "eval.log")
    write_run_manifest(plan, command, best_epoch=selected_epoch, status="evaluated")


def make_parser() -> argparse.ArgumentParser:
    configured_name = os.environ.get("SER_CONFIG_NAME")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3949)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"training batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument("--protocol", choices=PROTOCOLS, default="earlystop")
    parser.add_argument(
        "--max-epochs",
        type=positive_int,
        default=None,
        help="earlystop inner-valid scan cap (default: dataset-specific)",
    )
    parser.add_argument(
        "--fixed-epochs",
        type=positive_int,
        default=None,
        help=f"fixed protocol training epochs (default: {DEFAULT_FIXED_EPOCHS})",
    )
    parser.add_argument(
        "--norm-mode",
        choices=("audioset", "fold"),
        default="audioset",
        help="audioset uses EAT fixed stats; fold uses per-fold train stats",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SER_NUM_WORKERS", "6")),
        help="DataLoader workers for training (default: 6 or SER_NUM_WORKERS)",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and print commands only")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("SER_DATA_ROOT", "/root/songhao/datasets/SER_DATASET")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("SER_OUTPUT_ROOT", "/root/songhao/experiments/EAT2EMO")),
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path(os.environ.get("SER_PRETRAINED", DEFAULT_PRETRAINED)),
    )
    parser.add_argument(
        "--fairseq-root",
        type=Path,
        default=Path(os.environ.get("FAIRSEQ_ROOT", "/root/songhao/ProJ/fairseq")),
    )
    parser.add_argument("--config-name", default=configured_name or DEFAULT_CONFIG_NAME)
    parser.set_defaults(auto_config_name=configured_name is None)
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argument_list = list(sys.argv[1:] if argv is None else argv)
    args = make_parser().parse_args(argument_list)
    # An explicit CLI name is an intentional namespace and must not be rewritten.
    if any(
        token == "--config-name" or token.startswith("--config-name=")
        for token in argument_list
    ):
        args.auto_config_name = False
    args.config_name = resolve_config_name(args)
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.pretrained = args.pretrained.expanduser().resolve()
    args.fairseq_root = args.fairseq_root.expanduser().resolve()
    try:
        validate_runtime(args)
        validate_norm_stats_available(args)
        plan = build_run_plan(args)
        ensure_fresh_output([plan.output_dir])
        execute_run(plan, args.dry_run)
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
