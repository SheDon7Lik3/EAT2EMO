#!/usr/bin/env python3
"""Run one EAT SER fold with the frozen nested or fixed protocol."""

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
BATCH_SIZE = 32
DEFAULT_FIXED_EPOCHS = 60
DEFAULT_CONFIG_NAME = "eat_base_sr16k_audiosetnorm_bs32"
DEFAULT_PRETRAINED = Path(
    "/root/songhao/ProJ/resources/EAT_Weight/EAT-base_epoch30_pt.pt"
)


@dataclass(frozen=True)
class DatasetSpec:
    num_classes: int
    target_length: int
    first_fold: int
    last_fold: int
    selection_epochs: int


DATASETS = {
    "emodb": DatasetSpec(7, 640, 1, 10, 80),
    "iemocap": DatasetSpec(4, 1024, 1, 5, 20),
    "ravdess-speech": DatasetSpec(8, 512, 0, 4, 80),
    "ravdess-song": DatasetSpec(6, 640, 0, 4, 80),
}


@dataclass(frozen=True)
class StagePlan:
    dataset: str
    protocol: str
    config_name: str
    fold: int
    seed: int
    stage: str
    train_subset: str
    valid_subset: str
    epochs: int
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


def compute_schedule(samples: int, epochs: int, batch_size: int = BATCH_SIZE) -> tuple[int, int, int]:
    if samples <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("samples, epochs, and batch_size must all be positive")
    updates_per_epoch = math.ceil(samples / batch_size)
    max_updates = updates_per_epoch * epochs
    # Match the reference protocol exactly: floor(10%); tiny smoke runs may use 0.
    warmup_updates = int(max_updates * 0.1)
    return updates_per_epoch, max_updates, warmup_updates


def fold_dir_name(fold: int) -> str:
    return f"fold_{fold:02d}"


def selection_max_epochs(args: argparse.Namespace) -> int:
    return args.selection_max_epochs or DATASETS[args.dataset].selection_epochs


def resolve_config_name(args: argparse.Namespace) -> str:
    """Keep smoke/non-default epoch runs isolated from formal defaults."""
    name = args.config_name
    if not args.auto_config_name:
        return name
    if args.norm_mode == "fold":
        name = f"{name}_foldnorm"
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.protocol == "fixed" and args.fixed_epochs != DEFAULT_FIXED_EPOCHS:
        return f"{name}_fixedep{args.fixed_epochs}"
    default_selection_epochs = DATASETS[args.dataset].selection_epochs
    selected_max = selection_max_epochs(args)
    if args.protocol == "nested" and selected_max != default_selection_epochs:
        return f"{name}_selectmaxep{selected_max}"
    return name


def stage_output_dir(args: argparse.Namespace, stage: str) -> Path:
    return (
        args.output_root
        / args.dataset
        / args.protocol
        / args.config_name
        / f"seed_{args.seed}"
        / fold_dir_name(args.fold)
        / stage
    )


def require_split(data_dir: Path, split: str) -> None:
    for suffix in ("tsv", "lbl"):
        path = data_dir / f"{split}.{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} file: {path}")


def norm_stats_path_for(args: argparse.Namespace, stage: str) -> Path:
    return (
        args.data_root
        / "norm_stats"
        / args.dataset
        / stage
        / f"{fold_dir_name(args.fold)}.json"
    )


def validate_norm_stats_available(args: argparse.Namespace) -> None:
    if args.norm_mode != "fold":
        return
    if args.protocol == "fixed":
        stages = ("fixed",)
    else:
        stages = []
        if args.stage in ("all", "select"):
            stages.append("selection")
        if args.stage in ("all", "final"):
            stages.append("final")
    for stage in stages:
        path = norm_stats_path_for(args, stage)
        if not path.is_file():
            raise FileNotFoundError(f"Missing norm stats: {path}")


def build_stage_plan(
    args: argparse.Namespace,
    stage: str,
    epochs: int,
) -> StagePlan:
    spec = DATASETS[args.dataset]
    is_selection = stage == "selection"
    train_subset = "inner_train" if is_selection else "train"
    valid_subset = "inner_valid" if is_selection else "valid"
    data_dir = args.data_root / "manifests" / args.dataset / fold_dir_name(args.fold)
    require_split(data_dir, train_subset)
    require_split(data_dir, valid_subset)
    train_samples = count_manifest_samples(data_dir / f"{train_subset}.tsv")
    updates_per_epoch, max_updates, warmup_updates = compute_schedule(train_samples, epochs)
    norm_stats_path = None
    if args.norm_mode == "fold":
        norm_stats_path = norm_stats_path_for(args, stage)
        if not norm_stats_path.is_file():
            raise FileNotFoundError(f"Missing norm stats: {norm_stats_path}")
    return StagePlan(
        dataset=args.dataset,
        protocol=args.protocol,
        config_name=args.config_name,
        fold=args.fold,
        seed=args.seed,
        stage=stage,
        train_subset=train_subset,
        valid_subset=valid_subset,
        epochs=epochs,
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
        or f"{args.dataset}-{args.protocol}-fold{args.fold:02d}-seed{args.seed}-{stage}-{args.pretrained.stem}",
        output_dir=stage_output_dir(args, stage),
        pretrained=args.pretrained,
        fairseq_root=args.fairseq_root,
        python=args.python,
    )


def hydra_train_command(plan: StagePlan) -> list[str]:
    validation_enabled = plan.stage == "selection"
    # Fairseq 0.12.2 eagerly loads valid_subset even when validation is disabled.
    # Point it at train in formal stages so sealed outer-valid is not touched.
    fairseq_valid_subset = plan.valid_subset if validation_enabled else plan.train_subset
    output = plan.output_dir
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
        f"dataset.disable_validation={str(not validation_enabled).lower()}",
        f"dataset.batch_size={BATCH_SIZE}",
        f"dataset.num_workers={plan.num_workers}",
        "dataset.validate_interval=1",
        f"optimization.max_epoch={plan.epochs}",
        f"optimization.max_update={plan.max_updates}",
        f"optimizer.groups.default.lr_scheduler.warmup_updates={plan.warmup_updates}",
        f"checkpoint.save_dir={output}",
        f"checkpoint.restore_file={output / '__fresh_start__.pt'}",
        "checkpoint.patience=-1",
        "checkpoint.no_epoch_checkpoints=true",
        "checkpoint.no_last_checkpoints=false",
        "checkpoint.best_checkpoint_metric=uar",
        "checkpoint.maximize_best_checkpoint_metric=true",
    ]
    if plan.norm_stats_path is not None:
        command.append(f"task.norm_stats_path={plan.norm_stats_path}")
    if plan.wandb_project:
        command.append(f"common.wandb_project={plan.wandb_project}")
    return command


def eval_command(plan: StagePlan, checkpoint: Path, selected_epoch: int | None) -> list[str]:
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
        "--stage",
        plan.stage,
        "--split",
        "valid",
        "--target-length",
        str(plan.target_length),
        "--norm-mode",
        plan.norm_mode,
    ]
    if plan.norm_mode == "fold":
        command.extend(["--norm-stats-path", str(plan.norm_stats_path)])
    else:
        command.extend(["--norm-mean", "-4.268", "--norm-std", "4.569"])
    if plan.protocol == "fixed":
        command.extend(["--fixed-epochs", str(plan.epochs)])
    else:
        if selected_epoch is None:
            raise ValueError("nested final evaluation requires selected_epoch")
        command.extend(["--selected-epoch", str(selected_epoch)])
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


def validation_history(log_path: Path) -> list[dict[str, float | int]]:
    if not log_path.is_file():
        return []
    by_epoch: dict[int, float] = {}
    for prefix, record in _json_records(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        score = next(
            (value for key, value in record.items() if key.endswith("valid_uar")),
            None,
        )
        is_validation = "valid" in prefix.lower() or "valid" in str(record.get("split", "")).lower()
        if score is None and is_validation:
            score = record.get("uar")
        if score is None:
            continue
        try:
            epoch = int(float(record["epoch"]))
            by_epoch[epoch] = float(score)
        except (KeyError, TypeError, ValueError):
            continue
    return [{"epoch": epoch, "uar": by_epoch[epoch]} for epoch in sorted(by_epoch)]


def select_best_epoch(output_dir: Path, max_epochs: int) -> tuple[int, list[dict[str, float | int]]]:
    checkpoint = output_dir / "checkpoint_best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"Selection did not produce {checkpoint}")
    history = validation_history(output_dir / "train.log")
    observed_epochs = [int(row["epoch"]) for row in history]
    expected_epochs = list(range(1, max_epochs + 1))
    if observed_epochs != expected_epochs:
        raise RuntimeError(
            f"Expected inner-valid UAR for epochs 1..{max_epochs}, got {observed_epochs}: "
            f"{output_dir / 'train.log'}"
        )
    best = min(history, key=lambda row: (-float(row["uar"]), int(row["epoch"])))
    epoch = int(best["epoch"])
    return epoch, history


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_run_manifest(plan: StagePlan, command: Sequence[str], selected_epoch: int | None) -> None:
    payload = asdict(plan)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload.update(
        {
            "batch_size": BATCH_SIZE,
            "audio_normalization": {"source": "AudioSet", "mean": -4.268, "std": 4.569},
            "trained_epochs": plan.epochs,
            "fixed_epochs": plan.epochs if plan.protocol == "fixed" else None,
            "selected_epoch": selected_epoch,
            "validation_disabled": plan.stage != "selection",
            "fairseq_valid_subset": (
                plan.valid_subset if plan.stage == "selection" else plan.train_subset
            ),
            "command": list(command),
        }
    )
    write_json(plan.output_dir / "run_manifest.json", payload)


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


def promote_final_checkpoint(output_dir: Path) -> Path:
    source = output_dir / "checkpoint_last.pt"
    target = output_dir / "checkpoint_final.pt"
    if not source.is_file():
        raise RuntimeError(f"Training did not produce {source}")
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
            raise FileExistsError(f"Refusing to overwrite existing stage directory: {path}")


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
    if args.protocol == "fixed" and args.stage != "all":
        raise ValueError("--stage applies only to --protocol nested")
    if not args.dry_run:
        train_entry = args.fairseq_root / "fairseq_cli" / "hydra_train.py"
        if not train_entry.is_file():
            raise FileNotFoundError(f"Fairseq training entrypoint not found: {train_entry}")
        if not args.pretrained.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained}")


def read_selection_summary(path: Path, args: argparse.Namespace) -> int:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing selection summary: {path}; run --stage select first"
        ) from exc
    if not isinstance(summary, dict):
        raise ValueError(f"Selection summary must be a JSON object: {path}")
    expected = {
        "dataset": args.dataset,
        "protocol": "nested",
        "config_name": args.config_name,
        "fold": args.fold,
        "seed": args.seed,
        "max_epochs": selection_max_epochs(args),
        "pretrained": str(args.pretrained.resolve()),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Selection summary {key}={summary.get(key)!r}, expected {value!r}")
    epoch = summary.get("best_epoch")
    if not isinstance(epoch, int) or not 1 <= epoch <= expected["max_epochs"]:
        raise ValueError(f"Invalid best_epoch in {path}: {epoch!r}")
    return epoch


def print_dry_run(plan: StagePlan, command: Sequence[str], selected_epoch: int | None) -> None:
    payload = asdict(plan)
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
    payload["selected_epoch"] = selected_epoch
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("TRAIN:", shlex.join(command))
    if plan.stage != "selection":
        checkpoint = plan.output_dir / "checkpoint_final.pt"
        print("EVAL:", shlex.join(eval_command(plan, checkpoint, selected_epoch)))


def execute_stage(plan: StagePlan, selected_epoch: int | None, dry_run: bool) -> int | None:
    command = hydra_train_command(plan)
    if dry_run:
        print_dry_run(plan, command, selected_epoch)
        return selected_epoch

    plan.output_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(plan, command, selected_epoch)
    train_env = os.environ.copy()
    if plan.wandb_project:
        train_env["WANDB_NAME"] = plan.wandb_run_name
        # Keep all folds/seeds of one dataset/protocol/config together in W&B.
        train_env["WANDB_RUN_GROUP"] = (
            f"{plan.dataset}-{plan.protocol}-{plan.config_name}"
        )
        train_env["WANDB_JOB_TYPE"] = plan.stage
    run_and_tee(command, plan.output_dir / "train.log", env=train_env)
    if plan.stage == "selection":
        best_epoch, history = select_best_epoch(plan.output_dir, plan.epochs)
        score_by_epoch = {int(row["epoch"]): float(row["uar"]) for row in history}
        write_json(
            plan.output_dir / "selection_summary.json",
            {
                "dataset": plan.dataset,
                "protocol": plan.protocol,
                "config_name": plan.config_name,
                "pretrained": str(plan.pretrained.resolve()),
                "fold": plan.fold,
                "seed": plan.seed,
                "metric": "uar",
                "tie_break": "earliest_epoch",
                "best_epoch": best_epoch,
                "best_uar": score_by_epoch.get(best_epoch),
                "max_epochs": plan.epochs,
                "history": history,
            },
        )
        print(f"Selected epoch {best_epoch} by inner-valid UAR")
        return best_epoch

    checkpoint = promote_final_checkpoint(plan.output_dir)
    evaluation = eval_command(plan, checkpoint, selected_epoch)
    run_and_tee(evaluation, plan.output_dir / "eval.log")
    return selected_epoch


def make_parser() -> argparse.ArgumentParser:
    configured_name = os.environ.get("SER_CONFIG_NAME")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3949)
    parser.add_argument("--protocol", choices=("nested", "fixed"), default="nested")
    parser.add_argument("--stage", choices=("all", "select", "final"), default="all")
    parser.add_argument(
        "--norm-mode",
        choices=("audioset", "fold"),
        default="audioset",
        help="audioset uses EAT fixed stats; fold uses per-fold train stats",
    )
    parser.add_argument("--fixed-epochs", type=positive_int, default=DEFAULT_FIXED_EPOCHS)
    parser.add_argument(
        "--selection-max-epochs",
        type=positive_int,
        help="override the dataset default (intended for smoke tests)",
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
        if args.protocol == "fixed":
            plan = build_stage_plan(args, "fixed", args.fixed_epochs)
            ensure_fresh_output([plan.output_dir])
            execute_stage(plan, selected_epoch=None, dry_run=args.dry_run)
            return 0

        selection_dir = stage_output_dir(args, "selection")
        final_dir = stage_output_dir(args, "final")
        requested = []
        if args.stage in ("all", "select"):
            requested.append(selection_dir)
        if args.stage in ("all", "final"):
            requested.append(final_dir)
        ensure_fresh_output(requested)

        selected_epoch: int | None = None
        if args.stage in ("all", "select"):
            selection = build_stage_plan(args, "selection", selection_max_epochs(args))
            selected_epoch = execute_stage(selection, selected_epoch=None, dry_run=args.dry_run)
        if args.stage in ("all", "final"):
            if args.dry_run and args.stage == "all":
                if selection_max_epochs(args) == 1:
                    selected_epoch = 1
                    final = build_stage_plan(args, "final", selected_epoch)
                    execute_stage(final, selected_epoch=selected_epoch, dry_run=True)
                else:
                    final_template = build_stage_plan(args, "final", 1)
                    print(
                        "FINAL: selected epoch will be read from "
                        f"{selection_dir / 'selection_summary.json'}; "
                        f"train_samples={final_template.train_samples}, "
                        f"updates_per_epoch={final_template.updates_per_epoch}"
                    )
                return 0
            if selected_epoch is None:
                selected_epoch = read_selection_summary(selection_dir / "selection_summary.json", args)
            final = build_stage_plan(args, "final", selected_epoch)
            execute_stage(final, selected_epoch=selected_epoch, dry_run=args.dry_run)
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
