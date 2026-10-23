#!/usr/bin/env python3
"""Precompute SER fbank feature caches for a training stage.

This does not train.  It instantiates the same FileAudioDataset used by
fine-tuning, which reads the per-fold norm stats and writes one .npy per source
file under feature_cache/<dataset>/<target_length>_stats<hash>/...

Stages:
    selection -> earlystop training, inner_train
    fixed     -> fixed training, train
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from EAT2EMO.data.raw_audio_dataset import FileAudioDataset  # noqa: E402
from ser_protocol import DATASETS, fold_dir_name  # noqa: E402

torch.set_num_threads(1)

STAGE_SPLITS = {
    "selection": "inner_train",
    "fixed": "train",
}
DEFAULT_DATA_ROOT = os.environ.get(
    "SER_DATA_ROOT", "/root/songhao/datasets/SER_DATASET"
)


def build_dataset(
    manifest_path: Path,
    cache_root: Path,
    stats_path: Path,
    target_length: int,
    split: str,
) -> FileAudioDataset:
    return FileAudioDataset(
        manifest_path=str(manifest_path),
        sample_rate=16000,
        max_sample_size=325000,
        min_sample_size=0,
        pad=False,
        normalize=True,
        num_buckets=0,
        compute_mask=False,
        h5_format=False,
        downsr_16hz=True,
        wav2fbank=True,
        target_length=target_length,
        esc50_eval=False,
        spcv2_eval=False,
        roll_mag_aug=False,
        noise=False,
        train_mode=split,
        feature_cache_root=str(cache_root),
        norm_mode="fold",
        norm_stats_path=str(stats_path),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute fbank feature caches for SER training splits"
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_SPLITS),
        default="selection",
        help="selection for earlystop; fixed for fixed",
    )
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="cache root (default: <data-root>/feature_cache)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = DATASETS[args.dataset]
    allowed_folds = tuple(range(spec.first_fold, spec.last_fold + 1))
    if args.all_folds == (args.fold is not None):
        raise SystemExit("specify exactly one of --fold or --all-folds")
    folds = allowed_folds if args.all_folds else (args.fold,)
    invalid = [fold for fold in folds if fold not in allowed_folds]
    if invalid:
        raise SystemExit(
            f"invalid fold(s) {invalid} for {args.dataset}; expected "
            f"{allowed_folds[0]}..{allowed_folds[-1]}"
        )

    data_root = args.data_root.expanduser().resolve()
    cache_root = (
        args.cache_root.expanduser().resolve()
        if args.cache_root
        else data_root / "feature_cache"
    ) / args.dataset
    stats_root = data_root / "norm_stats" / args.dataset
    stats_path = stats_root / args.stage / f"{fold_dir_name(folds[0])}.json"
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)

    split = STAGE_SPLITS[args.stage]
    for fold in folds:
        folder = fold_dir_name(fold)
        manifest_dir = data_root / "manifests" / args.dataset / folder
        fold_stats = stats_root / args.stage / f"{folder}.json"
        if not fold_stats.is_file():
            raise FileNotFoundError(fold_stats)
        manifest = manifest_dir / f"{split}.tsv"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        print(
            f"=== {args.dataset} fold {fold} {args.stage} / {split} ===",
            flush=True,
        )
        dataset = build_dataset(
            manifest_path=manifest,
            cache_root=cache_root,
            stats_path=fold_stats,
            target_length=spec.target_length,
            split=split,
        )
        for index in range(len(dataset)):
            dataset[index]
            if (index + 1) % 500 == 0:
                print(
                    f"  {index + 1}/{len(dataset)} {split} cached",
                    flush=True,
                )
        print(f"  done {len(dataset)} {split} samples", flush=True)


if __name__ == "__main__":
    main()
