#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset


@dataclass
class UserDirModule:
    user_dir: str


def read_manifest(data_dir: Path, split: str):
    manifest_lines = (data_dir / f"{split}.tsv").read_text().splitlines()
    root = Path(manifest_lines[0])
    paths = [root / line.split("\t")[0] for line in manifest_lines[1:]]
    label_names = {}
    with (data_dir / "label_descriptors.csv").open() as stream:
        for index, label, *_ in csv.reader(stream):
            label_names[label.strip()] = int(index)
    labels = []
    with (data_dir / f"{split}.lbl").open() as stream:
        for line in stream:
            labels.append(label_names[line.rstrip().split("\t", 1)[1]])
    if len(paths) != len(labels):
        raise ValueError(f"Manifest has {len(paths)} files but {len(labels)} labels")
    return paths, labels, {index: label for label, index in label_names.items()}


class SerDataset(Dataset):
    def __init__(self, paths, labels, target_length, norm_mean, norm_std):
        self.paths = paths
        self.labels = labels
        self.target_length = target_length
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        waveform, sample_rate = sf.read(self.paths[index], dtype="float32")
        waveform = torch.from_numpy(waveform)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform - waveform.mean()
        features = torchaudio.compliance.kaldi.fbank(
            waveform.unsqueeze(0), htk_compat=True, sample_frequency=16000,
            use_energy=False, window_type="hanning", num_mel_bins=128,
            dither=0.0, frame_shift=10,
        ).unsqueeze(0)
        difference = self.target_length - features.shape[1]
        if difference > 0:
            features = F.pad(features, (0, 0, 0, difference))
        elif difference < 0:
            features = features[:, :self.target_length, :]
        features = (features - self.norm_mean) / (self.norm_std * 2)
        return features, self.labels[index]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned EAT SER checkpoint")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--norm-mean", type=float, default=-4.268)
    parser.add_argument("--norm-std", type=float, default=4.569)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    fairseq.utils.import_user_module(UserDirModule(str(args.model_dir.resolve())))
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(args.checkpoint)])
    model = models[0].eval().to(args.device)
    paths, labels, label_names = read_manifest(args.data, args.split)
    loader = DataLoader(
        SerDataset(paths, labels, args.target_length, args.norm_mean, args.norm_std),
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
    )

    predictions, targets = [], []
    with torch.no_grad():
        for features, target in loader:
            logits = model(features.to(args.device))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(target.tolist())

    class_ids = sorted(label_names)
    result = {
        "samples": len(targets),
        "accuracy": metrics.accuracy_score(targets, predictions),
        "war": metrics.accuracy_score(targets, predictions),
        "uar": metrics.recall_score(targets, predictions, labels=class_ids, average="macro", zero_division=0),
        "macro_f1": metrics.f1_score(targets, predictions, labels=class_ids, average="macro", zero_division=0),
    }
    confusion = metrics.confusion_matrix(targets, predictions, labels=class_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "confusion_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/pred"] + [label_names[index] for index in class_ids])
        for index, row in zip(class_ids, confusion):
            writer.writerow([label_names[index]] + row.tolist())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
