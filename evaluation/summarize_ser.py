#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Summarize cross-validation SER metrics")
    parser.add_argument("root", type=Path, help="Directory containing fold_*/test/metrics.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metric_files = sorted(args.root.glob("fold_*/test/metrics.json"))
    if not metric_files:
        raise SystemExit(f"No fold metrics found under {args.root}")
    folds = {path.parents[1].name: json.loads(path.read_text()) for path in metric_files}
    metric_names = ("accuracy", "war", "uar", "macro_f1")
    summary = {
        "num_folds": len(folds),
        "folds": folds,
        "aggregate": {
            metric: {
                "mean": float(np.mean([values[metric] for values in folds.values()])),
                "std": float(np.std([values[metric] for values in folds.values()])),
            }
            for metric in metric_names
        },
    }
    output = args.output or args.root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
