#!/usr/bin/env python3
"""Print a compact status view for a running SER experiment.

Usage:
    SER_OUTPUT_ROOT=/root/songhao/experiments/... python tools/ser_status.py
    python tools/ser_status.py --root /root/songhao/experiments/... --watch 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESTAMP = re.compile(r"^\[([^\]]+)\]")


def find_run(root: Path) -> Path | None:
    candidates = sorted(
        root.rglob("run_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest in candidates:
        if (manifest.parent / "train.log").is_file():
            return manifest.parent
    return candidates[0].parent if candidates else None


def read_manifest(run_dir: Path) -> dict:
    path = run_dir / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_epochs(train_log: Path) -> tuple[int, dict | None, float | None]:
    if not train_log.is_file():
        return 0, None, None
    count = 0
    latest = None
    latest_time = None
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[train]" not in line or '"epoch"' not in line:
            continue
        start = line.find("{")
        if start < 0:
            continue
        try:
            record = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if "epoch" not in record:
            continue
        count += 1
        latest = record
        stamp = TIMESTAMP.match(line)
        if stamp:
            try:
                latest_time = datetime.strptime(
                    stamp.group(1), "%Y-%m-%d %H:%M:%S,%f"
                ).timestamp()
            except ValueError:
                latest_time = None
    return count, latest, latest_time


def process_running(run_dir: Path) -> bool:
    needle = str(run_dir)
    for pattern in ("ser_protocol.py", "hydra_train.py"):
        result = subprocess.run(
            ["pgrep", "-af", pattern],
            text=True,
            capture_output=True,
            check=False,
        )
        if any(needle in line for line in result.stdout.splitlines()):
            return True
    return False


def gpu_status() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return "unavailable"
    values = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2:
            values.append(f"{parts[0]}% util, {parts[1]} MiB")
    return "; ".join(values)


def read_metrics(run_dir: Path) -> dict | None:
    path = run_dir / "results" / "metrics.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def render(run_dir: Path | None) -> str:
    if run_dir is None:
        return "No run_manifest.json found under the requested root."

    manifest = read_manifest(run_dir)
    train_log = run_dir / "train.log"
    completed, latest, latest_time = read_epochs(train_log)
    total = int(manifest.get("epochs", 0) or 0)
    running = process_running(run_dir)
    metrics = read_metrics(run_dir)
    lines = [
        f"Run:      {run_dir}",
        f"Dataset:  {manifest.get('dataset')}  fold={manifest.get('fold')}  "
        f"seed={manifest.get('seed')}  stage={manifest.get('stage')}",
        f"Protocol: {manifest.get('protocol')}  norm={manifest.get('norm_mode')}  "
        f"target={manifest.get('target_length')}  workers={manifest.get('num_workers')}",
        f"Epochs:   {completed}/{total}" if total else f"Epochs:   {completed}",
        f"State:    {'RUNNING' if running else 'not running'}",
        f"GPU:      {gpu_status()}",
    ]
    if latest is not None:
        keep = {
            key: latest.get(key)
            for key in ("epoch", "train_loss", "train_num_updates", "train_wps")
            if key in latest
        }
        lines.append(f"Latest:   {json.dumps(keep, ensure_ascii=False)}")
    if latest_time is not None:
        age = max(0.0, time.time() - latest_time)
        lines.append(f"Last log: {age:.0f}s ago")
    if metrics is not None:
        keep = {
            key: metrics.get(key)
            for key in ("accuracy", "uar", "macro_f1", "loss", "norm_mode")
            if key in metrics
        }
        lines.append(f"Metrics:  {json.dumps(keep, ensure_ascii=False)}")
    elif not running and completed == 0:
        lines.append("Hint:     no epoch has finished yet; check the wrapper log for errors.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show SER experiment progress")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("SER_OUTPUT_ROOT", "/root/songhao/experiments/EAT2EMO")),
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0,
        help="refresh every N seconds (0 = once)",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"root does not exist: {root}")

    run_dir = find_run(root)
    if args.watch <= 0:
        print(render(run_dir))
        return
    try:
        while True:
            sys.stdout.write("\033[2J\033[H")
            print(render(run_dir))
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
