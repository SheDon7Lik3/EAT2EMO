#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SER_DATA_ROOT="${SER_DATA_ROOT:-/root/songhao/datasets/ser_datasets}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/songhao/datasets/checkpoints}"
DATASET="${1:-ravdess}"
FOLD="${2:-fold_01}"
CHECKPOINT="${3:-${CHECKPOINT_ROOT}/ser/${DATASET}/${FOLD}/checkpoint_best.pt}"
OUTPUT_DIR="${4:-${CHECKPOINT_ROOT}/ser/${DATASET}/${FOLD}/test}"

cd "${EAT_ROOT}"

python "${EAT_ROOT}/evaluation/eval_ser.py" \
  --data "${SER_DATA_ROOT}/manifests/${DATASET}/${FOLD}" \
  --checkpoint "${CHECKPOINT}" \
  --model-dir "${EAT_ROOT}" \
  --output "${OUTPUT_DIR}"
