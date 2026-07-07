#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAIRSEQ_ROOT="${FAIRSEQ_ROOT:?Set FAIRSEQ_ROOT to the Fairseq source directory}"
SER_DATA_ROOT="${SER_DATA_ROOT:-/root/songhao/datasets/ser_datasets}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/songhao/datasets/checkpoints}"
DATASET="${1:-ravdess}"
FOLD="${2:-fold_01}"
PRETRAINED="${3:-${CHECKPOINT_ROOT}/EAT-base_epoch30_pt.pt}"
OUTPUT_DIR="${4:-${CHECKPOINT_ROOT}/ser/${DATASET}/${FOLD}}"
DATA_DIR="${SER_DATA_ROOT}/manifests/${DATASET}/${FOLD}"

cd "${EAT_ROOT}"

case "${DATASET}" in
  ravdess) NUM_CLASSES=8 ;;
  emodb) NUM_CLASSES=7 ;;
  iemocap) NUM_CLASSES=4 ;;
  *) echo "Unsupported SER dataset: ${DATASET}" >&2; exit 2 ;;
esac

test -f "${DATA_DIR}/train.tsv"
test -f "${DATA_DIR}/eval.tsv"
test -f "${PRETRAINED}"
mkdir -p "${OUTPUT_DIR}"

python "${FAIRSEQ_ROOT}/fairseq_cli/hydra_train.py" -m \
  --config-dir "${EAT_ROOT}/config" \
  --config-name finetuning_SER \
  common.user_dir="${EAT_ROOT}" \
  task.data="${DATA_DIR}" \
  model.model_path="${PRETRAINED}" \
  model.num_classes="${NUM_CLASSES}" \
  checkpoint.save_dir="${OUTPUT_DIR}"
