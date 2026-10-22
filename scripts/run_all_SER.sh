#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SER_DATA_ROOT="${SER_DATA_ROOT:-/root/songhao/datasets/SER_DATASET}"
SER_OUTPUT_ROOT="${SER_OUTPUT_ROOT:-/root/songhao/experiments/EAT2EMO}"
if [[ -n "${SER_CONFIG_NAME+x}" ]]; then
  CONFIG_NAME_EXPLICIT=true
else
  CONFIG_NAME_EXPLICIT=false
fi
SER_CONFIG_NAME="${SER_CONFIG_NAME:-eat_base_sr16k_audiosetnorm_bs32}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASETS="emodb,iemocap,ravdess-speech,ravdess-song"
DEFAULT_SEEDS="3949,42,2026,1024,666"
SEEDS="all"
FOLDS="all"
NORM_MODE="audioset"
MAX_EPOCHS=""
NUM_WORKERS="${SER_NUM_WORKERS:-6}"
WANDB_PROJECT="${SER_WANDB_PROJECT:-}"
DRY_RUN=false

usage() {
  echo "Usage: $0 [--datasets LIST] [--seeds LIST|all] [--folds LIST|all] [--norm-mode audioset|fold] [--max-epochs N] [--num-workers N] [--wandb-project NAME] [--data-root PATH] [--output-root PATH] [--config-name NAME] [--dry-run]"
  echo "Defaults: all four datasets; seeds ${DEFAULT_SEEDS}; folds all; dataset-specific max_epochs."
  echo "Every fold runs the earlystop protocol and is strictly summarized per dataset/config."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --folds) FOLDS="${2:-}"; shift 2 ;;
    --norm-mode) NORM_MODE="${2:-}"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="${2:-}"; shift 2 ;;
    --num-workers) NUM_WORKERS="${2:-}"; shift 2 ;;
    --wandb-project) WANDB_PROJECT="${2:-}"; shift 2 ;;
    --data-root) SER_DATA_ROOT="${2:-}"; shift 2 ;;
    --output-root) SER_OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --config-name) SER_CONFIG_NAME="${2:-}"; CONFIG_NAME_EXPLICIT=true; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$NORM_MODE" == audioset || "$NORM_MODE" == fold ]] || { echo "Invalid norm mode: $NORM_MODE" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "Invalid num workers: $NUM_WORKERS" >&2; exit 2; }
if [[ -n "$MAX_EPOCHS" && ! "$MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-epochs must be positive" >&2
  exit 2
fi
[[ -n "$SER_DATA_ROOT" && -n "$SER_OUTPUT_ROOT" ]] || { echo "Data/output roots cannot be empty" >&2; exit 2; }
[[ -n "$SEEDS" && -n "$FOLDS" ]] || { echo "--seeds/--folds cannot be empty" >&2; exit 2; }

if [[ "$SEEDS" == "all" ]]; then
  SEEDS_CSV="$DEFAULT_SEEDS"
else
  SEEDS_CSV="$SEEDS"
fi

if [[ "$FOLDS" == "all" ]]; then
  FOLDS_ALL=true
  requested_folds=()
else
  FOLDS_ALL=false
  IFS=',' read -r -a requested_folds <<< "$FOLDS"
  [[ ${#requested_folds[@]} -gt 0 ]] || { echo "--folds list cannot be empty" >&2; exit 2; }
  seen_folds=","
  for fold in "${requested_folds[@]}"; do
    [[ "$fold" =~ ^[0-9]+$ ]] || { echo "Invalid fold: $fold" >&2; exit 2; }
    case "$seen_folds" in
      *",${fold},"*) echo "Duplicate fold: $fold" >&2; exit 2 ;;
    esac
    seen_folds="${seen_folds}${fold},"
  done
fi

IFS=',' read -r -a dataset_list <<< "$DATASETS"
IFS=',' read -r -a seed_list <<< "$SEEDS_CSV"
[[ ${#dataset_list[@]} -gt 0 && ${#seed_list[@]} -gt 0 ]] || { echo "Dataset and seed lists cannot be empty" >&2; exit 2; }
[[ "$SER_CONFIG_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid config name: $SER_CONFIG_NAME" >&2; exit 2; }

seen_datasets=","
for dataset in "${dataset_list[@]}"; do
  [[ -n "$dataset" ]] || { echo "Dataset list contains an empty value" >&2; exit 2; }
  case "$dataset" in
    emodb|iemocap|ravdess-speech|ravdess-song) ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
  esac
  case "$seen_datasets" in
    *",${dataset},"*) echo "Duplicate dataset: $dataset" >&2; exit 2 ;;
  esac
  seen_datasets="${seen_datasets}${dataset},"
done

seen_seeds=","
for seed in "${seed_list[@]}"; do
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $seed" >&2; exit 2; }
  case "$seen_seeds" in
    *",${seed},"*) echo "Duplicate seed: $seed" >&2; exit 2 ;;
  esac
  seen_seeds="${seen_seeds}${seed},"
done

# Validate explicit folds against every selected dataset before launching anything.
if ! $FOLDS_ALL; then
  for dataset in "${dataset_list[@]}"; do
    case "$dataset" in
      emodb) first_fold=1; last_fold=10 ;;
      iemocap) first_fold=1; last_fold=5 ;;
      ravdess-speech|ravdess-song) first_fold=0; last_fold=4 ;;
    esac
    for fold in "${requested_folds[@]}"; do
      if (( fold < first_fold || fold > last_fold )); then
        echo "Fold $fold is invalid for $dataset (allowed $first_fold..$last_fold)" >&2
        exit 2
      fi
    done
  done
fi

for dataset in "${dataset_list[@]}"; do
  case "$dataset" in
    emodb) first_fold=1; last_fold=10; default_max_epochs=80 ;;
    iemocap) first_fold=1; last_fold=5; default_max_epochs=20 ;;
    ravdess-speech|ravdess-song) first_fold=0; last_fold=4; default_max_epochs=80 ;;
  esac
  fold_list=()
  if $FOLDS_ALL; then
    for ((fold=first_fold; fold<=last_fold; fold++)); do
      fold_list+=("$fold")
    done
  else
    fold_list=("${requested_folds[@]}")
  fi

  effective_config="$SER_CONFIG_NAME"
  if ! $CONFIG_NAME_EXPLICIT; then
    if [[ "$NORM_MODE" == fold ]]; then
      effective_config="${effective_config}_foldnorm"
    fi
    if [[ -n "$MAX_EPOCHS" && "$MAX_EPOCHS" != "$default_max_epochs" ]]; then
      effective_config="${effective_config}_maxep${MAX_EPOCHS}"
    fi
  fi

  for seed in "${seed_list[@]}"; do
    for fold in "${fold_list[@]}"; do
      command=(
        "${EAT_ROOT}/scripts/finetuning_SER.sh"
        --dataset "$dataset"
        --fold "$fold"
        --seed "$seed"
        --norm-mode "$NORM_MODE"
        --num-workers "$NUM_WORKERS"
        --data-root "$SER_DATA_ROOT"
        --output-root "$SER_OUTPUT_ROOT"
      )
      if $CONFIG_NAME_EXPLICIT; then
        command+=(--config-name "$SER_CONFIG_NAME")
      fi
      if [[ -n "$MAX_EPOCHS" ]]; then
        command+=(--max-epochs "$MAX_EPOCHS")
      fi
      if [[ -n "$WANDB_PROJECT" ]]; then
        command+=(--wandb-project "$WANDB_PROJECT")
      fi
      if $DRY_RUN; then
        command+=(--dry-run)
      fi
      "${command[@]}"
    done
  done

  if ! $FOLDS_ALL; then
    echo "Partial folds selected (--folds ${FOLDS}); skipping automatic summary for ${dataset}." >&2
  else
    summary_root="${SER_OUTPUT_ROOT}/${dataset}/earlystop/${effective_config}"
    summary_command=(
      "$PYTHON_BIN" "${EAT_ROOT}/evaluation/summarize_ser.py"
      "$summary_root"
      --dataset "$dataset"
      --config "$effective_config"
      --expected-seeds "$SEEDS_CSV"
      --output "${summary_root}/summary.json"
    )
    if $DRY_RUN; then
      printf 'SUMMARY:'
      printf ' %q' "${summary_command[@]}"
      printf '\n'
    else
      "${summary_command[@]}"
    fi
  fi
done
