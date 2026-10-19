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
SEEDS="1,2,3,4,5"
PROTOCOL="nested"
STAGE="all"
FIXED_EPOCHS="60"
SELECTION_MAX_EPOCHS=""
DRY_RUN=false

usage() {
  echo "Usage: $0 [--datasets LIST] [--seeds LIST] [--protocol nested|fixed] [--stage all|select|final] [--fixed-epochs N] [--selection-max-epochs N] [--data-root PATH] [--output-root PATH] [--config-name NAME] [--dry-run]"
  echo "Comma-separated defaults: all four datasets; seeds 1,2,3,4,5."
  echo "Completed formal runs are strictly summarized per dataset/protocol/config."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --protocol) PROTOCOL="${2:-}"; shift 2 ;;
    --stage) STAGE="${2:-}"; shift 2 ;;
    --fixed-epochs) FIXED_EPOCHS="${2:-}"; shift 2 ;;
    --selection-max-epochs) SELECTION_MAX_EPOCHS="${2:-}"; shift 2 ;;
    --data-root) SER_DATA_ROOT="${2:-}"; shift 2 ;;
    --output-root) SER_OUTPUT_ROOT="${2:-}"; shift 2 ;;
    --config-name) SER_CONFIG_NAME="${2:-}"; CONFIG_NAME_EXPLICIT=true; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$PROTOCOL" == nested || "$PROTOCOL" == fixed ]] || { echo "Invalid protocol: $PROTOCOL" >&2; exit 2; }
[[ "$STAGE" == all || "$STAGE" == select || "$STAGE" == final ]] || { echo "Invalid stage: $STAGE" >&2; exit 2; }
[[ "$FIXED_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "--fixed-epochs must be positive" >&2; exit 2; }
if [[ -n "$SELECTION_MAX_EPOCHS" && ! "$SELECTION_MAX_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--selection-max-epochs must be positive" >&2
  exit 2
fi
if [[ "$PROTOCOL" == fixed && "$STAGE" != all ]]; then
  echo "--stage applies only to --protocol nested" >&2
  exit 2
fi
if [[ "$PROTOCOL" == fixed && -n "$SELECTION_MAX_EPOCHS" ]]; then
  echo "--selection-max-epochs applies only to --protocol nested" >&2
  exit 2
fi
[[ -n "$SER_DATA_ROOT" && -n "$SER_OUTPUT_ROOT" ]] || { echo "Data/output roots cannot be empty" >&2; exit 2; }

IFS=',' read -r -a dataset_list <<< "$DATASETS"
IFS=',' read -r -a seed_list <<< "$SEEDS"
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

for dataset in "${dataset_list[@]}"; do
  case "$dataset" in
    emodb) first_fold=1; last_fold=10; default_selection_epochs=80 ;;
    iemocap) first_fold=1; last_fold=5; default_selection_epochs=20 ;;
    ravdess-speech|ravdess-song) first_fold=0; last_fold=4; default_selection_epochs=80 ;;
  esac
  effective_config="$SER_CONFIG_NAME"
  if ! $CONFIG_NAME_EXPLICIT; then
    if [[ "$PROTOCOL" == fixed && "$FIXED_EPOCHS" != 60 ]]; then
      effective_config="${effective_config}_fixedep${FIXED_EPOCHS}"
    elif [[
      "$PROTOCOL" == nested
      && -n "$SELECTION_MAX_EPOCHS"
      && "$SELECTION_MAX_EPOCHS" != "$default_selection_epochs"
    ]]; then
      effective_config="${effective_config}_selectmaxep${SELECTION_MAX_EPOCHS}"
    fi
  fi
  for seed in "${seed_list[@]}"; do
    for ((fold=first_fold; fold<=last_fold; fold++)); do
      command=(
        "${EAT_ROOT}/scripts/finetuning_SER.sh"
        --dataset "$dataset"
        --fold "$fold"
        --seed "$seed"
        --protocol "$PROTOCOL"
        --fixed-epochs "$FIXED_EPOCHS"
        --data-root "$SER_DATA_ROOT"
        --output-root "$SER_OUTPUT_ROOT"
      )
      if $CONFIG_NAME_EXPLICIT; then
        command+=(--config-name "$SER_CONFIG_NAME")
      fi
      if [[ "$PROTOCOL" == nested ]]; then
        command+=(--stage "$STAGE")
        if [[ -n "$SELECTION_MAX_EPOCHS" ]]; then
          command+=(--selection-max-epochs "$SELECTION_MAX_EPOCHS")
        fi
      fi
      if $DRY_RUN; then
        command+=(--dry-run)
      fi
      "${command[@]}"
    done
  done

  if [[ "$PROTOCOL" == fixed || "$STAGE" != select ]]; then
    summary_root="${SER_OUTPUT_ROOT}/${dataset}/${PROTOCOL}/${effective_config}"
    summary_command=(
      "$PYTHON_BIN" "${EAT_ROOT}/evaluation/summarize_ser.py"
      "$summary_root"
      --dataset "$dataset"
      --protocol "$PROTOCOL"
      --config "$effective_config"
      --expected-seeds "$SEEDS"
      --output "${summary_root}/summary.json"
    )
    if [[ "$PROTOCOL" == fixed ]]; then
      summary_command+=(--fixed-epochs "$FIXED_EPOCHS")
    fi
    if $DRY_RUN; then
      printf 'SUMMARY:'
      printf ' %q' "${summary_command[@]}"
      printf '\n'
    else
      "${summary_command[@]}"
    fi
  fi
done
