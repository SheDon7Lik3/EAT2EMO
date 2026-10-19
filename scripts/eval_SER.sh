#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SER_DATA_ROOT="${SER_DATA_ROOT:-/root/songhao/datasets/SER_DATASET}"
SER_OUTPUT_ROOT="${SER_OUTPUT_ROOT:-/root/songhao/experiments/EAT2EMO}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${SER_CONFIG_NAME+x}" ]]; then
  config_explicit="true"
else
  config_explicit="false"
fi

usage() {
  cat <<'EOF'
Usage: scripts/eval_SER.sh --dataset DATASET --fold N --seed N [options]

Required:
  --dataset emodb|iemocap|ravdess-speech|ravdess-song
  --fold N
  --seed N

Options:
  --protocol nested|fixed       Default: nested
  --config NAME                 Default: eat_base_sr16k_audiosetnorm_bs32
  --fixed-epochs N              Default: 60 (fixed only)
  --selected-epoch N            Record nested selection result
  --checkpoint PATH             Default: computed run checkpoint_final.pt
  --output PATH                 Default: computed run stage/results
  --data PATH                   Default: SER_DATA_ROOT/manifests/.../fold_XX
  --audio-root PATH             Override the root line in the TSV manifest
  --split NAME                  Default: valid
  --target-length N             Otherwise uses the dataset default
  --batch-size N                Default: 32
  --num-workers N               Default: 4
  --device DEVICE               Default: CUDA when available
  --overwrite                   Explicitly replace existing evaluation artifacts
EOF
}

dataset=""
fold=""
seed=""
protocol="nested"
config="${SER_CONFIG_NAME:-eat_base_sr16k_audiosetnorm_bs32}"
fixed_epochs="60"
selected_epoch=""
checkpoint=""
output=""
data=""
audio_root=""
split="valid"
target_length=""
batch_size="32"
num_workers="4"
device=""
overwrite="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) dataset="${2:-}"; shift 2 ;;
    --fold) fold="${2:-}"; shift 2 ;;
    --seed) seed="${2:-}"; shift 2 ;;
    --protocol) protocol="${2:-}"; shift 2 ;;
    --config) config="${2:-}"; config_explicit="true"; shift 2 ;;
    --fixed-epochs) fixed_epochs="${2:-}"; shift 2 ;;
    --selected-epoch) selected_epoch="${2:-}"; shift 2 ;;
    --checkpoint) checkpoint="${2:-}"; shift 2 ;;
    --output) output="${2:-}"; shift 2 ;;
    --data) data="${2:-}"; shift 2 ;;
    --audio-root) audio_root="${2:-}"; shift 2 ;;
    --split) split="${2:-}"; shift 2 ;;
    --target-length) target_length="${2:-}"; shift 2 ;;
    --batch-size) batch_size="${2:-}"; shift 2 ;;
    --num-workers) num_workers="${2:-}"; shift 2 ;;
    --device) device="${2:-}"; shift 2 ;;
    --overwrite) overwrite="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$dataset" ]] || { echo "Error: --dataset is required." >&2; exit 2; }
[[ "$fold" =~ ^[0-9]+$ ]] || { echo "Error: --fold must be a non-negative integer." >&2; exit 2; }
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "Error: --seed must be a non-negative integer." >&2; exit 2; }
[[ "$fixed_epochs" =~ ^[1-9][0-9]*$ ]] || { echo "Error: --fixed-epochs must be positive." >&2; exit 2; }
[[ "$config" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Error: invalid config name: $config" >&2; exit 2; }

case "$dataset" in
  emodb|EMODB|emo-db|EMO-DB) dataset="emodb" ;;
  iemocap|IEMOCAP|iemocap-5531|IEMOCAP-5531) dataset="iemocap" ;;
  ravdess|RAVDESS|ravdess-speech|RAVDESS-Speech) dataset="ravdess-speech" ;;
  ravdess-song|RAVDESS-Song) dataset="ravdess-song" ;;
  *) echo "Error: unsupported dataset: ${dataset}" >&2; exit 2 ;;
esac
case "$protocol" in
  nested) stage="final" ;;
  fixed)
    stage="fixed"
    if [[ "$config_explicit" == false && "$fixed_epochs" != 60 ]]; then
      config="${config}_fixedep${fixed_epochs}"
    fi
    ;;
  *) echo "Error: --protocol must be nested or fixed." >&2; exit 2 ;;
esac
if [[ -n "$selected_epoch" && ! "$selected_epoch" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --selected-epoch must be positive." >&2
  exit 2
fi

printf -v fold_name 'fold_%02d' "$((10#$fold))"
run_dir="${SER_OUTPUT_ROOT}/${dataset}/${protocol}/${config}/seed_${seed}/${fold_name}/${stage}"
data="${data:-${SER_DATA_ROOT}/manifests/${dataset}/${fold_name}}"
checkpoint="${checkpoint:-${run_dir}/checkpoint_final.pt}"
output="${output:-${run_dir}/results}"

[[ -f "${data}/${split}.tsv" ]] || { echo "Missing manifest: ${data}/${split}.tsv" >&2; exit 1; }
[[ -f "${data}/${split}.lbl" ]] || { echo "Missing labels: ${data}/${split}.lbl" >&2; exit 1; }
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 1; }

command=(
  "$PYTHON_BIN" "${EAT_ROOT}/evaluation/eval_ser.py"
  --data "$data"
  --checkpoint "$checkpoint"
  --model-dir "$EAT_ROOT"
  --output "$output"
  --dataset "$dataset"
  --protocol "$protocol"
  --fold "$fold"
  --seed "$seed"
  --stage "$stage"
  --fixed-epochs "$fixed_epochs"
  --config "$config"
  --split "$split"
  --batch-size "$batch_size"
  --num-workers "$num_workers"
)
[[ -z "$selected_epoch" ]] || command+=(--selected-epoch "$selected_epoch")
[[ -z "$audio_root" ]] || command+=(--audio-root "$audio_root")
[[ -z "$target_length" ]] || command+=(--target-length "$target_length")
[[ -z "$device" ]] || command+=(--device "$device")
[[ "$overwrite" == "false" ]] || command+=(--overwrite)

cd "$EAT_ROOT"
"${command[@]}"
