#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SER_OUTPUT_ROOT="${SER_OUTPUT_ROOT:-/root/songhao/experiments/EAT2EMO}"
LOG_PATH="${SER_OUTPUT_ROOT}.log"

mkdir -p "$(dirname "$LOG_PATH")"

echo "Starting: ${EAT_ROOT}/scripts/finetuning_SER.sh $*"
echo "Log:      ${LOG_PATH}"
echo

"${EAT_ROOT}/scripts/finetuning_SER.sh" "$@" >"${LOG_PATH}" 2>&1 &
pid=$!

cleanup() {
  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  echo
  echo "Interrupted."
  exit 130
}
trap cleanup INT TERM

while kill -0 "$pid" 2>/dev/null; do
  printf '\033c'
  if [ -d "$SER_OUTPUT_ROOT" ]; then
    "${PYTHON_BIN}" "${EAT_ROOT}/tools/ser_status.py" --root "$SER_OUTPUT_ROOT" || true
  else
    echo "Waiting for run manifest under ${SER_OUTPUT_ROOT} ..."
  fi
  echo
  echo "Raw log: tail -f ${LOG_PATH}"
  sleep 5
done

set +e
wait "$pid"
status=$?
set -e

printf '\033c'
if [ -d "$SER_OUTPUT_ROOT" ]; then
  "${PYTHON_BIN}" "${EAT_ROOT}/tools/ser_status.py" --root "$SER_OUTPUT_ROOT" || true
fi
echo
if [ "$status" -ne 0 ]; then
  echo "Training failed with exit code ${status}; last log lines:"
  tail -40 "$LOG_PATH" || true
fi
exit "$status"
