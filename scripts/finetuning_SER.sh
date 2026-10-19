#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-python}" "${EAT_ROOT}/tools/ser_protocol.py" "$@"
