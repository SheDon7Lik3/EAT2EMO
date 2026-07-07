#!/usr/bin/env bash
set -euo pipefail

EAT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_ROOT="${BACKUP_ROOT:-${EAT_ROOT}/_backups}"
BACKUP_NAME="${BACKUP_NAME:-$(date +%Y%m%d_%H%M%S)}"
BACKUP_DIR="${BACKUP_ROOT}/backup_${BACKUP_NAME}"

cd "${EAT_ROOT}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This directory is not a git repository: ${EAT_ROOT}" >&2
  exit 2
fi

mkdir -p "${BACKUP_DIR}"

echo "Backing up current working tree to: ${BACKUP_DIR}"

# Copy the current tree while skipping git metadata and large generated data.
# Use tar instead of rsync so the script works on minimal server images.
tar -cf - \
  --exclude='./.git' \
  --exclude='./_backups' \
  --exclude='./checkpoints' \
  --exclude='./results' \
  --exclude='./datasets' \
  . | tar -xf - -C "${BACKUP_DIR}"

echo "Current branch: $(git branch --show-current)"
echo "Remote status before pull:"
git status --short

CURRENT_BRANCH="$(git branch --show-current)"
if [[ -z "${CURRENT_BRANCH}" ]]; then
  echo "Detached HEAD detected; aborting pull." >&2
  exit 3
fi

echo "Pulling latest code from origin/${CURRENT_BRANCH}"
git pull --ff-only origin "${CURRENT_BRANCH}"

echo "Done."
echo "Backup saved at: ${BACKUP_DIR}"
