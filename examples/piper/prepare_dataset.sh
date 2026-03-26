#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_DATASET_SRC="../../datasets/richardshkim/piper_banana_v2"

PIPER_DATASET_SRC="${PIPER_DATASET_SRC:-${DEFAULT_DATASET_SRC}}"
if [[ "${PIPER_DATASET_SRC}" = /* ]]; then
    PIPER_DATASET_SRC_ABS="${PIPER_DATASET_SRC}"
else
    PIPER_DATASET_SRC_ABS="${OPENPI_ROOT}/${PIPER_DATASET_SRC}"
fi
PIPER_DATASET_BASE_DIR="$(cd "$(dirname "$(dirname "${PIPER_DATASET_SRC_ABS}")")" && pwd)"
PIPER_REPO_ID="${PIPER_REPO_ID:-richardshkim/piper_banana_v2_openpi}"
PIPER_LEROBOT_HOME="${PIPER_LEROBOT_HOME:-${HF_LEROBOT_HOME:-${PIPER_DATASET_BASE_DIR}}}"

cd "${OPENPI_ROOT}"

exec env \
    HF_LEROBOT_HOME="${PIPER_LEROBOT_HOME}" \
    uv run examples/piper/convert_piper_dataset.py \
    --src "${PIPER_DATASET_SRC}" \
    --repo-id "${PIPER_REPO_ID}" \
    --dst-root "${PIPER_LEROBOT_HOME}" \
    "$@"
