#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${OPENPI_ROOT}/../.." && pwd)"

DEFAULT_STAGED_DATASET="datasets/richardshkim/piper_banana_v2_openpi"
PIPER_STAGED_DATASET="${PIPER_STAGED_DATASET:-${DEFAULT_STAGED_DATASET}}"

if [[ "${PIPER_STAGED_DATASET}" = /* ]]; then
    PIPER_DATASET_DIR_RAW="${PIPER_STAGED_DATASET}"
else
    PIPER_DATASET_DIR_RAW="${PROJECT_ROOT}/${PIPER_STAGED_DATASET}"
fi
PIPER_DATASET_DIR="$(cd "$(dirname "${PIPER_DATASET_DIR_RAW}")" && pwd)/$(basename "${PIPER_DATASET_DIR_RAW}")"
PIPER_LEROBOT_HOME="$(cd "$(dirname "$(dirname "${PIPER_DATASET_DIR}")")" && pwd)"
PIPER_CONFIG_NAME="${PIPER_CONFIG_NAME:-pi05_piper}"

cd "${OPENPI_ROOT}"

if [[ ! -f "${PIPER_DATASET_DIR}/meta/info.json" ]]; then
    echo "Expected staged Piper dataset at ${PIPER_DATASET_DIR}." >&2
    echo "Set PIPER_STAGED_DATASET to a valid staged dataset directory." >&2
    exit 1
fi

if [[ -n "${PIPER_ASSETS_BASE_DIR:-}" ]]; then
    echo "PIPER_ASSETS_BASE_DIR is not supported by compute_norm_stats.sh." >&2
    echo "compute_norm_stats.py writes to the assets directory configured in ${PIPER_CONFIG_NAME}." >&2
    exit 1
fi

PIPER_CONFIG_REPO_ID="$(
    env PIPER_CONFIG_NAME="${PIPER_CONFIG_NAME}" uv run python - <<'PY'
import os
import openpi.training.config as config

cfg = config.get_config(os.environ["PIPER_CONFIG_NAME"])
print(cfg.data.repo_id)
PY
)"
PIPER_CONFIG_DATASET_DIR="${PIPER_LEROBOT_HOME}/${PIPER_CONFIG_REPO_ID}"

if [[ "${PIPER_DATASET_DIR}" != "${PIPER_CONFIG_DATASET_DIR}" ]]; then
    echo "Staged dataset path ${PIPER_DATASET_DIR} does not match config repo_id ${PIPER_CONFIG_REPO_ID}." >&2
    echo "compute_norm_stats.py uses the repo_id defined in ${PIPER_CONFIG_NAME} and no longer supports wrapper overrides." >&2
    exit 1
fi

exec env \
    HF_LEROBOT_HOME="${PIPER_LEROBOT_HOME}" \
    uv run scripts/compute_norm_stats.py \
    --config-name "${PIPER_CONFIG_NAME}" \
    "$@"
