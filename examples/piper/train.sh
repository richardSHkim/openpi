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
PIPER_REPO_ID="$(basename "$(dirname "${PIPER_DATASET_DIR}")")/$(basename "${PIPER_DATASET_DIR}")"
PIPER_CONFIG_NAME="${PIPER_CONFIG_NAME:-pi05_piper}"
PIPER_ASSETS_BASE_DIR="${PIPER_ASSETS_BASE_DIR:-./assets}"
PIPER_NORM_STATS_PATH="${PIPER_ASSETS_BASE_DIR}/${PIPER_CONFIG_NAME}/${PIPER_REPO_ID}/norm_stats.json"

cd "${OPENPI_ROOT}"

if [[ ! -f "${PIPER_DATASET_DIR}/meta/info.json" ]]; then
    echo "Expected staged Piper dataset at ${PIPER_DATASET_DIR}." >&2
    echo "Set PIPER_STAGED_DATASET to a valid staged dataset directory." >&2
    exit 1
fi

if [[ ! -f "${PIPER_NORM_STATS_PATH}" ]]; then
    echo "Expected Piper norm stats at ${PIPER_NORM_STATS_PATH}." >&2
    echo "Run bash examples/piper/compute_norm_stats.sh before training." >&2
    exit 1
fi

exec env \
    HF_LEROBOT_HOME="${PIPER_LEROBOT_HOME}" \
    WANDB_MODE="${WANDB_MODE:-online}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py \
    "${PIPER_CONFIG_NAME}" \
    --exp-name "${PIPER_EXP_NAME:-pi05_piper_$(date -u +%Y%m%d_%H%M%S)}" \
    --fsdp-devices "${PIPER_FSDP_DEVICES:-1}" \
    --data.repo-id "${PIPER_REPO_ID}" \
    --assets-base-dir "${PIPER_ASSETS_BASE_DIR}" \
    "$@"
