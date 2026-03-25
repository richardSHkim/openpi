#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${OPENPI_ROOT}"

exec env \
    WANDB_MODE="${WANDB_MODE:-disabled}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
    uv run scripts/train.py \
    "${PIPER_CONFIG_NAME:-pi05_piper}" \
    --exp-name "${PIPER_EXP_NAME:-pi05_piper_$(date -u +%Y%m%d_%H%M%S)}" \
    --fsdp-devices "${PIPER_FSDP_DEVICES:-2}" \
    "$@"
