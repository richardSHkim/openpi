#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${OPENPI_ROOT}"

exec uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config "${PIPER_CONFIG_NAME:-pi05_piper}" \
    --policy.dir "${PIPER_POLICY_DIR:?Set PIPER_POLICY_DIR to a checkpoint directory.}" \
    "$@"
