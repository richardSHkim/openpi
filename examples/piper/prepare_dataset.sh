#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_DATASET_SRC="../../datasets/richardshkim/piper_banana_v2_v2.1"
if [[ -f "${OPENPI_ROOT}/../../datasets/richardshkim/piper_banana_v2/meta/info.json" ]]; then
    DEFAULT_DATASET_SRC="../../datasets/richardshkim/piper_banana_v2"
fi

cd "${OPENPI_ROOT}"

exec uv run examples/piper/convert_piper_dataset.py \
    --src "${PIPER_DATASET_SRC:-${DEFAULT_DATASET_SRC}}" \
    --repo-id "${PIPER_REPO_ID:-richardshkim/piper_banana_v2}" \
    "$@"
