#!/usr/bin/env bash
# Single-GPU fine-tuning (no torchrun).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DS=()
[[ -n "${DATASET_DIR:-}" ]] && DS+=(--dataset-dir "${DATASET_DIR}")
OD=()
[[ -n "${OUTPUT_DIR:-}" ]] && OD+=(--output-dir "${OUTPUT_DIR}")

exec python "${ROOT}/scripts/finetune.py" "${DS[@]}" "${OD[@]}" "$@"
