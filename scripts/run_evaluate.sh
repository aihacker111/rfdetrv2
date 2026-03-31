#!/usr/bin/env bash
# Run COCO-style validation metrics (requires --weights).
#
# Example:
#   ./scripts/run_evaluate.sh --weights ./out/checkpoint_best_total.pth --dataset-dir /data/coco --variant base

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DS=()
[[ -n "${DATASET_DIR:-}" ]] && DS+=(--dataset-dir "${DATASET_DIR}")

exec python "${ROOT}/scripts/evaluate.py" "${DS[@]}" "$@"
