#!/usr/bin/env bash
# RF-DETR v2 — single-GPU supervised training (plain Python, no torchrun).
#
# Usage:
#   ./scripts/run_train.sh --dataset-dir /data/COCO --output-dir ./out
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_train.sh --dataset-dir /data/COCO --output-dir ./out
#
# Env (optional):
#   CUDA_VISIBLE_DEVICES  default: 0 (single GPU). Override: 1 or 2,3 etc.
#   DATASET_DIR, OUTPUT_DIR, PRETRAINED_ENCODER  see scripts/train.py
#
# Use bash or ./scripts/run_train.sh — not plain ``sh`` on dash-based systems.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 "${SCRIPT_DIR}/train.py" "$@"
