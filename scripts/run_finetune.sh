#!/usr/bin/env bash
# RF-DETR v2 — single-GPU fine-tuning (plain Python, no torchrun).
#
# Usage:
#   ./scripts/run_finetune.sh --dataset-dir /data/custom --output-dir ./out
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_finetune.sh --dataset-dir /data/custom --output-dir ./out
#
# Env (optional):
#   CUDA_VISIBLE_DEVICES  default: 0 (single GPU). Override as needed.
#   DATASET_DIR, OUTPUT_DIR, COCO_WEIGHTS, PRETRAINED_ENCODER  see scripts/finetune.py
#
# Use bash or ./scripts/run_finetune.sh — not plain ``sh`` on dash-based systems.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 "${SCRIPT_DIR}/finetune.py" "$@"
