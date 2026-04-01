#!/usr/bin/env bash
# RF-DETR v2 — single-GPU supervised training (plain Python, no torchrun).
#
# Usage:
#   ./scripts/run_train.sh --dataset-dir /data/COCO --output-dir ./out
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_train.sh --dataset-dir /data/COCO --output-dir ./out
#
# Env (optional):
#   CUDA_VISIBLE_DEVICES  default: all visible GPUs; set to e.g. 0 to pin one GPU
#   DATASET_DIR, OUTPUT_DIR, PRETRAINED_ENCODER  see scripts/train.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

exec python3 "${SCRIPT_DIR}/train.py" "$@"
