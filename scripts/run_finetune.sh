#!/usr/bin/env bash
# RF-DETR v2 — single-GPU fine-tuning (plain Python, no torchrun).
#
# Usage:
#   ./scripts/run_finetune.sh --dataset-dir /data/custom --output-dir ./out
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_finetune.sh --dataset-dir /data/custom --output-dir ./out
#
# Env (optional):
#   CUDA_VISIBLE_DEVICES  e.g. 0 to use one GPU
#   DATASET_DIR, OUTPUT_DIR, COCO_WEIGHTS, PRETRAINED_ENCODER  see scripts/finetune.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

exec python3 "${SCRIPT_DIR}/finetune.py" "$@"
