#!/usr/bin/env bash
# RF-DETR v2 — multi-GPU supervised training via torchrun.
#
# Usage:
#   ./scripts/run_train_multigpu.sh --dataset-dir /data/COCO --output-dir ./out
#   NPROC=8 ./scripts/run_train_multigpu.sh --dataset-dir /data/COCO --output-dir ./out
#
# Env (optional):
#   NPROC              GPUs per node (default: 4)
#   CUDA_VISIBLE_DEVICES  e.g. 0,1,2,3
#   MASTER_ADDR        multi-node master (default: 127.0.0.1)
#   MASTER_PORT        free TCP port (default: 29500)
#   DATASET_DIR, OUTPUT_DIR  forwarded to train.py if set in environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/train.py" "$@"
