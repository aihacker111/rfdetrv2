#!/usr/bin/env bash
# RF-DETR v2 — multi-GPU fine-tuning via torchrun.
#
# Usage:
#   ./scripts/run_finetune_multigpu.sh --dataset-dir /data/custom --output-dir ./out
#   NPROC=2 ./scripts/run_finetune_multigpu.sh --dataset-dir /data/custom --freeze-encoder --unfreeze-at-epoch 5
#   CUDA_VISIBLE_DEVICES=3 NPROC=1 ./scripts/run_finetune_multigpu.sh ...   # only physical GPU 3
#
# Env (optional):
#   NPROC              GPUs per node (default: 4); must match how many IDs you pass in CUDA_VISIBLE_DEVICES
#   CUDA_VISIBLE_DEVICES  default: 0,1,...,NPROC-1 if unset; else your list (e.g. 3 or 2,3)
#   MASTER_ADDR        multi-node master (default: 127.0.0.1)
#   MASTER_PORT        free TCP port (default: 29501 to avoid clash with train)
#   DATASET_DIR, OUTPUT_DIR, COCO_WEIGHTS  see scripts/finetune.py
#
# Run with bash (or chmod +x and ./this_script). Do not use plain ``sh`` on
# Ubuntu/Debian: dash lacks ``pipefail`` / ``[[``; if you must, use:
#   bash scripts/run_finetune_multigpu.sh ...

set -euo pipefail

# Use $0 (not BASH_SOURCE): under ``sh`` or bash-as-sh, BASH_SOURCE is empty → wrong paths.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NPROC="${NPROC:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  _vis=""
  for ((i = 0; i < NPROC; i++)); do
    [[ -n "${_vis}" ]] && _vis+=","
    _vis+="${i}"
  done
  export CUDA_VISIBLE_DEVICES="${_vis}"
else
  export CUDA_VISIBLE_DEVICES
fi

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/finetune.py" "$@"
