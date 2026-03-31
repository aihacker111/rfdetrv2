#!/usr/bin/env bash
# Multi-GPU training via torchrun (sets distributed env for Pipeline).
#
# Env:
#   NUM_GPUS       default 2
#   MASTER_PORT    default 29500
#   CUDA_VISIBLE_DEVICES  e.g. 0,1
#   DATASET_DIR, OUTPUT_DIR  optional defaults passed as CLI flags

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

if command -v nvidia-smi >/dev/null 2>&1; then
  N="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${N}" =~ ^[0-9]+$ ]] && (( N < NUM_GPUS )); then
    echo "Warning: using ${N} GPU(s) (requested ${NUM_GPUS})." >&2
    NUM_GPUS="${N}"
  fi
fi
if (( NUM_GPUS < 1 )); then
  echo "Error: need at least one GPU." >&2
  exit 1
fi

DS=()
[[ -n "${DATASET_DIR:-}" ]] && DS+=(--dataset-dir "${DATASET_DIR}")
OD=()
[[ -n "${OUTPUT_DIR:-}" ]] && OD+=(--output-dir "${OUTPUT_DIR}")

exec torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${ROOT}/scripts/train.py" "${DS[@]}" "${OD[@]}" "$@"
