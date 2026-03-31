#!/usr/bin/env bash
# Single-process training (one GPU). Override paths via env or pass-through args.
#
# Examples:
#   ./scripts/run_train.sh --dataset-dir /data/coco --output-dir ./out --variant base
#   DATASET_DIR=/data/coco OUTPUT_DIR=./out ./scripts/run_train.sh --variant small

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DS=()
[[ -n "${DATASET_DIR:-}" ]] && DS+=(--dataset-dir "${DATASET_DIR}")
OD=()
[[ -n "${OUTPUT_DIR:-}" ]] && OD+=(--output-dir "${OUTPUT_DIR}")

exec python "${ROOT}/scripts/train.py" "${DS[@]}" "${OD[@]}" "$@"
