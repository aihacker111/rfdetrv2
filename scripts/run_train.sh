#!/usr/bin/env bash
# RF-DETR v2 — single-GPU training from scratch.
#
# Usage:
#   ./scripts/run_train.sh                                          # uses defaults below
#   ./scripts/run_train.sh --model-size nano --epochs 50           # override any flag
#   DATASET_DIR=/data/mydata OUTPUT_DIR=./out ./scripts/run_train.sh
#
# All variables below can be overridden via environment or CLI flags.
# For multi-GPU, replace "python3" with "torchrun --nproc_per_node=N".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── GPU ───────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_DIR="${DATASET_DIR:-/path/to/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-output/train}"
DATASET_FILE="${DATASET_FILE:-roboflow}"      # roboflow | coco | o365

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_SIZE="${MODEL_SIZE:-base}"              # nano | small | base | large
NUM_CLASSES="${NUM_CLASSES:-80}"

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"                 # per-GPU
GRAD_ACCUM="${GRAD_ACCUM:-4}"                 # effective = batch_size × grad_accum
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"

# ── Optimizer ─────────────────────────────────────────────────────────────────
LR="${LR:-3e-4}"
LR_ENCODER="${LR_ENCODER:-6e-5}"
LR_VIT_LAYER_DECAY="${LR_VIT_LAYER_DECAY:-0.8}"
LR_COMPONENT_DECAY="${LR_COMPONENT_DECAY:-0.7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
LR_DROP="${LR_DROP:-100}"
EMA_DECAY="${EMA_DECAY:-0.993}"

# ── Loss ──────────────────────────────────────────────────────────────────────
CLS_LOSS_COEF="${CLS_LOSS_COEF:-1.0}"
BBOX_LOSS_COEF="${BBOX_LOSS_COEF:-5.0}"
GIOU_LOSS_COEF="${GIOU_LOSS_COEF:-2.0}"

# ── Prototype alignment ───────────────────────────────────────────────────────
PROTO_LOSS_COEF="${PROTO_LOSS_COEF:-0.1}"
PROTO_MOMENTUM="${PROTO_MOMENTUM:-0.999}"
PROTO_WARMUP="${PROTO_WARMUP:-200}"
PROTO_TEMP="${PROTO_TEMP:-0.1}"
PROTO_ORTHO="${PROTO_ORTHO:-0.1}"
PROTO_DISAMBIG="${PROTO_DISAMBIG:-0.1}"
PROTO_SPARSE="${PROTO_SPARSE:-0.05}"
PROTO_IOU="${PROTO_IOU:-0.3}"

# ──────────────────────────────────────────────────────────────────────────────
exec python3 "${SCRIPT_DIR}/train.py" \
    --dataset-dir             "${DATASET_DIR}" \
    --output-dir              "${OUTPUT_DIR}" \
    --dataset-file            "${DATASET_FILE}" \
    --model-size              "${MODEL_SIZE}" \
    --num-classes             "${NUM_CLASSES}" \
    --epochs                  "${EPOCHS}" \
    --batch-size              "${BATCH_SIZE}" \
    --grad-accum-steps        "${GRAD_ACCUM}" \
    --num-workers             "${NUM_WORKERS}" \
    --checkpoint-interval     "${CHECKPOINT_INTERVAL}" \
    --amp \
    --use-ema \
    --use-varifocal-loss \
    --lr                      "${LR}" \
    --lr-encoder              "${LR_ENCODER}" \
    --lr-scale-mode           sqrt \
    --lr-vit-layer-decay      "${LR_VIT_LAYER_DECAY}" \
    --lr-component-decay      "${LR_COMPONENT_DECAY}" \
    --weight-decay            "${WEIGHT_DECAY}" \
    --warmup-epochs           "${WARMUP_EPOCHS}" \
    --lr-drop                 "${LR_DROP}" \
    --ema-decay               "${EMA_DECAY}" \
    --cls-loss-coef           "${CLS_LOSS_COEF}" \
    --bbox-loss-coef          "${BBOX_LOSS_COEF}" \
    --giou-loss-coef          "${GIOU_LOSS_COEF}" \
    --prototype-loss-coef     "${PROTO_LOSS_COEF}" \
    --prototype-momentum      "${PROTO_MOMENTUM}" \
    --prototype-warmup-steps  "${PROTO_WARMUP}" \
    --prototype-temperature   "${PROTO_TEMP}" \
    --prototype-ortho-coef    "${PROTO_ORTHO}" \
    --prototype-disambig-coef "${PROTO_DISAMBIG}" \
    --prototype-sparse-coef   "${PROTO_SPARSE}" \
    --prototype-iou-threshold "${PROTO_IOU}" \
    "$@"
