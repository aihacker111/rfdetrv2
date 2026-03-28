#!/bin/bash
# RF-DETR Supervised Training — Multi-GPU
# Supervised detection training via train_supervised.py

set -e

export CUDA_VISIBLE_DEVICES=0,1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${DATASET_DIR:-/workspace/coco_2017}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/coco_2017/output/rfdetrv2_nano_supervised}"
TRAIN_PY="${SCRIPT_DIR}/train_supervised.py"

NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"

echo "Checking GPU availability..."
nvidia-smi || echo "Warning: nvidia-smi not found."
NUM_AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs, but only $NUM_AVAILABLE_GPUS available."
    NUM_GPUS=$NUM_AVAILABLE_GPUS
fi
[ "$NUM_GPUS" -lt 1 ] && { echo "Error: No GPUs available."; exit 1; }

echo "Starting RF-DETR supervised training on $NUM_GPUS GPUs..."
echo "Dataset: $DATASET_DIR"
echo "Output:  $OUTPUT_DIR"
# LR: lr=2e-4, lr_encoder=2.5e-5 (effective ~5.66e-4 / ~7.07e-5 sau sqrt 8 GPU)

torchrun --standalone --nproc_per_node=$NUM_GPUS --master_port="${MASTER_PORT:-29500}" \
    "$TRAIN_PY" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE_PER_GPU" \
    --num-workers 16 \
    --epochs 50 \
    --model-size nano \
    --use-varifocal-loss \
    --tensorboard \
    --use-rsa \
    --sra-G 32 \
    --sra-heads 8 \
    --freeze-encoder
    # use_convnext_projector: bật mặc định
    # --sra-per-scale \   # mỗi mức feature một SRA (nhiều param hơn)
    # --use-windowed-attn \   # xung khắc với RSA
echo "Training completed!"
