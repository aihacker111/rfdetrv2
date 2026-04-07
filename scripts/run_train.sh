#!/usr/bin/env bash
# =============================================================================
# RF-DETR v2 — Train from Scratch (Single-GPU)
#
# ⚠  Đây là script TRAIN FROM SCRATCH:
#    - Detection head được khởi tạo ngẫu nhiên (KHÔNG tải RF-DETR COCO weights)
#    - Chỉ tải DINOv3 backbone encoder (tự động download HuggingFace lần đầu)
#
# → Muốn finetune từ checkpoint COCO?  Dùng scripts/run_finetune.sh
#
# Chỉnh các biến bên dưới rồi chạy:  bash scripts/run_train.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# =============================================================================
# PATHS — BẮT BUỘC chỉnh DATASET_DIR
# =============================================================================
DATASET_DIR="/workspace/coco2017"          # đường dẫn tới dataset (COCO layout)
OUTPUT_DIR="output/train"         # thư mục lưu checkpoint và log
DATASET_FILE="coco"               # coco | roboflow | o365
PRETRAINED_ENCODER=""             # đường dẫn .pth DINOv3 backbone (để trống = tự download HuggingFace)
# PRETRAIN_WEIGHTS không có ở đây — train from scratch không dùng RF-DETR COCO weights

# =============================================================================
# MODEL
# =============================================================================
MODEL_SIZE="nano"                 # nano | small | base | large
USE_WINDOWED_ATTN=false           # true = bật window attention (tiết kiệm VRAM)
FREEZE_ENCODER=false              # true = đóng băng DINOv3 backbone

# =============================================================================
# CPFE — Cortical Perceptual Feature Enhancement
# =============================================================================
USE_CPFE=true                     # master toggle — false = tắt toàn bộ CPFE
CPFE_USE_SDG=true                 # Spectral Decomposition Gate (Center-Surround)
CPFE_USE_DN=true                  # Divisive Normalization (Lateral Inhibition)
CPFE_USE_TPR=true                 # Top-Down Predictive Refinement (Cortical Feedback)

# =============================================================================
# TRAINING RUN
# =============================================================================
EPOCHS=50
BATCH_SIZE=64
GRAD_ACCUM_STEPS=4                # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS
NUM_WORKERS=8
AMP=true                          # true = FP16 mixed precision
TENSORBOARD=true
DEVICE="cuda"                     # cuda | cpu | mps
DEBUG_DATA_LIMIT=0                # 0 = full dataset; N > 0 = chỉ dùng N ảnh (smoke test)

# =============================================================================
# OPTIMIZER / LEARNING RATE SCHEDULE
# =============================================================================
LR=2e-4
LR_ENCODER=2.5e-5                 # LR riêng cho encoder (thường nhỏ hơn)
LR_SCALE_MODE="sqrt"              # linear | sqrt
WARMUP_EPOCHS=1
LR_SCHEDULER="cosine_restart"     # cosine_restart | cosine | multistep | linear | wsd
LR_RESTART_PERIOD=25              # số epoch mỗi chu kỳ cosine restart
LR_RESTART_DECAY=0.8              # hệ số giảm LR mỗi restart
LR_MIN_FACTOR=0.05                # LR tối thiểu = LR × LR_MIN_FACTOR

# =============================================================================
# LOSS
# =============================================================================
USE_VARIFOCAL_LOSS=false          # true = dùng varifocal loss thay focal loss
CLS_LOSS_COEF=1.0
BBOX_LOSS_COEF=5.0
GIOU_LOSS_COEF=2.0

# =============================================================================
# PROTOTYPE ALIGNMENT (CPFE memory)
# =============================================================================
USE_PROTOTYPE_ALIGN=true
PROTOTYPE_LOSS_COEF=0.1
PROTOTYPE_MOMENTUM=0.999
PROTOTYPE_WARMUP_STEPS=200
PROTOTYPE_TEMPERATURE=0.1
PROTOTYPE_REPULSION_COEF=0.1
PROTOTYPE_USE_FREQ_WEIGHT=true
PROTOTYPE_USE_QUALITY_WEIGHT=true
PROTOTYPE_USE_REPULSION=true

# =============================================================================
# GPU
# =============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# =============================================================================
# BUILD ARGS
# =============================================================================
ARGS=(
    --dataset-dir        "${DATASET_DIR}"
    --output-dir         "${OUTPUT_DIR}"
    --dataset-file       "${DATASET_FILE}"
    --model-size         "${MODEL_SIZE}"
    --epochs             "${EPOCHS}"
    --batch-size         "${BATCH_SIZE}"
    --grad-accum-steps   "${GRAD_ACCUM_STEPS}"
    --num-workers        "${NUM_WORKERS}"
    --device             "${DEVICE}"
    --lr                 "${LR}"
    --lr-encoder         "${LR_ENCODER}"
    --lr-scale-mode      "${LR_SCALE_MODE}"
    --lr-restart-period  "${LR_RESTART_PERIOD}"
    --lr-restart-decay   "${LR_RESTART_DECAY}"
    --lr-min-factor      "${LR_MIN_FACTOR}"
    --cls-loss-coef      "${CLS_LOSS_COEF}"
    --bbox-loss-coef     "${BBOX_LOSS_COEF}"
    --giou-loss-coef     "${GIOU_LOSS_COEF}"
    --prototype-loss-coef       "${PROTOTYPE_LOSS_COEF}"
    --prototype-momentum        "${PROTOTYPE_MOMENTUM}"
    --prototype-warmup-steps    "${PROTOTYPE_WARMUP_STEPS}"
    --prototype-temperature     "${PROTOTYPE_TEMPERATURE}"
    --prototype-repulsion-coef  "${PROTOTYPE_REPULSION_COEF}"
)

# DINOv3 backbone weights (tuỳ chọn — để trống = auto-download)
[[ -n "${PRETRAINED_ENCODER}" ]] && ARGS+=(--pretrained-encoder "${PRETRAINED_ENCODER}")

# Flags
[[ "${AMP}"                         == "true" ]] && ARGS+=(--amp)
[[ "${TENSORBOARD}"                 == "true" ]] && ARGS+=(--tensorboard)         || ARGS+=(--no-tensorboard)
[[ "${USE_WINDOWED_ATTN}"           == "true" ]] && ARGS+=(--use-windowed-attn)
[[ "${FREEZE_ENCODER}"              == "true" ]] && ARGS+=(--freeze-encoder)
[[ "${USE_VARIFOCAL_LOSS}"          == "true"  ]] && ARGS+=(--use-varifocal-loss)
[[ "${USE_PROTOTYPE_ALIGN}"         == "false" ]] && ARGS+=(--no-prototype-align)
[[ "${USE_CPFE}"                    == "false" ]] && ARGS+=(--no-cpfe)
[[ "${CPFE_USE_SDG}"                == "false" ]] && ARGS+=(--no-cpfe-sdg)
[[ "${CPFE_USE_DN}"                 == "false" ]] && ARGS+=(--no-cpfe-dn)
[[ "${CPFE_USE_TPR}"                == "false" ]] && ARGS+=(--no-cpfe-tpr)
[[ "${PROTOTYPE_USE_FREQ_WEIGHT}"   == "false" ]] && ARGS+=(--no-prototype-use-freq-weight)
[[ "${PROTOTYPE_USE_QUALITY_WEIGHT}" == "false" ]] && ARGS+=(--no-prototype-use-quality-weight)
[[ "${PROTOTYPE_USE_REPULSION}"     == "false" ]] && ARGS+=(--no-prototype-use-repulsion)
[[ "${DEBUG_DATA_LIMIT}" -gt 0 ]]                && ARGS+=(--debug-data-limit "${DEBUG_DATA_LIMIT}")

# =============================================================================
# RUN
# =============================================================================
echo "============================================================"
echo "  RF-DETR v2 Training (from scratch)"
echo "  Model   : ${MODEL_SIZE}  |  Epochs: ${EPOCHS}  |  BS: ${BATCH_SIZE}×${GRAD_ACCUM_STEPS}  |  CPFE: ${USE_CPFE}"
echo "  Dataset : ${DATASET_DIR}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  GPU     : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

exec python3 "${SCRIPT_DIR}/train.py" "${ARGS[@]}"
