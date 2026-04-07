#!/bin/sh
# =============================================================================
# RF-DETR v2 — Train from Scratch (Single-GPU)
#
# ⚠  Đây là script TRAIN FROM SCRATCH:
#    - Detection head được khởi tạo ngẫu nhiên (KHÔNG tải RF-DETR COCO weights)
#    - Chỉ tải DINOv3 backbone encoder (tự động download HuggingFace lần đầu)
#
# → Muốn finetune từ checkpoint COCO?  Dùng scripts/run_finetune.sh
#
# Chạy:  sh scripts/run_train.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Boolean env: [ test ] phân biệt hoa thường — False ≠ false. Chuẩn hoá về lowercase.
_sh_lc() {
    printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

# =============================================================================
# PATHS — BẮT BUỘC chỉnh DATASET_DIR
# =============================================================================
DATASET_DIR="/workspace/coco2017"   # đường dẫn tới dataset (COCO layout)
OUTPUT_DIR="output/train"           # thư mục lưu checkpoint và log
DATASET_FILE="coco"                 # coco | roboflow | o365
PRETRAINED_ENCODER=""               # đường dẫn .pth DINOv3 backbone (để trống = tự download)
# PRETRAIN_WEIGHTS không có ở đây — train from scratch không dùng RF-DETR COCO weights

# =============================================================================
# MODEL
# =============================================================================
MODEL_SIZE="nano"           # nano | small | base | large
USE_WINDOWED_ATTN=false     # true = bật window attention (tiết kiệm VRAM)
FREEZE_ENCODER=false        # true = đóng băng DINOv3 backbone

# =============================================================================
# CPFE — Cortical Perceptual Feature Enhancement
# =============================================================================
USE_CPFE=false               # master toggle — false = tắt toàn bộ CPFE
CPFE_USE_SDG=false           # Spectral Decomposition Gate (Center-Surround)
CPFE_USE_DN=false            # Divisive Normalization (Lateral Inhibition)
CPFE_USE_TPR=false           # Top-Down Predictive Refinement (Cortical Feedback)

# =============================================================================
# LW-DETR++ — virtual FPN neck, scale-aware RoPE, enhanced prototype memory
# =============================================================================
# Virtual FPN: backbone vẫn chỉ cần một lưới P4 (cùng H×W); projector_scale
# ['P3','P4','P5'] (hoặc thêm P6) là tên các mức ảo cho MSDeformAttn — không
# phải “chạy backbone 3 lần”.
USE_VIRTUAL_FPN_PROJECTOR=true   # true = MultiDilationP4Projector
PROJECTOR_INCLUDES_P6=true       # true → projector_scale P3–P6 (thêm mức coarse)
USE_SCALE_AWARE_ROPE=true      # true = RoPE 2D + log(w,h) ở decoder self-attn
ENHANCED_PROTOTYPE_MEMORY=true # true = EnhancedPrototypeMemory (τ/lớp, hard-neg, …)
PROTOTYPE_REPULSION_MARGIN=0.0
PROTOTYPE_USE_ADAPTIVE_TEMP=true
PROTOTYPE_USE_DUAL_PROTO=true
PROTOTYPE_HARD_NEG_K=5

# =============================================================================
# TRAINING RUN
# =============================================================================
EPOCHS=50
BATCH_SIZE=32                # per-GPU batch (tăng GRAD_ACCUM_STEPS để bù)
GRAD_ACCUM_STEPS=4          # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 64
NUM_WORKERS=8               # giảm worker tránh "Too many open files"
AMP=false                    # true = FP16 mixed precision
TENSORBOARD=true
DEVICE="cuda"               # cuda | cpu | mps
DEBUG_DATA_LIMIT=0          # 0 = full dataset; N > 0 = chỉ dùng N ảnh (smoke test)

# =============================================================================
# OPTIMIZER / LEARNING RATE SCHEDULE
# =============================================================================
LR=3e-4
LR_ENCODER=2.5e-5           # LR riêng cho encoder (thường nhỏ hơn)
LR_SCALE_MODE="sqrt"        # linear | sqrt
WARMUP_EPOCHS=3
LR_SCHEDULER="cosine_restart"  # cosine_restart | cosine | multistep | linear | wsd
LR_RESTART_PERIOD=25        # số epoch mỗi chu kỳ cosine restart
LR_RESTART_DECAY=0.8        # hệ số giảm LR mỗi restart
LR_MIN_FACTOR=0.05          # LR tối thiểu = LR × LR_MIN_FACTOR

# =============================================================================
# LOSS
# =============================================================================
USE_VARIFOCAL_LOSS=false    # true = varifocal loss; false = focal loss (khuyến nghị cho train from scratch)
CLS_LOSS_COEF=1.0
BBOX_LOSS_COEF=5.0
GIOU_LOSS_COEF=2.0

# =============================================================================
# PROTOTYPE ALIGNMENT
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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

# =============================================================================
# BUILD ARGS (POSIX sh — dùng positional params "$@")
# =============================================================================
set -- \
    --dataset-dir        "${DATASET_DIR}" \
    --output-dir         "${OUTPUT_DIR}" \
    --dataset-file       "${DATASET_FILE}" \
    --model-size         "${MODEL_SIZE}" \
    --epochs             "${EPOCHS}" \
    --batch-size         "${BATCH_SIZE}" \
    --grad-accum-steps   "${GRAD_ACCUM_STEPS}" \
    --num-workers        "${NUM_WORKERS}" \
    --device             "${DEVICE}" \
    --lr                 "${LR}" \
    --lr-encoder         "${LR_ENCODER}" \
    --lr-scale-mode      "${LR_SCALE_MODE}" \
    --lr-restart-period  "${LR_RESTART_PERIOD}" \
    --lr-restart-decay   "${LR_RESTART_DECAY}" \
    --lr-min-factor      "${LR_MIN_FACTOR}" \
    --cls-loss-coef      "${CLS_LOSS_COEF}" \
    --bbox-loss-coef     "${BBOX_LOSS_COEF}" \
    --giou-loss-coef     "${GIOU_LOSS_COEF}" \
    --prototype-loss-coef      "${PROTOTYPE_LOSS_COEF}" \
    --prototype-momentum       "${PROTOTYPE_MOMENTUM}" \
    --prototype-warmup-steps   "${PROTOTYPE_WARMUP_STEPS}" \
    --prototype-temperature    "${PROTOTYPE_TEMPERATURE}" \
    --prototype-repulsion-coef "${PROTOTYPE_REPULSION_COEF}"

# DINOv3 backbone weights (tuỳ chọn — để trống = auto-download)
if [ -n "${PRETRAINED_ENCODER}" ]; then
    set -- "$@" --pretrained-encoder "${PRETRAINED_ENCODER}"
fi

# Flags
if [ "$(_sh_lc "${AMP}")" = "true" ]; then
    set -- "$@" --amp
fi
if [ "$(_sh_lc "${TENSORBOARD}")" = "true" ]; then
    set -- "$@" --tensorboard
else
    set -- "$@" --no-tensorboard
fi
if [ "$(_sh_lc "${USE_WINDOWED_ATTN}")" = "true" ]; then
    set -- "$@" --use-windowed-attn
fi
if [ "$(_sh_lc "${FREEZE_ENCODER}")" = "true" ]; then
    set -- "$@" --freeze-encoder
fi
if [ "$(_sh_lc "${USE_VARIFOCAL_LOSS}")" = "true" ]; then
    set -- "$@" --use-varifocal-loss
fi

# Prototype flags
if [ "$(_sh_lc "${USE_PROTOTYPE_ALIGN}")" = "false" ]; then
    set -- "$@" --no-prototype-align
fi
if [ "$(_sh_lc "${PROTOTYPE_USE_FREQ_WEIGHT}")" = "false" ]; then
    set -- "$@" --no-prototype-use-freq-weight
fi
if [ "$(_sh_lc "${PROTOTYPE_USE_QUALITY_WEIGHT}")" = "false" ]; then
    set -- "$@" --no-prototype-use-quality-weight
fi
if [ "$(_sh_lc "${PROTOTYPE_USE_REPULSION}")" = "false" ]; then
    set -- "$@" --no-prototype-use-repulsion
fi

# CPFE flags — khi tắt: backbone gán self.cpfe = None, forward bỏ qua (không chạy ngầm).
if [ "$(_sh_lc "${USE_CPFE}")" = "false" ]; then
    set -- "$@" --no-cpfe
fi
if [ "$(_sh_lc "${CPFE_USE_SDG}")" = "false" ]; then
    set -- "$@" --no-cpfe-sdg
fi
if [ "$(_sh_lc "${CPFE_USE_DN}")" = "false" ]; then
    set -- "$@" --no-cpfe-dn
fi
if [ "$(_sh_lc "${CPFE_USE_TPR}")" = "false" ]; then
    set -- "$@" --no-cpfe-tpr
fi

# LW-DETR++
if [ "$(_sh_lc "${USE_VIRTUAL_FPN_PROJECTOR}")" = "true" ]; then
    set -- "$@" --use-virtual-fpn-projector
fi
if [ "$(_sh_lc "${PROJECTOR_INCLUDES_P6}")" = "true" ]; then
    set -- "$@" --projector-includes-p6
fi
if [ "$(_sh_lc "${USE_SCALE_AWARE_ROPE}")" = "true" ]; then
    set -- "$@" --use-scale-aware-rope
fi
if [ "$(_sh_lc "${ENHANCED_PROTOTYPE_MEMORY}")" = "true" ]; then
    set -- "$@" --enhanced-prototype-memory
fi
set -- "$@" --prototype-repulsion-margin "${PROTOTYPE_REPULSION_MARGIN}"
set -- "$@" --prototype-hard-neg-k "${PROTOTYPE_HARD_NEG_K}"
if [ "$(_sh_lc "${PROTOTYPE_USE_ADAPTIVE_TEMP}")" = "false" ]; then
    set -- "$@" --no-prototype-use-adaptive-temp
fi
if [ "$(_sh_lc "${PROTOTYPE_USE_DUAL_PROTO}")" = "false" ]; then
    set -- "$@" --no-prototype-use-dual-proto
fi

# Debug limit
if [ "${DEBUG_DATA_LIMIT}" -gt 0 ] 2>/dev/null; then
    set -- "$@" --debug-data-limit "${DEBUG_DATA_LIMIT}"
fi

# =============================================================================
# RUN
# =============================================================================
# Tăng giới hạn file descriptor để tránh "Too many open files"
ulimit -n 65536 2>/dev/null || true

echo "============================================================"
echo "  RF-DETR v2 Training (from scratch)"
echo "  Model   : ${MODEL_SIZE}  |  Epochs: ${EPOCHS}  |  BS: ${BATCH_SIZE}x${GRAD_ACCUM_STEPS}  |  CPFE: ${USE_CPFE}"
echo "  LW++    : vFPN=${USE_VIRTUAL_FPN_PROJECTOR}  P6=${PROJECTOR_INCLUDES_P6}  RoPE=${USE_SCALE_AWARE_ROPE}  EProto=${ENHANCED_PROTOTYPE_MEMORY}"
echo "  Dataset : ${DATASET_DIR}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  GPU     : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

exec python3 "${SCRIPT_DIR}/train.py" "$@"
