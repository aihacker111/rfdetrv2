#!/bin/sh
# RF-DETR supervised training (multi-GPU via torchrun).
#
# Environment variables (all optional — sensible defaults provided):
#   DATASET_DIR              - COCO root                 (default: /workspace/coco_2017)
#   OUTPUT_DIR               - run output dir            (default: /workspace/output/rfdetrv2_nano_supervised)
#   NUM_GPUS                 - torchrun --nproc_per_node (default: 2)
#   MASTER_PORT              - distributed port          (default: 29500)
#   CUDA_VISIBLE_DEVICES     - which GPUs to use         (default: 0,1)
#
# Prototype alignment (ENH-1..8) — override via env vars:
#   PROTO_LOSS_COEF          - overall prototype loss weight        (default: 0.1)
#   PROTO_TEMPERATURE        - cosine classifier temperature τ      (default: 0.1)
#   PROTO_MOMENTUM           - EMA decay target                     (default: 0.999)
#   PROTO_WARMUP_STEPS       - steps before loss is active          (default: 200)
#   PROTO_ARC_MARGIN         - [ENH-5] ArcFace additive margin      (default: 0.3)
#   PROTO_TRIPLET_MARGIN     - [ENH-6] hard-neg triplet margin      (default: 0.2)
#   PROTO_HARD_NEG_COEF      - [ENH-6] triplet loss weight          (default: 0.5)
#   PROTO_QUEUE_SIZE         - [ENH-7] features per class in queue  (default: 32)
#   PROTO_QUEUE_LOSS_COEF    - [ENH-7] queue cohesion loss weight   (default: 0.5)
#   PROTO_REPULSION_COEF     - [ENH-8] Gram orthogonality weight    (default: 0.1)
#
# To disable individual components (set to 1):
#   NO_PROTO=1               - disable all prototype alignment
#   NO_PROTO_ARC=1           - disable ArcFace margin     [ENH-5]
#   NO_PROTO_HARD_NEG=1      - disable hard-neg triplet   [ENH-6]
#   NO_PROTO_QUEUE=1         - disable feature queue      [ENH-7]
#   NO_PROTO_REPULSION=1     - disable Gram ortho         [ENH-8]
#   NO_PROTO_FREQ_WEIGHT=1   - disable freq weighting     [ENH-2]
#   NO_PROTO_QUALITY_WEIGHT=1 - disable quality weight    [ENH-4]

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly TRAIN_PY="${SCRIPT_DIR}/train_supervised.py"

# ── Training basics ────────────────────────────────────────────────────────────
DATASET_DIR="${DATASET_DIR:-/workspace/coco2017}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output/rfdetrv2_nano_supervised}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ── Prototype alignment hyperparameters ───────────────────────────────────────
PROTO_LOSS_COEF="${PROTO_LOSS_COEF:-0.1}"
PROTO_TEMPERATURE="${PROTO_TEMPERATURE:-0.1}"
PROTO_MOMENTUM="${PROTO_MOMENTUM:-0.999}"
PROTO_WARMUP_STEPS="${PROTO_WARMUP_STEPS:-200}"
PROTO_ARC_MARGIN="${PROTO_ARC_MARGIN:-0.3}"           # [ENH-5]
PROTO_TRIPLET_MARGIN="${PROTO_TRIPLET_MARGIN:-0.2}"   # [ENH-6]
PROTO_HARD_NEG_COEF="${PROTO_HARD_NEG_COEF:-0.5}"     # [ENH-6]
PROTO_QUEUE_SIZE="${PROTO_QUEUE_SIZE:-32}"             # [ENH-7]
PROTO_QUEUE_LOSS_COEF="${PROTO_QUEUE_LOSS_COEF:-0.5}" # [ENH-7]
PROTO_REPULSION_COEF="${PROTO_REPULSION_COEF:-0.1}"   # [ENH-8]

# ── GPU check ─────────────────────────────────────────────────────────────────
echo "Checking GPUs..."
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
  NUM_AVAILABLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
else
  echo "Warning: nvidia-smi not found."
  NUM_AVAILABLE_GPUS=0
fi

if [ "$NUM_AVAILABLE_GPUS" -gt 0 ] && [ "$NUM_AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
  echo "Warning: requested ${NUM_GPUS} GPU(s), only ${NUM_AVAILABLE_GPUS} visible — using ${NUM_AVAILABLE_GPUS}."
  NUM_GPUS="$NUM_AVAILABLE_GPUS"
fi
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "Error: need at least one GPU for this script." >&2
  exit 1
fi

# ── Build prototype flags ──────────────────────────────────────────────────────
# Use a plain string — all values are simple numbers/flags (no spaces in values),
# so unquoted word-splitting at call-site is correct and safe.
PROTO_FLAGS=""
pf() { PROTO_FLAGS="${PROTO_FLAGS} $*"; }

if [ "${NO_PROTO:-0}" = "1" ]; then
  pf --no-prototype-align
else
  # Core hyperparams
  pf --prototype-loss-coef      "$PROTO_LOSS_COEF"
  pf --prototype-temperature    "$PROTO_TEMPERATURE"
  pf --prototype-momentum       "$PROTO_MOMENTUM"
  pf --prototype-warmup-steps   "$PROTO_WARMUP_STEPS"
  # [ENH-5] ArcFace angular margin
  pf --prototype-arc-margin     "$PROTO_ARC_MARGIN"
  # [ENH-6] Hard negative triplet
  pf --prototype-triplet-margin "$PROTO_TRIPLET_MARGIN"
  pf --prototype-hard-neg-coef  "$PROTO_HARD_NEG_COEF"
  # [ENH-7] Per-class feature queue
  pf --prototype-queue-size      "$PROTO_QUEUE_SIZE"
  pf --prototype-queue-loss-coef "$PROTO_QUEUE_LOSS_COEF"
  # [ENH-8] Gram orthogonality
  pf --prototype-repulsion-coef  "$PROTO_REPULSION_COEF"

  # Optional kill-switches
  [ "${NO_PROTO_ARC:-0}"            = "1" ] && pf --no-prototype-arc-margin
  [ "${NO_PROTO_HARD_NEG:-0}"       = "1" ] && pf --no-prototype-hard-neg
  [ "${NO_PROTO_QUEUE:-0}"          = "1" ] && pf --no-prototype-queue
  [ "${NO_PROTO_REPULSION:-0}"      = "1" ] && pf --no-prototype-use-repulsion
  [ "${NO_PROTO_FREQ_WEIGHT:-0}"    = "1" ] && pf --no-prototype-use-freq-weight
  [ "${NO_PROTO_QUALITY_WEIGHT:-0}" = "1" ] && pf --no-prototype-use-quality-weight
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "Starting supervised training: ${NUM_GPUS} process(es)"
echo "  dataset : ${DATASET_DIR}"
echo "  output  : ${OUTPUT_DIR}"
if [ "${NO_PROTO:-0}" = "1" ]; then
  echo "  prototype alignment : DISABLED"
else
  echo "  prototype alignment : ENABLED"
  echo "    loss_coef=${PROTO_LOSS_COEF}  temperature=${PROTO_TEMPERATURE}  momentum=${PROTO_MOMENTUM}  warmup=${PROTO_WARMUP_STEPS}"
  echo "    [ENH-5] arc_margin=${PROTO_ARC_MARGIN}             (disable: NO_PROTO_ARC=1)"
  echo "    [ENH-6] triplet_margin=${PROTO_TRIPLET_MARGIN}  hard_neg_coef=${PROTO_HARD_NEG_COEF}  (disable: NO_PROTO_HARD_NEG=1)"
  echo "    [ENH-7] queue_size=${PROTO_QUEUE_SIZE}  queue_coef=${PROTO_QUEUE_LOSS_COEF}        (disable: NO_PROTO_QUEUE=1)"
  echo "    [ENH-8] repulsion_coef=${PROTO_REPULSION_COEF}                (disable: NO_PROTO_REPULSION=1)"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
# shellcheck disable=SC2086  # intentional word-split of PROTO_FLAGS
torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${TRAIN_PY}" \
  --dataset-dir  "${DATASET_DIR}" \
  --output-dir   "${OUTPUT_DIR}" \
  --batch-size   "${BATCH_SIZE_PER_GPU}" \
  --num-workers  4 \
  --epochs       50 \
  --model-size   nano \
  --use-varifocal-loss \
  --tensorboard \
  $PROTO_FLAGS

echo "Done."
