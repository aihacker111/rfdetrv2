#!/usr/bin/env python3
# =============================================================================
# RF-DETR v2 — Train from Scratch (Single-GPU / Multi-GPU)
#
# ⚠  Đây là script TRAIN FROM SCRATCH:
#    - Detection head được khởi tạo ngẫu nhiên (KHÔNG tải RF-DETR COCO weights)
#    - Chỉ tải DINOv3 backbone encoder (tự động download HuggingFace lần đầu)
#
# → Muốn finetune từ checkpoint COCO?  Dùng scripts/run_finetune.py
#
# Chạy:  python3 scripts/run_train.py
# Multi-GPU: NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/run_train.py
# =============================================================================

import os
import sys
import subprocess

# =============================================================================
# FIX: Ưu tiên venv để torchrun và python3 dùng cùng môi trường
# =============================================================================
VENV_BIN = "/venv/main/bin"
os.environ["PATH"] = f"{VENV_BIN}:{os.environ.get('PATH', '')}"

PYTHON  = os.path.join(VENV_BIN, "python3")
TORCHRUN = os.path.join(VENV_BIN, "torchrun")

# Kiểm tra torch sớm
try:
    result = subprocess.run(
        [PYTHON, "-c", "import torch"],
        check=True, capture_output=True,
    )
except subprocess.CalledProcessError:
    print(f"ERROR: PyTorch không tìm thấy trong {PYTHON}", file=sys.stderr)
    print("       Kiểm tra lại đường dẫn VENV_BIN hoặc cài torch trước.", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO_ROOT)

# =============================================================================
# PATHS — BẮT BUỘC chỉnh DATASET_DIR
# =============================================================================
DATASET_DIR      = "/workspace/coco2017"   # đường dẫn tới dataset (COCO layout)
OUTPUT_DIR       = "output/train"          # thư mục lưu checkpoint và log
DATASET_FILE     = "coco"                  # coco | roboflow | o365
PRETRAINED_ENCODER = ""                    # đường dẫn .pth DINOv3 backbone (để trống = tự download)

# =============================================================================
# MODEL
# =============================================================================
MODEL_SIZE        = "nano"   # nano | small | base | large
USE_WINDOWED_ATTN = False    # True = bật window attention (tiết kiệm VRAM)
FREEZE_ENCODER    = False    # True = đóng băng DINOv3 backbone

# =============================================================================
# CPFE — Cortical Perceptual Feature Enhancement
# =============================================================================
USE_CPFE     = False   # master toggle — False = tắt toàn bộ CPFE
CPFE_USE_SDG = False   # Spectral Decomposition Gate (Center-Surround)
CPFE_USE_DN  = False   # Divisive Normalization (Lateral Inhibition)
CPFE_USE_TPR = False   # Top-Down Predictive Refinement (Cortical Feedback)

# =============================================================================
# LW-DETR++ — virtual FPN neck, scale-aware RoPE, enhanced prototype memory
# =============================================================================
USE_VIRTUAL_FPN_PROJECTOR  = True    # True = MultiDilationP4Projector
PROJECTOR_INCLUDES_P6      = True    # True → projector_scale P3–P6 (thêm mức coarse)
USE_SCALE_AWARE_ROPE       = True    # True = RoPE 2D + log(w,h) ở decoder self-attn
ENHANCED_PROTOTYPE_MEMORY  = True    # True = EnhancedPrototypeMemory (τ/lớp, hard-neg, …)
PROTOTYPE_REPULSION_MARGIN = 0.0
PROTOTYPE_USE_ADAPTIVE_TEMP = True
PROTOTYPE_USE_DUAL_PROTO   = True
PROTOTYPE_HARD_NEG_K       = 5

# =============================================================================
# TRAINING RUN
# =============================================================================
EPOCHS           = 50
BATCH_SIZE       = 32    # per-GPU batch
GRAD_ACCUM_STEPS = 4     # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS = 128
NUM_WORKERS      = 8
AMP              = False  # True = FP16 mixed precision
TENSORBOARD      = True
DEVICE           = "cuda"  # cuda | cpu | mps
DEBUG_DATA_LIMIT = 0        # 0 = full dataset; N > 0 = chỉ dùng N ảnh (smoke test)

# =============================================================================
# OPTIMIZER / LEARNING RATE SCHEDULE
# =============================================================================
LR                = 3e-4
LR_ENCODER        = 2.5e-5   # LR riêng cho encoder
LR_SCALE_MODE     = "sqrt"   # linear | sqrt
WARMUP_EPOCHS     = 3
LR_SCHEDULER      = "cosine_restart"  # cosine_restart | cosine | multistep | linear | wsd
LR_RESTART_PERIOD = 25       # số epoch mỗi chu kỳ cosine restart
LR_RESTART_DECAY  = 0.8      # hệ số giảm LR mỗi restart
LR_MIN_FACTOR     = 0.05     # LR tối thiểu = LR × LR_MIN_FACTOR

# =============================================================================
# LOSS
# =============================================================================
USE_VARIFOCAL_LOSS = False   # False = focal loss (khuyến nghị cho train from scratch)
CLS_LOSS_COEF      = 1.0
BBOX_LOSS_COEF     = 5.0
GIOU_LOSS_COEF     = 2.0

# =============================================================================
# PROTOTYPE ALIGNMENT
# =============================================================================
USE_PROTOTYPE_ALIGN        = True
PROTOTYPE_LOSS_COEF        = 0.1
PROTOTYPE_MOMENTUM         = 0.999
PROTOTYPE_WARMUP_STEPS     = 200
PROTOTYPE_TEMPERATURE      = 0.1
PROTOTYPE_REPULSION_COEF   = 0.1
PROTOTYPE_USE_FREQ_WEIGHT  = True
PROTOTYPE_USE_QUALITY_WEIGHT = True
PROTOTYPE_USE_REPULSION    = True

# =============================================================================
# GPU / distributed (torchrun)
# =============================================================================
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

NPROC_PER_NODE = os.environ.get("NPROC_PER_NODE", "2")
TORCHRUN_EXTRA = os.environ.get("TORCHRUN_EXTRA", "")   # e.g. "--master_port 29501"

# =============================================================================
# BUILD ARGS
# =============================================================================
args = [
    "--dataset-dir",        DATASET_DIR,
    "--output-dir",         OUTPUT_DIR,
    "--dataset-file",       DATASET_FILE,
    "--model-size",         MODEL_SIZE,
    "--epochs",             str(EPOCHS),
    "--batch-size",         str(BATCH_SIZE),
    "--grad-accum-steps",   str(GRAD_ACCUM_STEPS),
    "--num-workers",        str(NUM_WORKERS),
    "--device",             DEVICE,
    "--lr",                 str(LR),
    "--lr-encoder",         str(LR_ENCODER),
    "--lr-scale-mode",      LR_SCALE_MODE,
    "--lr-restart-period",  str(LR_RESTART_PERIOD),
    "--lr-restart-decay",   str(LR_RESTART_DECAY),
    "--lr-min-factor",      str(LR_MIN_FACTOR),
    "--cls-loss-coef",      str(CLS_LOSS_COEF),
    "--bbox-loss-coef",     str(BBOX_LOSS_COEF),
    "--giou-loss-coef",     str(GIOU_LOSS_COEF),
    "--prototype-loss-coef",      str(PROTOTYPE_LOSS_COEF),
    "--prototype-momentum",       str(PROTOTYPE_MOMENTUM),
    "--prototype-warmup-steps",   str(PROTOTYPE_WARMUP_STEPS),
    "--prototype-temperature",    str(PROTOTYPE_TEMPERATURE),
    "--prototype-repulsion-coef", str(PROTOTYPE_REPULSION_COEF),
    "--prototype-repulsion-margin", str(PROTOTYPE_REPULSION_MARGIN),
    "--prototype-hard-neg-k",     str(PROTOTYPE_HARD_NEG_K),
]

# DINOv3 backbone weights (tuỳ chọn)
if PRETRAINED_ENCODER:
    args += ["--pretrained-encoder", PRETRAINED_ENCODER]

# --- boolean flags ---
def flag(condition, flag_true, flag_false=None):
    """Append flag_true if condition else flag_false (if given)."""
    if condition:
        args.append(flag_true)
    elif flag_false:
        args.append(flag_false)

flag(AMP,               "--amp")
flag(TENSORBOARD,       "--tensorboard",          "--no-tensorboard")
flag(USE_WINDOWED_ATTN, "--use-windowed-attn")
flag(FREEZE_ENCODER,    "--freeze-encoder")
flag(USE_VARIFOCAL_LOSS,"--use-varifocal-loss")

# Prototype
flag(not USE_PROTOTYPE_ALIGN,         "--no-prototype-align")
flag(not PROTOTYPE_USE_FREQ_WEIGHT,   "--no-prototype-use-freq-weight")
flag(not PROTOTYPE_USE_QUALITY_WEIGHT,"--no-prototype-use-quality-weight")
flag(not PROTOTYPE_USE_REPULSION,     "--no-prototype-use-repulsion")
flag(not PROTOTYPE_USE_ADAPTIVE_TEMP, "--no-prototype-use-adaptive-temp")
flag(not PROTOTYPE_USE_DUAL_PROTO,    "--no-prototype-use-dual-proto")

# CPFE
flag(not USE_CPFE,     "--no-cpfe")
flag(not CPFE_USE_SDG, "--no-cpfe-sdg")
flag(not CPFE_USE_DN,  "--no-cpfe-dn")
flag(not CPFE_USE_TPR, "--no-cpfe-tpr")

# LW-DETR++
flag(USE_VIRTUAL_FPN_PROJECTOR, "--use-virtual-fpn-projector")
flag(PROJECTOR_INCLUDES_P6,     "--projector-includes-p6")
flag(USE_SCALE_AWARE_ROPE,      "--use-scale-aware-rope")
flag(ENHANCED_PROTOTYPE_MEMORY, "--enhanced-prototype-memory")

# Debug
if DEBUG_DATA_LIMIT > 0:
    args += ["--debug-data-limit", str(DEBUG_DATA_LIMIT)]

# =============================================================================
# PRINT BANNER
# =============================================================================
def get_torch_version():
    try:
        r = subprocess.run(
            [PYTHON, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"

def get_python_version():
    try:
        r = subprocess.run([PYTHON, "--version"], capture_output=True, text=True, check=True)
        return r.stdout.strip() or r.stderr.strip()
    except Exception:
        return "unknown"

print("=" * 60)
print("  RF-DETR v2 Training (from scratch)")
print(f"  Python  : {get_python_version()}")
print(f"  Torch   : {get_torch_version()}")
print(f"  Model   : {MODEL_SIZE}  |  Epochs: {EPOCHS}  |  BS: {BATCH_SIZE}x{GRAD_ACCUM_STEPS}  |  CPFE: {USE_CPFE}")
print(f"  LW++    : vFPN={USE_VIRTUAL_FPN_PROJECTOR}  P6={PROJECTOR_INCLUDES_P6}  RoPE={USE_SCALE_AWARE_ROPE}  EProto={ENHANCED_PROTOTYPE_MEMORY}")
print(f"  Dataset : {DATASET_DIR}")
print(f"  Output  : {OUTPUT_DIR}")
print(f"  GPU     : CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}  nproc_per_node={NPROC_PER_NODE}")
print("=" * 60)

# =============================================================================
# RAISE FILE DESCRIPTOR LIMIT
# =============================================================================
try:
    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
except Exception:
    pass  # không nghiêm trọng — tiếp tục

# =============================================================================
# RUN torchrun
# =============================================================================
train_script = os.path.join(SCRIPT_DIR, "train.py")

torchrun_cmd = [
    TORCHRUN,
    f"--nproc_per_node={NPROC_PER_NODE}",
]

# TORCHRUN_EXTRA có thể chứa nhiều flags cách nhau bởi dấu cách
if TORCHRUN_EXTRA.strip():
    torchrun_cmd.extend(TORCHRUN_EXTRA.split())

torchrun_cmd.append(train_script)
torchrun_cmd.extend(args)

print(f"\n[run_train] Executing:\n  {' '.join(torchrun_cmd)}\n")

os.execv(TORCHRUN, torchrun_cmd)   # thay thế process hiện tại — signal/exit code truyền thẳng