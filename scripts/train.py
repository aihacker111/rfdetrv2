#!/usr/bin/env python3
"""RF-DETR v2 — supervised training from scratch (single or multi-GPU).

Usage
-----
  # Single GPU
  python scripts/train.py --dataset-dir /data/mydata --output-dir ./out

  # Multi-GPU (torchrun)
  torchrun --nproc_per_node=4 scripts/train.py --dataset-dir /data/mydata --output-dir ./out

Env overrides: DATASET_DIR, OUTPUT_DIR, PRETRAINED_ENCODER
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Final

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rfdetrv2 import RFDETRV2Base, RFDETRV2Large, RFDETRV2Nano, RFDETRV2Small

_DEFAULT_DATASET    = os.environ.get("DATASET_DIR", "")
_DEFAULT_OUTPUT     = os.environ.get("OUTPUT_DIR", "output/train")
_DEFAULT_PRETRAINED = os.environ.get("PRETRAINED_ENCODER")

_MODELS: Final[dict[str, Callable[..., Any]]] = {
    "nano":  RFDETRV2Nano,
    "small": RFDETRV2Small,
    "base":  RFDETRV2Base,
    "large": RFDETRV2Large,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RF-DETR v2 supervised training from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Paths ----
    g = p.add_argument_group("paths")
    g.add_argument("--dataset-dir", default=_DEFAULT_DATASET or None,
                   required=not _DEFAULT_DATASET,
                   help="Dataset root (COCO layout or Roboflow export)")
    g.add_argument("--output-dir", default=_DEFAULT_OUTPUT)
    g.add_argument("--dataset-file", choices=["coco", "roboflow", "o365"], default="roboflow")
    g.add_argument("--resume", default=None, help="Resume from checkpoint .pth")

    # ---- Model ----
    g = p.add_argument_group("model")
    g.add_argument("--model-size", choices=list(_MODELS), default="base")
    g.add_argument("--pretrained-encoder", default=_DEFAULT_PRETRAINED,
                   help="DINOv3 .pth path; omit to auto-download")
    g.add_argument("--num-classes", type=int, default=80)
    g.add_argument("--use-windowed-attn", action="store_true")
    g.add_argument("--use-fsca", action="store_true",
                   help="Replace ConvNeXt projector blocks with FSCAv2")
    g.add_argument("--fsca-heads", type=int, default=8)
    g.add_argument("--no-convnext-projector", action="store_true")
    g.add_argument("--freeze-encoder", action="store_true",
                   help="Freeze DINOv3 backbone (train neck + head only)")

    # ---- Training ----
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--batch-size", type=int, default=4,
                   help="Per-GPU batch size")
    g.add_argument("--grad-accum-steps", type=int, default=4)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--amp", action="store_true", default=True,
                   help="bfloat16 mixed precision (recommended)")
    g.add_argument("--no-amp", action="store_false", dest="amp")
    g.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    g.add_argument("--debug-data-limit", type=int, default=0,
                   help="Cap images per split for smoke tests (0 = full)")
    g.add_argument("--run-test", action="store_true")
    g.add_argument("--tensorboard", action="store_true", default=True)
    g.add_argument("--no-tensorboard", action="store_false", dest="tensorboard")
    g.add_argument("--wandb", action="store_true", default=False)
    g.add_argument("--checkpoint-interval", type=int, default=10)

    # ---- Optimizer / LR ----
    g = p.add_argument_group("optimizer")
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--lr-encoder", type=float, default=6e-5)
    g.add_argument("--lr-scale-mode", choices=["linear", "sqrt"], default="sqrt",
                   help="Auto-scale LR with world_size")
    g.add_argument("--lr-vit-layer-decay", type=float, default=0.8)
    g.add_argument("--lr-component-decay", type=float, default=0.7)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--warmup-epochs", type=float, default=1.0)
    g.add_argument("--lr-drop", type=int, default=100)
    g.add_argument("--use-ema", action="store_true", default=True)
    g.add_argument("--ema-decay", type=float, default=0.993)
    g.add_argument("--group-detr", type=int, default=13)

    # ---- Loss ----
    g = p.add_argument_group("loss")
    g.add_argument("--cls-loss-coef", type=float, default=1.0)
    g.add_argument("--bbox-loss-coef", type=float, default=5.0)
    g.add_argument("--giou-loss-coef", type=float, default=2.0)
    g.add_argument("--use-varifocal-loss", action="store_true", default=True)
    g.add_argument("--no-varifocal-loss", action="store_false", dest="use_varifocal_loss")

    # ---- Prototype alignment ----
    g = p.add_argument_group("prototype alignment (SuperpositionAware)")
    g.add_argument("--no-prototype-align",  action="store_false", dest="use_prototype_align")
    g.add_argument("--prototype-loss-coef",     type=float, default=0.1)
    g.add_argument("--prototype-momentum",      type=float, default=0.999)
    g.add_argument("--prototype-warmup-steps",  type=int,   default=200)
    g.add_argument("--prototype-temperature",   type=float, default=0.1)
    g.add_argument("--prototype-ortho-coef",    type=float, default=0.1)
    g.add_argument("--prototype-disambig-coef", type=float, default=0.1)
    g.add_argument("--prototype-sparse-coef",   type=float, default=0.05)
    g.add_argument("--prototype-iou-threshold", type=float, default=0.3)

    p.set_defaults(use_prototype_align=True, use_varifocal_loss=True, use_ema=True, amp=True)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    use_convnext = not args.no_convnext_projector

    model = _MODELS[args.model_size](
        num_classes            = args.num_classes,
        pretrained_encoder     = args.pretrained_encoder,  # None → auto-download
        use_windowed_attn      = args.use_windowed_attn,
        use_convnext_projector = use_convnext,
        use_fsca               = args.use_fsca,
        fsca_heads             = args.fsca_heads,
        freeze_encoder         = args.freeze_encoder,
        device                 = args.device,
    )

    model.train(
        # Dataset
        dataset_dir         = args.dataset_dir,
        dataset_file        = args.dataset_file,
        coco_path           = args.dataset_dir if args.dataset_file == "coco" else None,
        output_dir          = args.output_dir,
        resume              = args.resume,
        # Training loop
        epochs              = args.epochs,
        batch_size          = args.batch_size,
        grad_accum_steps    = args.grad_accum_steps,
        num_workers         = args.num_workers,
        amp                 = args.amp,
        device              = args.device,
        debug_data_limit    = args.debug_data_limit,
        run_test            = args.run_test,
        tensorboard         = args.tensorboard,
        wandb               = args.wandb,
        checkpoint_interval = args.checkpoint_interval,
        # Optimizer
        lr                  = args.lr,
        lr_encoder          = args.lr_encoder,
        lr_scale_mode       = args.lr_scale_mode,
        lr_vit_layer_decay  = args.lr_vit_layer_decay,
        lr_component_decay  = args.lr_component_decay,
        weight_decay        = args.weight_decay,
        warmup_epochs       = args.warmup_epochs,
        lr_drop             = args.lr_drop,
        use_ema             = args.use_ema,
        ema_decay           = args.ema_decay,
        group_detr          = args.group_detr,
        # Loss
        cls_loss_coef       = args.cls_loss_coef,
        bbox_loss_coef      = args.bbox_loss_coef,
        giou_loss_coef      = args.giou_loss_coef,
        use_varifocal_loss  = args.use_varifocal_loss,
        # Prototype alignment
        use_prototype_align     = args.use_prototype_align,
        prototype_loss_coef     = args.prototype_loss_coef,
        prototype_momentum      = args.prototype_momentum,
        prototype_warmup_steps  = args.prototype_warmup_steps,
        prototype_temperature   = args.prototype_temperature,
        prototype_ortho_coef    = args.prototype_ortho_coef,
        prototype_disambig_coef = args.prototype_disambig_coef,
        prototype_sparse_coef   = args.prototype_sparse_coef,
        prototype_iou_threshold = args.prototype_iou_threshold,
        # Model flags
        use_convnext_projector  = use_convnext,
        freeze_encoder          = args.freeze_encoder,
    )


if __name__ == "__main__":
    main()
