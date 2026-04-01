#!/usr/bin/env python3
"""
RF-DETR v2 — fine-tune from COCO (or custom) RF-DETR weights on a new dataset.

Resolves DINOv3 encoder + optional RF-DETR ``pretrain_weights`` (CLI or HuggingFace),
then runs ``model.finetune()`` (two-phase encoder freeze supported).

Usage
-----
  python scripts/finetune.py --dataset-dir /data/custom --output-dir ./out

Env: DATASET_DIR, OUTPUT_DIR, COCO_WEIGHTS (RF-DETR .pth), RFDETR_SKIP_COCO_CHECKPOINT=1
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
from rfdetrv2.util.dinov3_pretrained import resolve_pretrained_encoder_path
from rfdetrv2.util.rfdetr_pretrained import resolve_rfdetr_coco_checkpoint

_DEFAULT_DATASET = os.environ.get("DATASET_DIR", "")
_DEFAULT_OUTPUT = os.environ.get("OUTPUT_DIR", "output/finetune")
_DEFAULT_PRETRAINED = os.environ.get("PRETRAINED_ENCODER")
_DEFAULT_COCO_WEIGHTS = os.environ.get("COCO_WEIGHTS")

DINO_WEIGHTS_BY_SIZE: Final[dict[str, str]] = {
    "nano": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "small": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}

_MODELS: Final[dict[str, Callable[..., Any]]] = {
    "nano": RFDETRV2Nano,
    "small": RFDETRV2Small,
    "base": RFDETRV2Base,
    "large": RFDETRV2Large,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RF-DETR v2 fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    paths = p.add_argument_group("paths")
    paths.add_argument(
        "--dataset-dir",
        default=_DEFAULT_DATASET or None,
        required=_DEFAULT_DATASET == "",
        help="Dataset root",
    )
    paths.add_argument("--output-dir", default=_DEFAULT_OUTPUT)
    paths.add_argument("--dataset-file", choices=["coco", "roboflow", "o365"], default="coco")

    ckpt = p.add_argument_group("checkpoints")
    ckpt.add_argument(
        "--coco-weights",
        default=_DEFAULT_COCO_WEIGHTS,
        help="RF-DETR .pth; unset → download/use rfdetr_pretrained/ (unless skip)",
    )
    ckpt.add_argument(
        "--skip-coco-checkpoint",
        action="store_true",
        help="Train detection head from DINO only (same as RFDETR_SKIP_COCO_CHECKPOINT=1)",
    )

    run = p.add_argument_group("run")
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--epochs", type=int, default=20)
    run.add_argument("--grad-accum-steps", type=int, default=4)
    run.add_argument("--run-test", action="store_true")
    run.add_argument("--tensorboard", action="store_true", default=True)
    run.add_argument("--no-tensorboard", action="store_false", dest="tensorboard")
    run.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    run.add_argument("--amp", action="store_true", default=True)
    run.add_argument("--no-amp", action="store_false", dest="amp")

    model = p.add_argument_group("model")
    model.add_argument("--model-size", choices=list(_MODELS.keys()), default="nano")
    model.add_argument("--pretrained-encoder", default=_DEFAULT_PRETRAINED)
    model.add_argument("--use-windowed-attn", action="store_true")
    model.add_argument("--no-convnext-projector", action="store_true")

    ft = p.add_argument_group("fine-tune")
    ft.add_argument("--freeze-encoder", action="store_true", help="Start with frozen backbone")
    ft.add_argument("--unfreeze-at-epoch", type=int, default=None, help="Epoch to unfreeze (needs --freeze-encoder)")
    ft.add_argument("--backbone-lora", action="store_true", help="LoRA on backbone (requires peft)")

    opt = p.add_argument_group("optimizer / schedule")
    opt.add_argument("--lr", type=float, default=2e-4)
    opt.add_argument("--lr-encoder", type=float, default=2.5e-5)
    opt.add_argument("--lr-scale-mode", choices=["linear", "sqrt"], default="sqrt")
    opt.add_argument("--warmup-epochs", type=float, default=1.0)
    opt.add_argument("--lr-scheduler", default="cosine_restart")
    opt.add_argument("--lr-restart-period", type=int, default=5)
    opt.add_argument("--lr-restart-decay", type=float, default=0.8)
    opt.add_argument("--lr-min-factor", type=float, default=0.05)

    loss = p.add_argument_group("loss")
    loss.add_argument("--use-varifocal-loss", action="store_true")
    loss.add_argument("--cls-loss-coef", type=float, default=1.0)
    loss.add_argument("--bbox-loss-coef", type=float, default=5.0)
    loss.add_argument("--giou-loss-coef", type=float, default=2.0)

    proto = p.add_argument_group("prototype alignment")
    proto.add_argument("--no-prototype-align", action="store_false", dest="use_prototype_align")
    proto.add_argument("--prototype-loss-coef", type=float, default=0.1)
    proto.add_argument("--prototype-momentum", type=float, default=0.999)
    proto.add_argument("--prototype-warmup-steps", type=int, default=200)
    proto.add_argument("--prototype-temperature", type=float, default=0.1)
    proto.add_argument("--prototype-repulsion-coef", type=float, default=0.1)
    proto.add_argument("--no-prototype-use-freq-weight", action="store_false", dest="prototype_use_freq_weight")
    proto.add_argument("--no-prototype-use-quality-weight", action="store_false", dest="prototype_use_quality_weight")
    proto.add_argument("--no-prototype-use-repulsion", action="store_false", dest="prototype_use_repulsion")

    p.set_defaults(
        use_prototype_align=True,
        prototype_use_freq_weight=True,
        prototype_use_quality_weight=True,
        prototype_use_repulsion=True,
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.skip_coco_checkpoint:
        os.environ["RFDETR_SKIP_COCO_CHECKPOINT"] = "1"

    explicit_enc = args.pretrained_encoder or _DEFAULT_PRETRAINED
    pretrained_path = resolve_pretrained_encoder_path(
        _PROJECT_ROOT,
        args.model_size,
        explicit=explicit_enc if explicit_enc else None,
        weights_by_size=DINO_WEIGHTS_BY_SIZE,
    )

    coco_ckpt = resolve_rfdetr_coco_checkpoint(
        _PROJECT_ROOT,
        args.model_size,
        explicit=args.coco_weights if args.coco_weights else None,
    )

    use_convnext = not args.no_convnext_projector
    model_kw: dict[str, Any] = dict(
        pretrained_encoder=pretrained_path,
        use_windowed_attn=args.use_windowed_attn,
        use_rsa=False,
        use_convnext_projector=use_convnext,
        device=args.device,
    )
    if coco_ckpt:
        model_kw["pretrain_weights"] = coco_ckpt
        print(f"[finetune] pretrain_weights: {coco_ckpt}", file=sys.stderr)
    else:
        print("[finetune] No RF-DETR COCO checkpoint — encoder-only init.", file=sys.stderr)

    model = _MODELS[args.model_size](**model_kw)

    coco_path = args.dataset_dir if args.dataset_file == "coco" else None
    model.finetune(
        coco_path=coco_path or args.dataset_dir,
        dataset_dir=args.dataset_dir,
        dataset_file=args.dataset_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        use_ema=True,
        num_workers=args.num_workers,
        run_test=args.run_test,
        device=args.device,
        output_dir=args.output_dir,
        amp=args.amp,
        tensorboard=args.tensorboard,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_scale_mode=args.lr_scale_mode,
        lr_scheduler=args.lr_scheduler,
        lr_min_factor=args.lr_min_factor,
        lr_restart_period=args.lr_restart_period,
        lr_restart_decay=args.lr_restart_decay,
        use_varifocal_loss=args.use_varifocal_loss,
        cls_loss_coef=args.cls_loss_coef,
        bbox_loss_coef=args.bbox_loss_coef,
        giou_loss_coef=args.giou_loss_coef,
        use_convnext_projector=use_convnext,
        use_prototype_align=args.use_prototype_align,
        prototype_loss_coef=args.prototype_loss_coef,
        prototype_momentum=args.prototype_momentum,
        prototype_warmup_steps=args.prototype_warmup_steps,
        prototype_temperature=args.prototype_temperature,
        prototype_repulsion_coef=args.prototype_repulsion_coef,
        prototype_use_freq_weight=args.prototype_use_freq_weight,
        prototype_use_quality_weight=args.prototype_use_quality_weight,
        prototype_use_repulsion=args.prototype_use_repulsion,
        freeze_encoder=args.freeze_encoder,
        unfreeze_at_epoch=args.unfreeze_at_epoch,
        backbone_lora=args.backbone_lora,
    )


if __name__ == "__main__":
    main()
