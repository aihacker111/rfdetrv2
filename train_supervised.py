"""
RF-DETR Supervised Training (no RL).
Standard supervised detection training only - no reward-based loss weighting.
"""
from pathlib import Path
import argparse
import os
import sys

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from rfdetrv2 import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall
from rfdetrv2.util.dinov3_pretrained import resolve_pretrained_encoder_path

# Default paths (override via CLI or env)
DEFAULT_DATASET = os.environ.get("DATASET_DIR", "/lustre/scratch/client/scratch/dms/dms_group/COCO2017")
DEFAULT_OUTPUT = os.environ.get(
    "OUTPUT_DIR",
    "/lustre/scratch/client/scratch/dms/dms_group/tinvna/output/rfdetrv2_nano_supervised",
)
# DINOv3 pretrained weights — thư mục chứa .pth (override via PRETRAINED_ENCODER env)
DINO_WEIGHTS_BY_SIZE = {
    "nano":  "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",      # ViT-S 21M
    "small": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",   # ViT-S+ 29M
    "base":  "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}
DEFAULT_PRETRAINED = os.environ.get("PRETRAINED_ENCODER")  # None = use DINOV3_PRETRAINED_DIR/<model>.pth


def parse_args():
    parser = argparse.ArgumentParser("RF-DETR Supervised Training (no RL)")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET, help="Path to COCO dataset")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Output directory for checkpoints")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per process")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--model-size", choices=["nano", "small", "base", "large"], default="base",
                        help="nano=vits16 (21M), small=vits16plus (29M)")
    parser.add_argument("--pretrained-encoder", default=DEFAULT_PRETRAINED,
                        help="Path to DINOv3 .pth. If unset, uses dinov3_pretrained/<model>.pth")
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=False, help="Mixed precision training")
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--use-windowed-attn", action="store_true", help="Enable DINOv3 windowed attention (reduces VRAM)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Decoder LR (base, trước scaling). 2e-4 → ~5.66e-4 effective (8 GPU sqrt)")
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5,
                        help="Encoder LR. 2.5e-5 → ~7.07e-5 effective (8 GPU sqrt), tỉ lệ 1:8 vs decoder")
    parser.add_argument("--lr-scale-mode", type=str, default="sqrt",
                        choices=["linear", "sqrt"],
                        help="Multi-GPU LR scaling: sqrt (recommended) hoặc linear")
    parser.add_argument("--lr-restart-period", type=int, default=25,
                        help="Cosine restart: số epochs mỗi cycle")
    parser.add_argument("--lr-restart-decay", type=float, default=0.8,
                        help="Cosine restart: LR peak giảm mỗi cycle")
    parser.add_argument("--lr-min-factor", type=float, default=0.05,
                        help="LR tối thiểu cuối cycle. 0.05 = cosine decay sâu hơn (0.1→jump lớn khi restart)")
    parser.add_argument("--use-varifocal-loss", action="store_true", help="Use Varifocal loss instead of Focal loss")
    parser.add_argument("--cls-loss-coef", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--bbox-loss-coef", type=float, default=5.0, help="L1 bbox loss weight")
    parser.add_argument("--giou-loss-coef", type=float, default=2.0, help="GIoU loss weight")
    parser.add_argument("--no-convnext-projector", action="store_true", dest="no_use_convnext_projector",
                        help="Tắt ConvNeXt projector (dùng C2f). Mặc định: bật ConvNeXt.")
    # Prototype Alignment (lwdetr_query) — EMA class prototypes
    parser.add_argument("--no-prototype-align", action="store_false", dest="use_prototype_align",
                        help="Tắt prototype alignment loss")
    parser.add_argument("--prototype-loss-coef", type=float, default=0.1,
                        help="Weight của loss_proto_align. 0.1=safe default")
    parser.add_argument("--prototype-momentum", type=float, default=0.999,
                        help="EMA decay cho prototype. 0.999=stable, 0.99=adapt nhanh hơn")
    parser.add_argument("--prototype-warmup-steps", type=int, default=200,
                        help="Số bước đầu chỉ update prototype, chưa tính loss")
    parser.add_argument("--prototype-temperature", type=float, default=0.1,
                        help="τ trong cosine classifier. 0.1=hard negatives")
    # Enhanced PrototypeMemory (lwdetr_prototype)
    parser.add_argument("--prototype-repulsion-coef", type=float, default=0.1,
                        help="[ENH-3] Inter-class repulsion loss weight")
    parser.add_argument("--no-prototype-use-freq-weight", action="store_false",
                        dest="prototype_use_freq_weight",
                        help="[ENH-2] Tắt class-frequency weighting")
    parser.add_argument("--no-prototype-use-quality-weight", action="store_false",
                        dest="prototype_use_quality_weight",
                        help="[ENH-4] Tắt prototype quality weighting")
    parser.add_argument("--no-prototype-use-repulsion", action="store_false",
                        dest="prototype_use_repulsion",
                        help="[ENH-3] Tắt inter-class repulsion")
    parser.add_argument("--freeze-encoder", action="store_true", dest="freeze_encoder",
                        help="Freeze DINOv3 backbone (no gradient update, faster training)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # Resolve pretrained path: CLI > env > dinov3_pretrained | project root | auto-download
    explicit = args.pretrained_encoder or DEFAULT_PRETRAINED
    pretrained = resolve_pretrained_encoder_path(
        project_root,
        args.model_size,
        explicit=explicit if explicit else None,
        weights_by_size=DINO_WEIGHTS_BY_SIZE,
    )

    model_kwargs = dict(
        pretrained_encoder=pretrained,
        use_windowed_attn=getattr(args, "use_windowed_attn", False),
        use_rsa=False,  # supervised script: no Semantic Routing Attention (SRA)
        use_convnext_projector=not getattr(args, "no_use_convnext_projector", False),
        freeze_encoder=getattr(args, "freeze_encoder", False),
    )
    if args.model_size == "nano":
        model = RFDETRNano(**model_kwargs)
    elif args.model_size == "small":
        model = RFDETRSmall(**model_kwargs)
    elif args.model_size == "large":
        model = RFDETRLarge(**model_kwargs)
    else:
        model = RFDETRBase(**model_kwargs)

    model.train(
        coco_path=dataset_dir,
        dataset_dir=dataset_dir,
        dataset_file="coco",
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        use_ema=True,
        num_workers=args.num_workers,
        run_test=args.run_test,
        device="cuda",
        output_dir=output_dir,
        debug_data_limit=0,
        amp=args.amp,
        tensorboard=args.tensorboard,
        warmup_epochs=1,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        lr_scale_mode=getattr(args, "lr_scale_mode", "sqrt"),
        lr_scheduler="cosine_restart",  # Warm restart giúp thoát local minima
        lr_min_factor=args.lr_min_factor,
        lr_restart_period=getattr(args, "lr_restart_period", 50),
        lr_restart_decay=getattr(args, "lr_restart_decay", 0.8),
        use_varifocal_loss=getattr(args, "use_varifocal_loss", False),
        cls_loss_coef=getattr(args, "cls_loss_coef", 1.0),
        bbox_loss_coef=getattr(args, "bbox_loss_coef", 5.0),
        giou_loss_coef=getattr(args, "giou_loss_coef", 2.0),
        use_convnext_projector=not getattr(args, "no_use_convnext_projector", False),
        use_prototype_align=getattr(args, "use_prototype_align", True),
        prototype_loss_coef=getattr(args, "prototype_loss_coef", 0.1),
        prototype_momentum=getattr(args, "prototype_momentum", 0.999),
        prototype_warmup_steps=getattr(args, "prototype_warmup_steps", 200),
        prototype_temperature=getattr(args, "prototype_temperature", 0.1),
        prototype_repulsion_coef=getattr(args, "prototype_repulsion_coef", 0.1),
        prototype_use_freq_weight=getattr(args, "prototype_use_freq_weight", True),
        prototype_use_quality_weight=getattr(args, "prototype_use_quality_weight", True),
        prototype_use_repulsion=getattr(args, "prototype_use_repulsion", True),
        freeze_encoder=getattr(args, "freeze_encoder", False),
    )


if __name__ == "__main__":
    main()
