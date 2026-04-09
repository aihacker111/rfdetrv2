"""Count parameters and GFLOPs for RF-DETR v2 model variants.

Usage:
    python tools/count_params.py

Requires:
    pip install thop
"""

import argparse
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from thop import profile

from rfdetrv2.models.lwdetr import build_model
from rfdetrv2.config import (
    RFDETRV2NanoConfig,
    RFDETRV2SmallConfig,
    RFDETRV2BaseConfig,
    RFDETRV2LargeConfig,
)
from rfdetrv2.util.misc import NestedTensor


# (display_name, config_class, use_fsca)
VARIANTS = [
    ("RF-DETR-v2 Nano",       RFDETRV2NanoConfig,  False),
    ("RF-DETR-v2 Nano+FSCA",  RFDETRV2NanoConfig,  True),
    ("RF-DETR-v2 Small",      RFDETRV2SmallConfig, False),
    ("RF-DETR-v2 Small+FSCA", RFDETRV2SmallConfig, True),
    ("RF-DETR-v2 Base",       RFDETRV2BaseConfig,  False),
    ("RF-DETR-v2 Base+FSCA",  RFDETRV2BaseConfig,  True),
    ("RF-DETR-v2 Large",      RFDETRV2LargeConfig, False),
    ("RF-DETR-v2 Large+FSCA", RFDETRV2LargeConfig, True),
]


def make_args(cfg, use_fsca: bool) -> argparse.Namespace:
    """Build the minimal args Namespace needed by build_model."""
    R = cfg.resolution
    return argparse.Namespace(
        # Backbone
        encoder                  = cfg.encoder,
        vit_encoder_num_layers   = cfg.out_feature_indexes[-1] + 1,
        pretrained_encoder       = None,
        window_block_indexes     = [],
        drop_path                = 0.0,
        hidden_dim               = cfg.hidden_dim,
        out_feature_indexes      = cfg.out_feature_indexes,
        projector_scale          = cfg.projector_scale,
        use_cls_token            = False,
        position_embedding       = "sine",
        freeze_encoder           = False,
        layer_norm               = cfg.layer_norm,
        resolution               = R,
        rms_norm                 = False,
        backbone_lora            = False,
        force_no_pretrain        = False,
        gradient_checkpointing   = False,
        patch_size               = cfg.patch_size,
        num_windows              = cfg.num_windows,
        positional_encoding_size = cfg.positional_encoding_size,
        use_windowed_attn        = cfg.use_windowed_attn,
        use_convnext_projector   = cfg.use_convnext_projector,
        use_fsca                 = use_fsca,
        fsca_heads               = cfg.fsca_heads,
        # Transformer
        sa_nheads                = cfg.sa_nheads,
        ca_nheads                = cfg.ca_nheads,
        num_queries              = getattr(cfg, 'num_queries', 300),
        dropout                  = 0.0,
        dim_feedforward          = cfg.hidden_dim * 4,
        dec_layers               = cfg.dec_layers,
        group_detr               = 1,
        two_stage                = cfg.two_stage,
        num_feature_levels       = len(cfg.projector_scale),
        dec_n_points             = cfg.dec_n_points,
        lite_refpoint_refine     = cfg.lite_refpoint_refine,
        decoder_norm             = "LN",
        bbox_reparam             = cfg.bbox_reparam,
        use_rope                 = True,
        # Model
        num_classes              = cfg.num_classes,
        aux_loss                 = False,
        segmentation_head        = False,
        mask_downsample_ratio    = 4,
        device                   = "cpu",
        encoder_only             = False,
        backbone_only            = False,
    )


def build_from_config(cfg, use_fsca: bool) -> torch.nn.Module:
    args = make_args(cfg, use_fsca)
    return build_model(args).eval()


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_gflops(model, resolution):
    images = torch.zeros(1, 3, resolution, resolution)
    masks  = torch.zeros(1, resolution, resolution, dtype=torch.bool)
    dummy  = NestedTensor(images, masks)
    try:
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return macs * 2 / 1e9
    except Exception:
        return float("nan")


def main():
    col = f"{'Model':<26} {'Params (M)':>11} {'Trainable (M)':>14} {'GFLOPs':>9} {'Res':>6}"
    print(f"\n{col}\n{'-' * len(col)}")

    for name, config_cls, use_fsca in VARIANTS:
        try:
            cfg    = config_cls(dataset_dir="/tmp")
            model  = build_from_config(cfg, use_fsca)
            total, trainable = count_params(model)
            gflops = count_gflops(model, cfg.resolution)
            print(
                f"{name:<26} {total/1e6:>10.2f}M {trainable/1e6:>13.2f}M "
                f"{gflops:>9.2f} {cfg.resolution:>5}px"
            )
        except Exception as e:
            import traceback
            print(f"{name:<26}  ERROR: {e}")
            traceback.print_exc()

    print()


if __name__ == "__main__":
    main()
