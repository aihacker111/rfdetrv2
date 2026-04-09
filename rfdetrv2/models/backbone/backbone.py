# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""Backbone modules."""

import logging

import torch
import torch.nn.functional as F
from peft import PeftModel

from rfdetrv2.models.backbone.base import BackboneBase
from rfdetrv2.models.backbone.dinov3 import DinoV3
from rfdetrv2.models.backbone.convnext_projector import MultiScaleProjector
from rfdetrv2.util.misc import NestedTensor

logger = logging.getLogger(__name__)

__all__ = ["Backbone"]

_SUPPORTED_ENCODERS = frozenset({
    "dinov3_nano",
    "dinov3_small",
    "dinov3_base",
    "dinov3_large",
})


class Backbone(BackboneBase):
    """RF-DETR backbone: DINOv3 encoder + multi-scale projector."""

    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,   # kept for API compat; unused
        drop_path: float = 0.0,
        out_channels: int = 256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,          # kept for API compat; unused
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,          # kept for API compat; unused
        gradient_checkpointing: bool = False,
        load_dinov3_weights: bool = True,
        patch_size: int = 16,
        num_windows: int = 2,
        positional_encoding_size: int = 0,
        use_windowed_attn: bool = False,
        use_convnext_projector: bool = True,
        use_fsca: bool = False,
        base_grid_size: int = 0,
        fsca_heads: int = 8,
    ):
        super().__init__()

        if name not in _SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder '{name}'. Expected one of: {sorted(_SUPPORTED_ENCODERS)}."
            )
        if window_block_indexes is not None:
            logger.warning("window_block_indexes is ignored (windowing is applied at input level).")
        if use_cls_token:
            logger.warning("use_cls_token=True is not implemented for DINOv3; ignoring.")
        if backbone_lora:
            logger.warning("backbone_lora=True is not implemented for DINOv3; ignoring.")

        size = name.split("_", maxsplit=1)[1]  # "nano" | "small" | "base" | "large"

        self.encoder = DinoV3(
            size=size,
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=False,
            use_windowed_attn=use_windowed_attn,
            gradient_checkpointing=gradient_checkpointing,
            load_dinov3_weights=load_dinov3_weights,
            pretrained_encoder=pretrained_encoder,
            patch_size=patch_size,
            num_windows=num_windows,
            positional_encoding_size=positional_encoding_size,
            drop_path_rate=drop_path,
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("DINOv3 encoder weights frozen.")

        self.projector_scale = [projector_scale] if isinstance(projector_scale, str) else list(projector_scale)
        assert len(self.projector_scale) > 0, "projector_scale must not be empty"
        assert sorted(self.projector_scale) == self.projector_scale, (
            "projector_scale must be in ascending order (e.g. ['P3','P4','P5'])."
        )

        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            use_convnext=use_convnext_projector,
            use_fsca=use_fsca,
            base_grid_size=base_grid_size,
            fsca_heads=fsca_heads,
        )

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        self.encoder.export()
        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights.")
            self.encoder.merge_and_unload()

    def forward(self, tensor_list: NestedTensor):
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        out_feats, out_masks = [], []
        for feat in feats:
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        """Build per-parameter {lr, weight_decay} mappings with layer-wise LR decay."""
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}

        for n, p in self.named_parameters():
            full_name = f"{prefix}.{n}"
            if not p.requires_grad:
                continue
            if backbone_key in full_name:
                lr = (
                    args.lr_encoder
                    * _get_dino_lr_decay_rate(full_name, lr_decay_rate=args.lr_vit_layer_decay, num_layers=num_layers)
                    * args.lr_component_decay ** 2
                )
                wd = args.weight_decay * _get_dino_weight_decay_rate(full_name)
                named_param_lr_pairs[full_name] = {"params": p, "lr": lr, "weight_decay": wd}

        return named_param_lr_pairs


# ---------------------------------------------------------------------------
# LR / weight-decay helpers
# ---------------------------------------------------------------------------

_EMBED_TOKENS = frozenset({
    "patch_embed", "cls_token", "pos_embed", "register_tokens", "embeddings",
})

_NO_WD_TOKENS = frozenset({
    "gamma", "pos_embed", "rel_pos", "bias", "norm", "embeddings", "cls_token", "register_tokens",
})


def get_dino_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Public alias for backwards-compatibility."""
    return _get_dino_lr_decay_rate(name, lr_decay_rate, num_layers)


def _get_dino_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """Layer-wise LR decay. Embedding layers → layer 0; block N → layer N+1; rest → no decay."""
    layer_id = num_layers + 1
    if any(tok in name for tok in _EMBED_TOKENS):
        return lr_decay_rate ** (num_layers + 1)
    if ".blocks." in name:
        try:
            layer_id = int(name.split(".blocks.")[1].split(".")[0]) + 1
        except (IndexError, ValueError):
            pass
    elif ".layer." in name and ".residual." not in name:
        try:
            layer_id = int(name[name.find(".layer."):].split(".")[2]) + 1
        except (IndexError, ValueError):
            pass
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_dino_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """Public alias for backwards-compatibility."""
    return _get_dino_weight_decay_rate(name, weight_decay_rate)


def _get_dino_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """Return 0.0 for parameters that should not receive weight decay."""
    if any(tok in name for tok in _NO_WD_TOKENS):
        return 0.0
    return weight_decay_rate
