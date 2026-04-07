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

"""
Backbone modules.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetrv2.models.backbone.base import BackboneBase
from rfdetrv2.models.backbone.convnext_projector import (
    MultiDilationP4Projector,
    MultiScaleProjector,
)
from rfdetrv2.models.backbone.cpfe import CorticalPerceptualFeatureEnhancement
from rfdetrv2.models.backbone.dinov3 import DinoV3
from rfdetrv2.util.misc import NestedTensor

logger = logging.getLogger(__name__)

__all__ = ["Backbone"]

# ---------------------------------------------------------------------------
# Supported encoder names
# ---------------------------------------------------------------------------

_SUPPORTED_ENCODERS = frozenset({
    "dinov3_nano",
    "dinov3_small",
    "dinov3_base",
    "dinov3_large",
})



# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class Backbone(BackboneBase):
    """RF-DETR backbone: DINOv3 encoder + multi-scale projector."""

    def __init__(
        self,
        name: str,
        pretrained_encoder: str = None,
        window_block_indexes: list = None,   # kept for API compatibility; unused
        drop_path: float = 0.0,
        out_channels: int = 256,
        out_feature_indexes: list = None,
        projector_scale: list = None,
        use_cls_token: bool = False,          # kept for API compat; unused
        freeze_encoder: bool = False,
        layer_norm: bool = False,
        target_shape: tuple[int, int] = (640, 640),
        rms_norm: bool = False,
        backbone_lora: bool = False,          # kept for API compat; unused
        gradient_checkpointing: bool = False,
        load_dinov3_weights: bool = True,
        patch_size: int = 16,
        num_windows: int = 2,
        positional_encoding_size: int = 0,
        use_windowed_attn: bool = False,
        use_convnext_projector: bool = True,
        # CPFE — Cortical Perceptual Feature Enhancement
        use_cpfe: bool = True,
        cpfe_use_sdg: bool = True,
        cpfe_use_dn: bool = True,
        cpfe_use_tpr: bool = True,
        use_virtual_fpn_projector: bool = False,
    ):
        super().__init__()

        if name not in _SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder '{name}'. "
                f"Expected one of: {sorted(_SUPPORTED_ENCODERS)}."
            )

        if window_block_indexes is not None:
            logger.warning(
                "window_block_indexes is not used by the DINOv3 hub backbone "
                "(windowing is applied at the input level).  The argument is "
                "ignored."
            )
        if use_cls_token:
            logger.warning(
                "use_cls_token=True is not implemented for DINOv3; ignoring."
            )
        if backbone_lora:
            logger.warning(
                "backbone_lora=True is not implemented for DINOv3; ignoring."
            )

        size = name.split("_", maxsplit=1)[1]  # "small" | "base" | "large"

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

        # ------------------------------------------------------------------
        # Projector
        # ------------------------------------------------------------------
        # Normalize: ensure list (e.g. 'P4' -> ['P4'])
        self.projector_scale = [projector_scale] if isinstance(projector_scale, str) else list(projector_scale)
        assert len(self.projector_scale) > 0, "projector_scale must not be empty"
        assert sorted(self.projector_scale) == self.projector_scale, (
            "projector_scale must be in ascending order (e.g. ['P3','P4','P5'])."
        )

        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.use_virtual_fpn_projector = bool(use_virtual_fpn_projector)
        if self.use_virtual_fpn_projector:
            ps = self.projector_scale
            if ps not in (["P3", "P4", "P5"], ["P3", "P4", "P5", "P6"]):
                raise ValueError(
                    "use_virtual_fpn_projector=True requires projector_scale "
                    "['P3','P4','P5'] or ['P3','P4','P5','P6']."
                )
            with_p6 = "P6" in ps
            if not use_convnext_projector:
                logger.warning(
                    "use_virtual_fpn_projector ignores no_convnext_projector; "
                    "MultiDilation path always uses ConvNeXt-style fusion."
                )
            self.projector = MultiDilationP4Projector(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                num_blocks=4,
                dilations=(1, 3, 5),
                with_p6=with_p6,
            )
            logger.info(
                "Virtual FPN projector (multi-dilation P4 fuse → P3/P4/P5%s).",
                " + P6" if with_p6 else "",
            )
        else:
            self.projector = MultiScaleProjector(
                in_channels=self.encoder._out_feature_channels,
                out_channels=out_channels,
                scale_factors=scale_factors,
                layer_norm=layer_norm,
                rms_norm=rms_norm,
                use_convnext=use_convnext_projector,
            )

        # ------------------------------------------------------------------
        # CPFE — Cortical Perceptual Feature Enhancement
        # Insert between encoder and projector to enhance raw backbone features
        # ------------------------------------------------------------------
        self.use_rsa = False  # legacy SRA disabled
        if use_cpfe:
            self.cpfe = CorticalPerceptualFeatureEnhancement(
                in_channels_list=self.encoder._out_feature_channels,
                use_sdg=cpfe_use_sdg,
                use_dn=cpfe_use_dn,
                use_tpr=cpfe_use_tpr,
            )
            logger.info(
                "CPFE enabled — SDG=%s, DN=%s, TPR=%s",
                cpfe_use_sdg, cpfe_use_dn, cpfe_use_tpr,
            )
        else:
            self.cpfe = None

        self._export = False

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        self.encoder.export()

        try:
            from peft import PeftModel
            if isinstance(self.encoder, PeftModel):
                logger.info("Merging and unloading LoRA weights.")
                self.encoder.merge_and_unload()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tensor_list: NestedTensor):
        feats = self.encoder(tensor_list.tensors)
        if self.cpfe is not None:
            feats = self.cpfe(feats)
        feats = self.projector(feats)
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(
                m[None].float(), size=feat.shape[-2:]
            ).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors: torch.Tensor):
        feats = self.encoder(tensors)
        if self.cpfe is not None:
            feats = self.cpfe(feats)
        feats = self.projector(feats)
        out_feats, out_masks = [], []
        for feat in feats:
            b, _, h, w = feat.shape
            out_masks.append(
                torch.zeros((b, h, w), dtype=torch.bool, device=feat.device)
            )
            out_feats.append(feat)
        return out_feats, out_masks

    # ------------------------------------------------------------------
    # Per-parameter learning-rate / weight-decay pairs
    # ------------------------------------------------------------------

    def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
        """Build per-parameter ``{lr, weight_decay}`` mappings for the encoder.

        Applies layer-wise LR decay following the DINO convention:
        earlier layers receive a smaller LR than later layers.

        Works with DINOv3 torch.hub models whose transformer blocks are
        named ``encoder.model.blocks.<N>.<...>`` (timm / DINOv3 convention),
        as well as the older DINOv2 HuggingFace naming
        ``encoder.layer.<N>.<...>`` (kept for compatibility).
        """
        num_layers = args.out_feature_indexes[-1] + 1
        backbone_key = "backbone.0.encoder"
        named_param_lr_pairs = {}

        for n, p in self.named_parameters():
            full_name = f"{prefix}.{n}"
            if not p.requires_grad:
                continue

            is_cpfe_param = ".cpfe." in full_name

            if backbone_key in full_name:
                lr = (
                    args.lr_encoder
                    * _get_dino_lr_decay_rate(
                        full_name,
                        lr_decay_rate=args.lr_vit_layer_decay,
                        num_layers=num_layers,
                    )
                    * args.lr_component_decay ** 2
                )
                wd = args.weight_decay * _get_dino_weight_decay_rate(full_name)
                named_param_lr_pairs[full_name] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }
            elif is_cpfe_param:
                # CPFE: neck-level LR — between encoder and projector.
                # Uses 1x lr_component_decay (faster than encoder, same as neck).
                lr = args.lr_encoder * args.lr_component_decay
                wd = args.weight_decay * _get_dino_weight_decay_rate(full_name)
                named_param_lr_pairs[full_name] = {
                    "params": p,
                    "lr": lr,
                    "weight_decay": wd,
                }

        return named_param_lr_pairs


# ---------------------------------------------------------------------------
# LR / weight-decay helpers
# ---------------------------------------------------------------------------

# Tokens that indicate a parameter belongs to the patch embedding / position
# encoding group (layer 0 — lowest LR).  Covers both timm (DINOv3 hub) and
# HuggingFace (DINOv2) naming conventions.
_EMBED_TOKENS = frozenset({
    "patch_embed",   # timm: model.patch_embed.proj.*
    "cls_token",     # timm: model.cls_token
    "pos_embed",     # timm: model.pos_embed  (absolute) or rope buffers
    "register_tokens",  # timm: model.register_tokens
    "embeddings",    # HF:   model.embeddings.*
})

# Parameters that should not receive weight decay.
_NO_WD_TOKENS = frozenset({
    "gamma",         # LayerScale parameters
    "pos_embed",     # position embeddings
    "rel_pos",       # relative position biases
    "bias",          # all bias terms
    "norm",          # all normalisation layers (weight + bias)
    "embeddings",    # HF patch / pos embeddings
    "cls_token",
    "register_tokens",
})


def get_dino_lr_decay_rate(
    name: str,
    lr_decay_rate: float = 1.0,
    num_layers: int = 12,
) -> float:
    """Public alias kept for backwards-compatibility."""
    return _get_dino_lr_decay_rate(name, lr_decay_rate, num_layers)


def _get_dino_lr_decay_rate(
    name: str,
    lr_decay_rate: float = 1.0,
    num_layers: int = 12,
) -> float:
    """Compute the layer-wise LR decay multiplier for parameter *name*.

    Layer assignment
    ----------------
    * Embedding layers (patch_embed, cls_token, pos_embed, register_tokens,
      HF embeddings) → layer 0  (smallest LR)
    * Transformer block N                                 → layer N + 1
    * Everything else (norm, head, …)                     → num_layers + 1
      (largest LR — no decay)

    Supports two naming conventions:

    1. **timm / DINOv3 torch.hub** — ``...model.blocks.<N>.<sub_param>``
    2. **HuggingFace DINOv2**      — ``...encoder.layer.<N>.<sub_param>``
    """
    layer_id = num_layers + 1  # default: no decay

    # ------------------------------------------------------------------
    # Embedding layers — layer 0
    # ------------------------------------------------------------------
    if any(tok in name for tok in _EMBED_TOKENS):
        return lr_decay_rate ** (num_layers + 1)

    # ------------------------------------------------------------------
    # timm / DINOv3 torch.hub: ``.blocks.<N>.``
    # ------------------------------------------------------------------
    if ".blocks." in name:
        try:
            # e.g. "...encoder.model.blocks.7.attn.qkv.weight"
            after_blocks = name.split(".blocks.")[1]   # "7.attn.qkv.weight"
            layer_id = int(after_blocks.split(".")[0]) + 1
        except (IndexError, ValueError):
            pass  # keep default

    # ------------------------------------------------------------------
    # HuggingFace DINOv2: ``.layer.<N>.``  (kept for compatibility)
    # ------------------------------------------------------------------
    elif ".layer." in name and ".residual." not in name:
        try:
            after_layer = name[name.find(".layer."):]  # ".layer.7.norm1…"
            layer_id = int(after_layer.split(".")[2]) + 1
        except (IndexError, ValueError):
            pass  # keep default

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_dino_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """Public alias kept for backwards-compatibility."""
    return _get_dino_weight_decay_rate(name, weight_decay_rate)


def _get_dino_weight_decay_rate(
    name: str, weight_decay_rate: float = 1.0
) -> float:
    """Return 0.0 for parameters that should not receive weight decay."""
    if any(tok in name for tok in _NO_WD_TOKENS):
        return 0.0
    return weight_decay_rate