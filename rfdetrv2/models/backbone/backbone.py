# # ------------------------------------------------------------------------
# # RF-DETR
# # Copyright (c) 2025 Roboflow. All Rights Reserved.
# # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# # ------------------------------------------------------------------------
# # Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# # Copyright (c) 2024 Baidu. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# # Copyright (c) 2021 Microsoft. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Copied from DETR (https://github.com/facebookresearch/detr)
# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# # ------------------------------------------------------------------------

# """
# Backbone modules.
# """

# import logging

# import torch
# import torch.nn.functional as F
# from peft import PeftModel

# from rfdetrv2.models.backbone.base import BackboneBase
# from rfdetrv2.models.backbone.dinov3 import DinoV3
# from rfdetrv2.models.backbone.projector import MultiScaleProjector
# from rfdetrv2.util.misc import NestedTensor

# logger = logging.getLogger(__name__)

# __all__ = ["Backbone"]


# class Backbone(BackboneBase):
#     """backbone."""

#     def __init__(
#         self,
#         name: str,
#         pretrained_encoder: str = None,
#         window_block_indexes: list = None,
#         drop_path=0.0,
#         out_channels=256,
#         out_feature_indexes: list = None,
#         projector_scale: list = None,
#         use_cls_token: bool = False,
#         freeze_encoder: bool = False,
#         layer_norm: bool = False,
#         target_shape: tuple[int, int] = (640, 640),
#         rms_norm: bool = False,
#         backbone_lora: bool = False,
#         gradient_checkpointing: bool = False,
#         load_dinov3_weights: bool = True,
#         patch_size: int = 14,
#         num_windows: int = 4,
#         positional_encoding_size: int = 0,
#     ):
#         super().__init__()
#         if name not in {"dinov3_small", "dinov3_base", "dinov3_large"}:
#             raise ValueError(
#                 f"Unsupported encoder '{name}'. Expected one of: dinov3_small, dinov3_base, dinov3_large."
#             )
#         size = name.split("_", maxsplit=1)[1]
#         self.encoder = DinoV3(
#             size=size,
#             out_feature_indexes=out_feature_indexes,
#             shape=target_shape,
#             use_registers=False,
#             use_windowed_attn=False,
#             gradient_checkpointing=gradient_checkpointing,
#             load_dinov3_weights=load_dinov3_weights,
#             pretrained_encoder=pretrained_encoder,
#             patch_size=patch_size,
#             num_windows=num_windows,
#             positional_encoding_size=positional_encoding_size,
#             drop_path_rate=drop_path,
#         )
#         if freeze_encoder:
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

#         self.projector_scale = projector_scale
#         assert len(self.projector_scale) > 0
#         assert sorted(self.projector_scale) == self.projector_scale, (
#             "only support projector scale P3/P4/P5/P6 in ascending order."
#         )
#         level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
#         scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

#         self.projector = MultiScaleProjector(
#             in_channels=self.encoder._out_feature_channels,
#             out_channels=out_channels,
#             scale_factors=scale_factors,
#             layer_norm=layer_norm,
#             rms_norm=rms_norm,
#         )

#         self._export = False

#     def export(self):
#         self._export = True
#         self._forward_origin = self.forward
#         self.forward = self.forward_export

#         if isinstance(self.encoder, PeftModel):
#             logger.info("Merging and unloading LoRA weights")
#             self.encoder.merge_and_unload()

#     def forward(self, tensor_list: NestedTensor):
#         """ """
#         feats = self.encoder(tensor_list.tensors)
#         feats = self.projector(feats)
#         out = []
#         for feat in feats:
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
#             out.append(NestedTensor(feat, mask))
#         return out

#     def forward_export(self, tensors: torch.Tensor):
#         feats = self.encoder(tensors)
#         feats = self.projector(feats)
#         out_feats = []
#         out_masks = []
#         for feat in feats:
#             b, _, h, w = feat.shape
#             out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
#             out_feats.append(feat)
#         return out_feats, out_masks

#     def get_named_param_lr_pairs(self, args, prefix: str = "backbone.0"):
#         num_layers = args.out_feature_indexes[-1] + 1
#         backbone_key = "backbone.0.encoder"
#         named_param_lr_pairs = {}
#         for n, p in self.named_parameters():
#             n = prefix + "." + n
#             if backbone_key in n and p.requires_grad:
#                 lr = (
#                     args.lr_encoder
#                     * get_dino_lr_decay_rate(
#                         n,
#                         lr_decay_rate=args.lr_vit_layer_decay,
#                         num_layers=num_layers,
#                     )
#                     * args.lr_component_decay**2
#                 )
#                 wd = args.weight_decay * get_dino_weight_decay_rate(n)
#                 named_param_lr_pairs[n] = {
#                     "params": p,
#                     "lr": lr,
#                     "weight_decay": wd,
#                 }
#         return named_param_lr_pairs


# def get_dino_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
#     layer_id = num_layers + 1
#     if name.startswith("backbone"):
#         if "embeddings" in name:
#             layer_id = 0
#         elif ".layer." in name and ".residual." not in name:
#             layer_id = int(name[name.find(".layer.") :].split(".")[2]) + 1
#     return lr_decay_rate ** (num_layers + 1 - layer_id)


# def get_dino_weight_decay_rate(name, weight_decay_rate=1.0):
#     if (
#         ("gamma" in name)
#         or ("pos_embed" in name)
#         or ("rel_pos" in name)
#         or ("bias" in name)
#         or ("norm" in name)
#         or ("embeddings" in name)
#     ):
#         weight_decay_rate = 0.0
#     return weight_decay_rate










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
from peft import PeftModel

from rfdetrv2.models.backbone.base import BackboneBase
from rfdetrv2.models.backbone.dinov3 import DinoV3, SIZE_TO_WIDTH
from rfdetrv2.models.backbone.convnext_projector import MultiScaleProjector
from rfdetrv2.models.backbone.sra import SemanticRoutingAttention
# from rfdetrv2.models.backbone.projector import MultiScaleProjector
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
        use_rsa: bool = False,
        sra_shared: bool = True,
        sra_G: int = 32,
        sra_heads: int = 8,
        use_convnext_projector: bool = True,
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

        # Full-resolution encoder features + SRA are incompatible with input tiling.
        effective_windowed = use_windowed_attn and not use_rsa
        if use_rsa and use_windowed_attn:
            logger.warning(
                "use_rsa=True: disabling DINOv3 windowed attention (full-res encoder + SRA)."
            )

        self.use_rsa = use_rsa
        self.encoder = DinoV3(
            size=size,
            out_feature_indexes=out_feature_indexes,
            shape=target_shape,
            use_registers=False,
            use_windowed_attn=effective_windowed,
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

        self.sra_module: nn.Module | None = None
        self.sra_modules: nn.ModuleList | None = None
        self.sra_shared = bool(sra_shared)
        if use_rsa:
            dim = SIZE_TO_WIDTH[size]
            # Mọi mức DINOv3 cùng channel dim → có thể dùng chung 1 SRA (giảm param ~× len(out_feature_indexes)).
            if sra_shared:
                self.sra_module = SemanticRoutingAttention(
                    dim=dim, G=sra_G, n_heads=sra_heads
                )
            else:
                self.sra_modules = nn.ModuleList(
                    SemanticRoutingAttention(dim=dim, G=sra_G, n_heads=sra_heads)
                    for _ in out_feature_indexes
                )

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

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
            layer_norm=layer_norm,
            rms_norm=rms_norm,
            use_convnext=use_convnext_projector,
        )

        self._export = False

    def _apply_sra(self, feats: list) -> list:
        """Run Semantic Routing Attention on each encoder scale (B, C, H, W) → same shape."""
        if not self.use_rsa:
            return feats
        sra = self.sra_module if self.sra_shared else None
        out = []
        for i, feat in enumerate(feats):
            b, c, h, w = feat.shape
            x = feat.flatten(2).transpose(1, 2)  # (B, N, C)
            if sra is not None:
                x = sra(x)
            else:
                x = self.sra_modules[i](x)
            out.append(x.transpose(1, 2).view(b, c, h, w))
        return out

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        self.encoder.export()

        if isinstance(self.encoder, PeftModel):
            logger.info("Merging and unloading LoRA weights.")
            self.encoder.merge_and_unload()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, tensor_list: NestedTensor):
        feats = self.encoder(tensor_list.tensors)
        feats = self._apply_sra(feats)
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
        feats = self._apply_sra(feats)
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

            is_sra_param = ".sra_modules." in full_name or ".sra_module." in full_name

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
            elif is_sra_param:
                # SRA: same scale as the ViT “top” block (no per-block decay).
                lr = args.lr_encoder * args.lr_component_decay ** 2
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