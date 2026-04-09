# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

import math
from typing import Callable, Dict, List

import torch
import torch.nn as nn

from rfdetrv2.models.backbone.backbone import *
from rfdetrv2.util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """Sine/cosine positional encoding for 2-D feature maps."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        self.normalize     = normalize
        self.scale         = 2 * math.pi

    def forward(self, tensor_list=None, mask: torch.Tensor = None, align_dim_orders: bool = True):
        # Accept either a NestedTensor or a raw mask tensor
        if mask is None:
            mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed  = not_mask.cumsum(1, dtype=torch.float32)
        x_embed  = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps      = 1e-6
            y_embed  = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed  = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t    = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t    = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x    = x_embed[:, :, :, None] / dim_t
        pos_y    = y_embed[:, :, :, None] / dim_t
        pos_x    = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y    = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos      = torch.cat((pos_y, pos_x), dim=3)
        return pos.permute(0, 3, 1, 2) if align_dim_orders else pos.permute(0, 3, 1, 2)


def build_position_encoding(hidden_dim: int, position_embedding: str = "sine") -> nn.Module:
    num_pos_feats = hidden_dim // 2
    if position_embedding in ("v2", "sine"):
        return PositionEmbeddingSine(num_pos_feats, normalize=True)
    raise ValueError(f"Unknown position_embedding type: {position_embedding}")


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor):
        """ """
        x = self[0](tensor_list)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if (
                hasattr(m, "export")
                and isinstance(m.export, Callable)
                and hasattr(m, "_export")
                and not m._export
            ):
                m.export()

    def forward_export(self, inputs: torch.Tensor):
        feats, masks = self[0](inputs)
        poss = []
        for feat, mask in zip(feats, masks):
            poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
        return feats, None, poss


def build_backbone(
    encoder,
    vit_encoder_num_layers,
    pretrained_encoder,
    window_block_indexes,
    drop_path,
    out_channels,
    out_feature_indexes,
    projector_scale,
    use_cls_token,
    hidden_dim,
    position_embedding,
    freeze_encoder,
    layer_norm,
    target_shape,
    rms_norm,
    backbone_lora,
    force_no_pretrain,
    gradient_checkpointing,
    load_dinov3_weights,
    patch_size,
    num_windows,
    positional_encoding_size,
    use_windowed_attn=False,
    use_convnext_projector=True,
    use_fsca=False,
    base_grid_size=0,
    fsca_heads=8,
):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(hidden_dim, position_embedding)

    backbone = Backbone(
        encoder,
        pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov3_weights=load_dinov3_weights,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
        use_windowed_attn=use_windowed_attn,
        use_convnext_projector=use_convnext_projector,
        use_fsca=use_fsca,
        base_grid_size=base_grid_size,
        fsca_heads=fsca_heads,
    )

    model = Joiner(backbone, position_embedding)
    return model
