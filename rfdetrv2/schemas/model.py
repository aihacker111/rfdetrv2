import os
from typing import List, Literal, Optional

import torch
from pydantic import BaseModel, field_validator

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class ModelConfig(BaseModel):
    encoder: Literal["dinov3_nano", "dinov3_small", "dinov3_base", "dinov3_large"]
    out_feature_indexes: List[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    pretrained_encoder: Optional[str] = None  # Path to DINOv3 .pth weights (avoids torch.hub cache)
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    use_windowed_attn: bool = False  # DINOv3: split image into num_windows² tiles for memory-efficient attention
    use_convnext_projector: bool = True  # True=ConvNeXt fusion, False=C2f (backbone projector)
    positional_encoding_size: int
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    freeze_encoder: bool = False  # Freeze DINOv3 backbone (no gradient update)
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    license: str = "Apache-2.0"

    @field_validator("pretrain_weights", "pretrained_encoder", mode="after")
    @classmethod
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Expand user paths (e.g., '~' or paths with separators) but leave simple filenames
        (like 'rf-detr-base.pth') unchanged so they can match hosted model keys.
        For pretrained_encoder "repo::weights" format, don't expand.
        """
        if v is None:
            return v
        if "::" in v:
            return v  # "repo::weights" format
        return os.path.realpath(os.path.expanduser(v))


class RFDETRNanoConfig(ModelConfig):
    """
    RF-DETR Nano: DINOv3 ViT-S (dinov3_vits16, 21M, MLP FFN).
    12 blocks, same out_feature_indexes as Small.
    """
    encoder: Literal["dinov3_nano", "dinov3_small", "dinov3_base", "dinov3_large"] = "dinov3_nano"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 8
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = None
    resolution: int = 384
    positional_encoding_size: int = 32
    use_windowed_attn: bool = False


class RFDETRBaseConfig(ModelConfig):
    """
    The configuration for an RF-DETR Base model.
    """
    encoder: Literal["dinov3_nano", "dinov3_small", "dinov3_base", "dinov3_large"] = "dinov3_base"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 8
    num_queries: int = 300
    num_select: int = 300
    # num_classes: int = 80
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = None
    resolution: int = 560
    positional_encoding_size: int = 35
    use_windowed_attn: bool = False

class RFDETRSmallConfig(ModelConfig):
    """
    The configuration for an RF-DETR Small model.
    DINOv3 ViT-S+ (dinov3_vits16plus) has 12 blocks (indices 0-11).
    Use [2, 5, 8, 11] to match the 4-quadrant distribution used by Base.
    """
    encoder: Literal["dinov3_nano", "dinov3_small", "dinov3_base", "dinov3_large"] = "dinov3_small"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 2
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 8
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]

    resolution: int = 512
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = None
    use_windowed_attn: bool = False

# class RFDETRLargeConfig(ModelConfig):
#     encoder: Literal["dinov3_nano", "dinov3_small", "dinov3_base", "dinov3_large"] = "dinov3_large"
#     hidden_dim: int = 256
#     dec_layers: int = 4
#     sa_nheads: int = 8
#     ca_nheads: int = 16
#     dec_n_points: int = 12
#     num_windows: int = 2
#     patch_size: int = 16
#     projector_scale: List[Literal["P4",]] = ["P4"]
#     out_feature_indexes: List[int] = [3, 6, 9, 12]
#     # num_classes: int = 80
#     positional_encoding_size: int = 640 // 16
#     pretrain_weights: Optional[str] = None
#     resolution: int = 640
#     use_windowed_attn: bool = False


class RFDETRLargeConfig(ModelConfig):
    """
    RF-DETR Large: DINOv3 ViT-L (``dinov3_vitl16`` hub); higher resolution than Base.
    """

    encoder: Literal["dinov3_nano", "dinov3_small", "dinov3_base", "dinov3_large"] = "dinov3_large"
    hidden_dim: int = 256
    patch_size: int = 16
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 8
    num_queries: int = 300
    num_select: int = 300
    # num_classes: int = 80
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P4", "P5"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = None
    resolution: int = 640
    positional_encoding_size: int = 35
    use_windowed_attn: bool = False
