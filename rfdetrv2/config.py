# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
from typing import List, Literal, Optional

import torch
from pydantic import BaseModel, field_validator

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def pydantic_dump(model: BaseModel) -> dict:
    """Serialize a Pydantic v1/v2 model to a plain dict."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

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
    use_fsca: bool = False               # Replace ConvNeXtBlock with FSCAv2Block (XCA + DCT)
    fsca_heads: int = 8                  # XCA attention heads inside FSCAv2Block
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


class RFDETRV2NanoConfig(ModelConfig):
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


class RFDETRV2BaseConfig(ModelConfig):
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

class RFDETRV2SmallConfig(ModelConfig):
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


class RFDETRV2LargeConfig(ModelConfig):
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
    resolution: int = 640
    positional_encoding_size: int = 35
    use_windowed_attn: bool = False

class TrainConfig(BaseModel):
    lr: float = 3e-4
    lr_encoder: float = 6e-5  # ~0.00017 sau sqrt(8), nhỏ hơn 8e-5
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 100
    resume: Optional[str] = None
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = 100
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    use_varifocal_loss: bool = True
    use_convnext_projector: bool = True
    # Superposition-Aware Prototype Alignment
    use_prototype_align: bool = True
    prototype_loss_coef: float = 0.1       # weight for pull loss
    prototype_momentum: float = 0.999
    prototype_warmup_steps: int = 200
    prototype_temperature: float = 0.1
    prototype_ortho_coef: float = 0.1      # [B] co-occurrence orthogonality
    prototype_disambig_coef: float = 0.1   # [C] IoU-conditioned disambiguation
    prototype_sparse_coef: float = 0.05    # [D] sparse decomposition
    prototype_iou_threshold: float = 0.3   # min IoU to trigger disambiguation
    dataset_file: Literal["coco", "o365", "roboflow"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: str  # COCO layout root (train/, val/, annotations/); also default for coco_path when unset
    coco_path: Optional[str] = None  # If None, train_from_config uses dataset_dir for build_coco
    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    # Document-style aug (albumentations) when using coco_album transforms
    document_aug: bool = False
    use_ema: bool = True
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    tensorboard: bool = True
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: List[str] = None
    run_test: bool = False
    # Cap train/val/test samples for smoke tests (0 = use full splits).
    debug_data_limit: int = 0
    segmentation_head: bool = False
    eval_max_dets: int = 500
    freeze_encoder: bool = False  # Freeze DINOv3 backbone (no gradient update)
    # Remap model class indices → COCO category_id for eval (Roboflow / custom COCO)
    label_to_cat_id: Optional[List[int]] = None

    @field_validator("dataset_dir", "output_dir", "coco_path", mode="after")
    @classmethod
    def expand_paths(cls, v: str | None) -> str | None:
        """
        Expand user paths (e.g., '~' or paths with separators) but leave simple filenames
        (like 'rf-detr-base.pth') unchanged so they can match hosted model keys.
        """
        if v is None:
            return v
        return os.path.realpath(os.path.expanduser(v))


class FineTuneConfig(TrainConfig):
    """Fine-tuning hyper-parameters (two-phase encoder freeze + optional LoRA)."""

    unfreeze_at_epoch: Optional[int] = None
    backbone_lora: bool = False


class SegmentationTrainConfig(TrainConfig):
    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 5.0
    segmentation_head: bool = True


# Backward-compatible aliases (prefer RFDETRV2*Config)
RFDETRNanoConfig = RFDETRV2NanoConfig
RFDETRBaseConfig = RFDETRV2BaseConfig
RFDETRSmallConfig = RFDETRV2SmallConfig
RFDETRLargeConfig = RFDETRV2LargeConfig
