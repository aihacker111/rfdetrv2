# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
build_model                    — construct the full LWDETR detection model.
build_criterion_and_postprocessors — construct SetCriterion + PostProcess.
"""

from __future__ import annotations

import torch

from rfdetrv2.models.backbone import build_backbone
from rfdetrv2.models.criterion import SetCriterion
from rfdetrv2.models.detector import LWDETR, PostProcess
from rfdetrv2.models.matcher import build_matcher
from rfdetrv2.models.prototype_memory import EnhancedPrototypeMemory, PrototypeMemory
from rfdetrv2.models.segmentation_head import SegmentationHead
from rfdetrv2.models.transformers_cdn import build_transformer


def build_model(args):
    """Build a complete LWDETR model from a flat args namespace."""
    num_classes = args.num_classes + 1
    torch.device(args.device)

    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=(
            args.shape if hasattr(args, "shape")
            else (args.resolution, args.resolution) if hasattr(args, "resolution")
            else (640, 640)
        ),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov3_weights=True,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
        use_windowed_attn=getattr(args, "use_windowed_attn", False),
        use_convnext_projector=getattr(args, "use_convnext_projector", True),
        use_cpfe=getattr(args, "use_cpfe", True),
        cpfe_use_sdg=getattr(args, "cpfe_use_sdg", True),
        cpfe_use_dn=getattr(args, "cpfe_use_dn", True),
        cpfe_use_tpr=getattr(args, "cpfe_use_tpr", True),
        use_virtual_fpn_projector=getattr(args, "use_virtual_fpn_projector", False),
    )

    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    segmentation_head = (
        SegmentationHead(
            args.hidden_dim, args.dec_layers,
            downsample_ratio=args.mask_downsample_ratio,
        )
        if args.segmentation_head else None
    )

    return LWDETR(
        backbone, transformer, segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
    )


def build_criterion_and_postprocessors(args):
    """Build SetCriterion and PostProcess from a flat args namespace."""
    device  = torch.device(args.device)
    matcher = build_matcher(args)

    weight_dict = {
        "loss_ce":   args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }
    if args.segmentation_head:
        weight_dict["loss_mask_ce"]   = args.mask_ce_loss_coef
        weight_dict["loss_mask_dice"] = args.mask_dice_loss_coef

    # ------------------------------------------------------------------
    # Prototype alignment — PrototypeMemory
    # ------------------------------------------------------------------
    use_proto        = getattr(args, "use_prototype_align", True)
    prototype_memory = None

    if use_proto:
        proto_coef = getattr(args, "prototype_loss_coef", 0.1)
        weight_dict["loss_proto_align"] = proto_coef

        proto_kw = dict(
            num_classes        = args.num_classes,
            feat_dim           = args.hidden_dim,
            momentum           = getattr(args, "prototype_momentum",           0.999),
            warmup_steps       = getattr(args, "prototype_warmup_steps",       200),
            temperature        = getattr(args, "prototype_temperature",        0.1),
            repulsion_coef     = getattr(args, "prototype_repulsion_coef",     0.1),
            use_freq_weight    = getattr(args, "prototype_use_freq_weight",    True),
            use_quality_weight = getattr(args, "prototype_use_quality_weight", True),
            use_repulsion      = getattr(args, "prototype_use_repulsion",      True),
        )
        if getattr(args, "enhanced_prototype_memory", False):
            prototype_memory = EnhancedPrototypeMemory(
                **proto_kw,
                repulsion_margin=getattr(args, "prototype_repulsion_margin", 0.0),
                use_adaptive_temp=getattr(args, "prototype_use_adaptive_temp", True),
                use_dual_proto=getattr(args, "prototype_use_dual_proto", True),
                hard_neg_k=getattr(args, "prototype_hard_neg_k", 5),
            ).to(device)
        else:
            prototype_memory = PrototypeMemory(**proto_kw).to(device)

    # ------------------------------------------------------------------
    # Aux-loss weight expansion
    # ------------------------------------------------------------------
    if args.aux_loss:
        aux = {}
        for i in range(args.dec_layers - 1):
            aux.update({k + f"_{i}": v for k, v in weight_dict.items() if k != "loss_proto_align"})
        if args.two_stage:
            aux.update({k + "_enc": v for k, v in weight_dict.items() if k != "loss_proto_align"})
        weight_dict.update(aux)

    losses = ["labels", "boxes", "cardinality"]
    if args.segmentation_head:
        losses.append("masks")
    if use_proto:
        losses.append("prototype_align")

    criterion_kwargs = dict(
        focal_alpha=args.focal_alpha,
        losses=losses,
        group_detr=args.group_detr,
        sum_group_losses=getattr(args, "sum_group_losses", False),
        use_varifocal_loss=args.use_varifocal_loss,
        use_position_supervised_loss=args.use_position_supervised_loss,
        ia_bce_loss=args.ia_bce_loss,
        prototype_memory=prototype_memory,
        prototype_loss_coef=getattr(args, "prototype_loss_coef", 0.1),
    )
    if args.segmentation_head:
        criterion_kwargs["mask_point_sample_ratio"] = args.mask_point_sample_ratio

    criterion = SetCriterion(
        args.num_classes + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        **criterion_kwargs,
    ).to(device)

    return criterion, PostProcess(num_select=args.num_select)
