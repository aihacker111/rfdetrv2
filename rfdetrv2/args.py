# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Argument definitions for RF-DETR training / evaluation.

Exports
-------
get_args_parser()  — argparse.ArgumentParser for CLI usage.
populate_args()    — construct argparse.Namespace from keyword arguments
                     (used by the Python API / pipelines).
"""

from __future__ import annotations

import argparse
import ast


def get_args_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the LWDETR training / eval CLI."""
    parser = argparse.ArgumentParser("LWDETR training and evaluation script", add_help=False)

    # Core training
    parser.add_argument("--num_classes",        default=2,      type=int)
    parser.add_argument("--grad_accum_steps",   default=1,      type=int)
    parser.add_argument("--amp",                default=False,  type=bool)
    parser.add_argument("--lr",                 default=2e-4,   type=float)
    parser.add_argument("--lr_encoder",         default=2.5e-5, type=float)
    parser.add_argument("--lr_scale_mode",      default="sqrt", type=str,
                        choices=["linear", "sqrt"])
    parser.add_argument("--batch_size",         default=2,      type=int)
    parser.add_argument("--weight_decay",       default=1e-4,   type=float)
    parser.add_argument("--epochs",             default=12,     type=int)
    parser.add_argument("--lr_drop",            default=11,     type=int)
    parser.add_argument("--clip_max_norm",      default=0.1,    type=float)
    parser.add_argument("--lr_vit_layer_decay", default=0.8,    type=float)
    parser.add_argument("--lr_component_decay", default=1.0,    type=float)
    parser.add_argument("--do_benchmark",       action="store_true")

    # Drop regularisation
    parser.add_argument("--dropout",       type=float, default=0)
    parser.add_argument("--drop_path",     type=float, default=0)
    parser.add_argument("--drop_mode",     type=str,   default="standard",
                        choices=["standard", "early", "late"])
    parser.add_argument("--drop_schedule", type=str,   default="constant",
                        choices=["constant", "linear"])
    parser.add_argument("--cutoff_epoch",  type=int,   default=0)

    # Pretrained weights
    parser.add_argument("--pretrained_encoder",           type=str, default=None)
    parser.add_argument("--pretrain_weights",             type=str, default=None)
    parser.add_argument("--pretrain_exclude_keys",        type=str, default=None, nargs="+")
    parser.add_argument("--pretrain_keys_modify_to_load", type=str, default=None, nargs="+")

    # Backbone
    parser.add_argument("--encoder",               default="vit_tiny", type=str)
    parser.add_argument("--vit_encoder_num_layers", default=12,         type=int)
    parser.add_argument("--window_block_indexes",  default=None,        type=int, nargs="+")
    parser.add_argument("--position_embedding",    default="sine",      type=str,
                        choices=("sine", "learned"))
    parser.add_argument("--out_feature_indexes",   default=[-1],        type=int, nargs="+")
    parser.add_argument("--freeze_encoder",        action="store_true", dest="freeze_encoder")
    parser.add_argument("--layer_norm",            action="store_true", dest="layer_norm")
    parser.add_argument("--rms_norm",              action="store_true", dest="rms_norm")
    parser.add_argument("--backbone_lora",         action="store_true", dest="backbone_lora")
    parser.add_argument("--force_no_pretrain",     action="store_true", dest="force_no_pretrain")

    # Transformer / decoder
    parser.add_argument("--dec_layers",           default=3,    type=int)
    parser.add_argument("--dim_feedforward",      default=2048, type=int)
    parser.add_argument("--hidden_dim",           default=256,  type=int)
    parser.add_argument("--sa_nheads",            default=8,    type=int)
    parser.add_argument("--ca_nheads",            default=8,    type=int)
    parser.add_argument("--num_queries",          default=300,  type=int)
    parser.add_argument("--group_detr",           default=13,   type=int)
    parser.add_argument("--two_stage",            action="store_true")
    parser.add_argument("--projector_scale",      default=["P3", "P4", "P5"], type=str, nargs="+",
                        choices=("P3", "P4", "P5", "P6"))
    parser.add_argument("--lite_refpoint_refine", action="store_true")
    parser.add_argument("--num_select",           default=100,  type=int)
    parser.add_argument("--dec_n_points",         default=8,    type=int)
    parser.add_argument("--decoder_norm",         default="LN", type=str)
    parser.add_argument("--bbox_reparam",         action="store_true")
    parser.add_argument("--freeze_batch_norm",    action="store_true")

    # Matcher
    parser.add_argument("--set_cost_class", default=2, type=float)
    parser.add_argument("--set_cost_bbox",  default=5, type=float)
    parser.add_argument("--set_cost_giou",  default=2, type=float)

    # Loss coefficients
    parser.add_argument("--cls_loss_coef",                  default=1.0,  type=float)
    parser.add_argument("--bbox_loss_coef",                 default=5.0,  type=float)
    parser.add_argument("--giou_loss_coef",                 default=2.0,  type=float)
    parser.add_argument("--focal_alpha",                    default=0.25, type=float)
    parser.add_argument("--no_aux_loss",    dest="aux_loss", action="store_false")
    parser.add_argument("--sum_group_losses",               action="store_true")
    parser.add_argument("--use_varifocal_loss",             action="store_true")
    parser.add_argument("--use_position_supervised_loss",   action="store_true")
    parser.add_argument("--ia_bce_loss",                    action="store_true")
    parser.add_argument("--no_convnext_projector",          action="store_true",
                        dest="no_use_convnext_projector")

    # Prototype alignment
    parser.add_argument("--no_prototype_align",            action="store_false", dest="use_prototype_align")
    parser.add_argument("--prototype_loss_coef",           default=0.1, type=float)
    parser.add_argument("--prototype_momentum",            default=0.999, type=float)
    parser.add_argument("--prototype_warmup_steps",        default=200, type=int)
    parser.add_argument("--prototype_temperature",         default=0.1, type=float)
    parser.add_argument("--prototype_repulsion_coef",      default=0.1, type=float)
    parser.add_argument("--no_prototype_use_freq_weight",  action="store_false",
                        dest="prototype_use_freq_weight")
    parser.add_argument("--no_prototype_use_quality_weight", action="store_false",
                        dest="prototype_use_quality_weight")
    parser.add_argument("--no_prototype_use_repulsion",    action="store_false",
                        dest="prototype_use_repulsion")

    # CPFE — Cortical Perceptual Feature Enhancement
    parser.add_argument("--no_cpfe",         action="store_false", dest="use_cpfe")
    parser.add_argument("--no_cpfe_sdg",     action="store_false", dest="cpfe_use_sdg")
    parser.add_argument("--no_cpfe_dn",      action="store_false", dest="cpfe_use_dn")
    parser.add_argument("--no_cpfe_tpr",     action="store_false", dest="cpfe_use_tpr")

    # LW-DETR++ — virtual FPN neck, scale-aware RoPE, enhanced prototype memory
    parser.add_argument("--use_virtual_fpn_projector", action="store_true")
    parser.add_argument("--use_scale_aware_rope",      action="store_true")
    parser.add_argument("--enhanced_prototype_memory", action="store_true")
    parser.add_argument("--prototype_repulsion_margin", default=0.0, type=float)
    parser.add_argument(
        "--no_prototype_use_adaptive_temp",
        action="store_false",
        dest="prototype_use_adaptive_temp",
        default=True,
    )
    parser.add_argument(
        "--no_prototype_use_dual_proto",
        action="store_false",
        dest="prototype_use_dual_proto",
        default=True,
    )
    parser.add_argument("--prototype_hard_neg_k", default=5, type=int)

    # Dataset
    parser.add_argument("--dataset_file",         default="coco")
    parser.add_argument("--coco_path",            type=str)
    parser.add_argument("--dataset_dir",          type=str)
    parser.add_argument("--square_resize_div_64", action="store_true")

    # Output / checkpointing
    parser.add_argument("--output_dir",          default="output")
    parser.add_argument("--dont_save_weights",   action="store_true")
    parser.add_argument("--checkpoint_interval", default=10,     type=int)
    parser.add_argument("--seed",                default=42,     type=int)
    parser.add_argument("--resume",              default="")
    parser.add_argument("--start_epoch",         default=0,      type=int, metavar="N")
    parser.add_argument("--eval",                action="store_true")
    parser.add_argument("--use_ema",             action="store_true")
    parser.add_argument("--ema_decay",           default=0.9997, type=float)
    parser.add_argument("--ema_tau",             default=0,      type=float)
    parser.add_argument("--num_workers",         default=2,      type=int)

    # Distributed
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--world_size", default=1,       type=int)
    parser.add_argument("--dist_url",   default="env://")
    parser.add_argument("--sync_bn",    default=True,    type=bool)

    # FP16
    parser.add_argument("--fp16_eval",  default=False, action="store_true")

    # Custom / misc
    parser.add_argument("--encoder_only",               action="store_true")
    parser.add_argument("--backbone_only",              action="store_true")
    parser.add_argument("--resolution",                 type=int, default=640)
    parser.add_argument("--use_cls_token",              action="store_true")
    parser.add_argument("--multi_scale",                action="store_true")
    parser.add_argument("--expanded_scales",            action="store_true")
    parser.add_argument("--do_random_resize_via_padding", action="store_true")
    parser.add_argument("--document_aug",               action="store_true")
    parser.add_argument("--warmup_epochs",              default=1,        type=float)
    parser.add_argument("--lr_scheduler",               default="linear",
                        choices=["step", "cosine", "linear", "multistep", "wsd", "cosine_restart"])
    parser.add_argument("--lr_milestones",              default=[80, 160], type=int, nargs="+")
    parser.add_argument("--lr_gamma",                   default=0.1,      type=float)
    parser.add_argument("--lr_min_factor",              default=0.05,     type=float)
    parser.add_argument("--lr_stable_ratio",            default=0.7,      type=float)
    parser.add_argument("--lr_restart_period",          default=50,       type=int)
    parser.add_argument("--lr_restart_decay",           default=0.8,      type=float)

    # Early stopping
    parser.add_argument("--early_stopping",            action="store_true")
    parser.add_argument("--early_stopping_patience",   default=10,    type=int)
    parser.add_argument("--early_stopping_min_delta",  default=0.001, type=float)
    parser.add_argument("--early_stopping_use_ema",    action="store_true")
    parser.add_argument("--debug_data_limit",          default=0,     type=int)

    # Sub-commands (ONNX export)
    subparsers = parser.add_subparsers(title="sub-commands", dest="subcommand")
    exp = subparsers.add_parser("export_model", help="LWDETR model export")
    exp.add_argument("--infer_dir",     type=str,              default=None)
    exp.add_argument("--verbose",       type=ast.literal_eval, default=False, nargs="?", const=True)
    exp.add_argument("--opset_version", type=int,              default=17)
    exp.add_argument("--simplify",      action="store_true")
    exp.add_argument("--tensorrt", "--trtexec", "--trt", action="store_true")
    exp.add_argument("--dry-run", "--test", "-t", action="store_true")
    exp.add_argument("--profile",       action="store_true")
    exp.add_argument("--shape",         type=int, nargs=2,    default=(640, 640))

    return parser


def populate_args(
    # --- Basic training ---
    num_classes=2,
    grad_accum_steps=1,
    amp=False,
    lr=2e-4,
    lr_encoder=2.5e-5,
    lr_scale_mode="sqrt",
    batch_size=2,
    weight_decay=1e-4,
    epochs=12,
    lr_drop=11,
    clip_max_norm=0.1,
    lr_vit_layer_decay=0.8,
    lr_component_decay=1.0,
    do_benchmark=False,
    # --- Drop regularisation ---
    dropout=0,
    drop_path=0,
    drop_mode="standard",
    drop_schedule="constant",
    cutoff_epoch=0,
    # --- Weights ---
    pretrained_encoder=None,
    pretrain_weights=None,
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
    pretrained_distiller=None,
    # --- Backbone ---
    encoder="vit_tiny",
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    position_embedding="sine",
    out_feature_indexes=None,
    freeze_encoder=False,
    layer_norm=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    # --- Transformer ---
    dec_layers=3,
    dim_feedforward=2048,
    hidden_dim=256,
    sa_nheads=8,
    ca_nheads=8,
    num_queries=300,
    group_detr=13,
    two_stage=False,
    projector_scale=None,
    lite_refpoint_refine=False,
    num_select=100,
    dec_n_points=8,
    decoder_norm="LN",
    bbox_reparam=False,
    freeze_batch_norm=False,
    # --- Matcher ---
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    # --- Loss ---
    cls_loss_coef=1.0,
    bbox_loss_coef=5.0,
    giou_loss_coef=2.0,
    focal_alpha=0.25,
    aux_loss=True,
    sum_group_losses=False,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    ia_bce_loss=False,
    # --- Dataset ---
    dataset_file="coco",
    coco_path=None,
    dataset_dir=None,
    square_resize_div_64=False,
    # --- Output ---
    output_dir="output",
    dont_save_weights=False,
    checkpoint_interval=10,
    seed=42,
    resume="",
    start_epoch=0,
    eval=False,
    use_ema=False,
    ema_decay=0.9997,
    ema_tau=0,
    num_workers=2,
    # --- Distributed ---
    device="cuda",
    world_size=1,
    dist_url="env://",
    sync_bn=True,
    # --- FP16 ---
    fp16_eval=False,
    # --- Custom ---
    encoder_only=False,
    backbone_only=False,
    resolution=640,
    use_cls_token=False,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
    document_aug=False,
    warmup_epochs=1,
    lr_scheduler="step",
    lr_milestones=None,
    lr_gamma=0.1,
    lr_min_factor=0.05,
    lr_stable_ratio=0.7,
    lr_restart_period=50,
    lr_restart_decay=0.8,
    # --- Early stopping ---
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=False,
    debug_data_limit=0,
    gradient_checkpointing=False,
    use_windowed_attn=False,
    use_convnext_projector=True,
    # --- Prototype alignment ---
    use_prototype_align=True,
    prototype_loss_coef=0.1,
    prototype_momentum=0.999,
    prototype_warmup_steps=200,
    prototype_temperature=0.1,
    prototype_repulsion_coef=0.1,
    prototype_use_freq_weight=True,
    prototype_use_quality_weight=True,
    prototype_use_repulsion=True,
    # --- CPFE ---
    use_cpfe=True,
    cpfe_use_sdg=True,
    cpfe_use_dn=True,
    cpfe_use_tpr=True,
    # --- LW-DETR++ ---
    use_virtual_fpn_projector=False,
    use_scale_aware_rope=False,
    enhanced_prototype_memory=False,
    prototype_repulsion_margin=0.0,
    prototype_use_adaptive_temp=True,
    prototype_use_dual_proto=True,
    prototype_hard_neg_k=5,
    # --- Misc ---
    subcommand=None,
    **extra_kwargs,
) -> argparse.Namespace:
    """Construct an ``argparse.Namespace`` from keyword arguments.

    This is the primary entry point for the Python API.  All positional CLI
    arguments map to keyword parameters here with their defaults.

    ``extra_kwargs`` are forwarded as-is (allows downstream callers to add
    arbitrary fields without breaking the signature).
    """
    # Handle the argparse flag alias from get_args_parser
    if extra_kwargs.pop("no_use_convnext_projector", None) is True:
        use_convnext_projector = False

    # Apply defaults that are mutable
    if out_feature_indexes is None:
        out_feature_indexes = [-1]
    if projector_scale is None:
        projector_scale = ["P3", "P4", "P5"]
    if lr_milestones is None:
        lr_milestones = [80, 160]

    return argparse.Namespace(
        num_classes=num_classes,
        grad_accum_steps=grad_accum_steps,
        amp=amp,
        lr=lr,
        lr_encoder=lr_encoder,
        lr_scale_mode=lr_scale_mode,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_drop=lr_drop,
        clip_max_norm=clip_max_norm,
        lr_vit_layer_decay=lr_vit_layer_decay,
        lr_component_decay=lr_component_decay,
        do_benchmark=do_benchmark,
        dropout=dropout,
        drop_path=drop_path,
        drop_mode=drop_mode,
        drop_schedule=drop_schedule,
        cutoff_epoch=cutoff_epoch,
        pretrained_encoder=pretrained_encoder,
        pretrain_weights=pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        pretrain_keys_modify_to_load=pretrain_keys_modify_to_load,
        pretrained_distiller=pretrained_distiller,
        encoder=encoder,
        vit_encoder_num_layers=vit_encoder_num_layers,
        window_block_indexes=window_block_indexes,
        position_embedding=position_embedding,
        out_feature_indexes=out_feature_indexes,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        force_no_pretrain=force_no_pretrain,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        projector_scale=projector_scale,
        lite_refpoint_refine=lite_refpoint_refine,
        num_select=num_select,
        dec_n_points=dec_n_points,
        decoder_norm=decoder_norm,
        bbox_reparam=bbox_reparam,
        freeze_batch_norm=freeze_batch_norm,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        focal_alpha=focal_alpha,
        aux_loss=aux_loss,
        sum_group_losses=sum_group_losses,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
        ia_bce_loss=ia_bce_loss,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        output_dir=output_dir,
        dont_save_weights=dont_save_weights,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        resume=resume,
        start_epoch=start_epoch,
        eval=eval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        num_workers=num_workers,
        device=device,
        world_size=world_size,
        dist_url=dist_url,
        sync_bn=sync_bn,
        fp16_eval=fp16_eval,
        encoder_only=encoder_only,
        backbone_only=backbone_only,
        resolution=resolution,
        use_cls_token=use_cls_token,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        do_random_resize_via_padding=do_random_resize_via_padding,
        document_aug=document_aug,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_milestones=lr_milestones,
        lr_gamma=lr_gamma,
        lr_min_factor=lr_min_factor,
        lr_stable_ratio=lr_stable_ratio,
        lr_restart_period=lr_restart_period,
        lr_restart_decay=lr_restart_decay,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        debug_data_limit=debug_data_limit,
        gradient_checkpointing=gradient_checkpointing,
        use_windowed_attn=use_windowed_attn,
        use_convnext_projector=use_convnext_projector,
        use_prototype_align=use_prototype_align,
        prototype_loss_coef=prototype_loss_coef,
        prototype_momentum=prototype_momentum,
        prototype_warmup_steps=prototype_warmup_steps,
        prototype_temperature=prototype_temperature,
        prototype_repulsion_coef=prototype_repulsion_coef,
        prototype_use_freq_weight=prototype_use_freq_weight,
        prototype_use_quality_weight=prototype_use_quality_weight,
        prototype_use_repulsion=prototype_use_repulsion,
        use_cpfe=use_cpfe,
        cpfe_use_sdg=cpfe_use_sdg,
        cpfe_use_dn=cpfe_use_dn,
        cpfe_use_tpr=cpfe_use_tpr,
        use_virtual_fpn_projector=use_virtual_fpn_projector,
        use_scale_aware_rope=use_scale_aware_rope,
        enhanced_prototype_memory=enhanced_prototype_memory,
        prototype_repulsion_margin=prototype_repulsion_margin,
        prototype_use_adaptive_temp=prototype_use_adaptive_temp,
        prototype_use_dual_proto=prototype_use_dual_proto,
        prototype_hard_neg_k=prototype_hard_neg_k,
        subcommand=subcommand,
        **extra_kwargs,
    )
