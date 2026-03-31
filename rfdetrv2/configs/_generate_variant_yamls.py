#!/usr/bin/env python3
"""Regenerate train_from_scratch / finetune / inference YAMLs from schemas (run from repo root)."""
from __future__ import annotations

from pathlib import Path

import yaml

VARIANTS = {
    "nano": {
        "encoder": "dinov3_nano",
        "resolution": 384,
        "num_windows": 2,
        "positional_encoding_size": 32,
    },
    "small": {
        "encoder": "dinov3_small",
        "resolution": 512,
        "num_windows": 2,
        "positional_encoding_size": 32,
    },
    "base": {
        "encoder": "dinov3_base",
        "resolution": 560,
        "num_windows": 4,
        "positional_encoding_size": 35,
    },
    "large": {
        "encoder": "dinov3_large",
        "resolution": 640,
        "num_windows": 4,
        "positional_encoding_size": 35,
    },
}

SCHEMA_CLASS = {
    "nano": "RFDETRNanoConfig",
    "small": "RFDETRSmallConfig",
    "base": "RFDETRBaseConfig",
    "large": "RFDETRLargeConfig",
}

COMMON_MODEL = {
    "vit_encoder_num_layers": 12,
    "window_block_indexes": None,
    "position_embedding": "sine",
    "out_feature_indexes": [2, 5, 8, 11],
    "freeze_encoder": False,
    "freeze_decoder": False,
    "freeze_detection_head": False,
    "layer_norm": True,
    "rms_norm": False,
    "backbone_lora": False,
    "force_no_pretrain": False,
    "dec_layers": 3,
    "dim_feedforward": 2048,
    "hidden_dim": 256,
    "sa_nheads": 8,
    "ca_nheads": 16,
    "num_queries": 300,
    "group_detr": 13,
    "two_stage": True,
    "projector_scale": ["P3", "P4", "P5"],
    "lite_refpoint_refine": True,
    "num_select": 300,
    "dec_n_points": 8,
    "decoder_norm": "LN",
    "bbox_reparam": True,
    "freeze_batch_norm": False,
    "patch_size": 16,
    "use_windowed_attn": False,
    "use_convnext_projector": True,
    "gradient_checkpointing": False,
    "segmentation_head": False,
    "mask_downsample_ratio": 4,
    "mask_point_sample_ratio": 16,
    "mask_ce_loss_coef": 5.0,
    "mask_dice_loss_coef": 5.0,
    "eval_max_dets": 500,
    "use_rope": True,
}


def _system() -> dict:
    return {
        "device": "cuda",
        "world_size": 1,
        "dist_url": "env://",
        "sync_bn": True,
        "rank": 0,
        "gpu": 0,
        "distributed": False,
        "dist_backend": "nccl",
        "subcommand": None,
        "class_names": None,
        "label_to_cat_id": None,
        "num_feature_levels": None,
        "shape": None,
        "infer_dir": None,
        "verbose": False,
        "opset_version": 17,
        "simplify": False,
        "tensorrt": False,
        "dry_run": False,
        "profile": False,
        "no_use_convnext_projector": False,
    }


def _train_base() -> dict:
    return {
        "num_classes": 90,
        "auto_infer_coco_classes": True,
        "grad_accum_steps": 1,
        "amp": False,
        "lr": 2.0e-4,
        "lr_encoder": 2.5e-5,
        "lr_scale_mode": "sqrt",
        "batch_size": 2,
        "weight_decay": 1.0e-4,
        "epochs": 12,
        "lr_drop": 11,
        "clip_max_norm": 0.1,
        "lr_vit_layer_decay": 0.8,
        "lr_component_decay": 1.0,
        "do_benchmark": False,
        "dropout": 0.0,
        "drop_path": 0.0,
        "drop_mode": "standard",
        "drop_schedule": "constant",
        "cutoff_epoch": 0,
        "pretrained_encoder": None,
        "pretrain_exclude_keys": None,
        "pretrain_keys_modify_to_load": None,
        "pretrained_distiller": None,
        "set_cost_class": 2.0,
        "set_cost_bbox": 5.0,
        "set_cost_giou": 2.0,
        "cls_loss_coef": 1.0,
        "bbox_loss_coef": 5.0,
        "giou_loss_coef": 2.0,
        "focal_alpha": 0.25,
        "aux_loss": True,
        "sum_group_losses": False,
        "use_varifocal_loss": False,
        "use_position_supervised_loss": False,
        "ia_bce_loss": True,
        "use_prototype_align": True,
        "prototype_loss_coef": 0.1,
        "prototype_momentum": 0.999,
        "prototype_warmup_steps": 200,
        "prototype_temperature": 0.1,
        "prototype_repulsion_coef": 0.1,
        "prototype_use_freq_weight": True,
        "prototype_use_quality_weight": True,
        "prototype_use_repulsion": True,
        "dataset_file": "coco",
        "coco_path": None,
        "dataset_dir": None,
        "square_resize_div_64": False,
        "dont_save_weights": False,
        "checkpoint_interval": 10,
        "seed": 42,
        "resume": "",
        "start_epoch": 0,
        "eval": False,
        "use_ema": False,
        "ema_decay": 0.9997,
        "ema_tau": 0,
        "num_workers": 2,
        "warmup_epochs": 1.0,
        "lr_scheduler": "step",
        "lr_milestones": [80, 160],
        "lr_gamma": 0.1,
        "lr_min_factor": 0.05,
        "lr_stable_ratio": 0.7,
        "lr_restart_period": 50,
        "lr_restart_decay": 0.8,
        "early_stopping": True,
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.001,
        "early_stopping_use_ema": False,
        "debug_data_limit": 0,
        "multi_scale": False,
        "expanded_scales": False,
        "do_random_resize_via_padding": False,
        "use_cls_token": False,
        "encoder_only": False,
        "backbone_only": False,
        "fp16_eval": False,
        "freeze_encoder": False,
        "freeze_decoder": False,
        "freeze_detection_head": False,
        "use_convnext_projector": True,
        "tensorboard": False,
        "wandb": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    for variant, vo in VARIANTS.items():
        model = {**COMMON_MODEL, **vo}
        sch = SCHEMA_CLASS[variant]

        # --- train from scratch ---
        train_s = _train_base()
        train_s["pretrain_weights"] = None
        train_s["output_dir"] = f"output/train_scratch_{variant}"
        doc_s = {"model": model, "train": train_s, "system": _system()}
        p_s = root / "train_from_scratch" / f"{variant}.yaml"
        hdr = (
            f"# Train from scratch — {variant} ({sch})\n"
            "# No detector checkpoint; DINOv3 backbone weights resolve via pretrained_encoder or auto-download.\n"
            "# Set train.dataset_dir (and train.num_classes) for your data.\n\n"
        )
        p_s.write_text(hdr + yaml.safe_dump(doc_s, sort_keys=False, allow_unicode=True))

        # --- finetune ---
        train_f = _train_base()
        train_f["pretrain_weights"] = f"rfdetrv2_{variant}"
        train_f["output_dir"] = f"output/finetune_{variant}"
        train_f["lr"] = 1.0e-4
        train_f["lr_encoder"] = 1.25e-5
        train_f["warmup_epochs"] = 0.5
        train_f["epochs"] = 24
        train_f["freeze_encoder"] = True
        train_f["freeze_decoder"] = False
        train_f["freeze_detection_head"] = False
        model_f = {**model, "freeze_encoder": True}
        doc_f = {"model": model_f, "train": train_f, "system": _system()}
        p_f = root / "finetune" / f"{variant}.yaml"
        hdr_f = (
            f"# Fine-tune — {variant} ({sch})\n"
            "# Loads COCO checkpoint via pretrain_weights alias (HF → rfdetr_pretrained/).\n"
            "# Tweak train.freeze_encoder / freeze_decoder / freeze_detection_head and learning rates as needed.\n"
        )
        if variant == "large":
            hdr_f += "# Large ViT-L backbone: supply pretrained_encoder or local dinov3_vitl16*.pth if missing.\n"
        hdr_f += "\n"
        p_f.write_text(hdr_f + yaml.safe_dump(doc_f, sort_keys=False, allow_unicode=True))

        # --- inference ---
        train_i = _train_base()
        train_i["pretrain_weights"] = f"rfdetrv2_{variant}"
        train_i["output_dir"] = f"output/inference_{variant}"
        train_i["batch_size"] = 1
        train_i["num_workers"] = 0
        train_i["epochs"] = 1
        train_i["eval"] = False
        train_i["early_stopping"] = False
        doc_i = {"model": model, "train": train_i, "system": _system()}
        p_i = root / "inference" / f"{variant}.yaml"
        hdr_i = (
            f"# Inference — {variant} ({sch})\n"
            "# For Pipeline(cfg=load_config(...)).predict(...); set train.num_classes to match checkpoint.\n"
            "# Optional: pipe.predict(..., save_path=\"out/vis.png\").\n"
        )
        if variant == "large":
            hdr_i += "# ViT-L: ensure DINOv3 backbone weights are available (pretrained_encoder or local .pth).\n"
        hdr_i += "\n"
        p_i.write_text(hdr_i + yaml.safe_dump(doc_i, sort_keys=False, allow_unicode=True))

    print("Wrote train_from_scratch/, finetune/, inference/ YAMLs for nano, small, base, large.")


if __name__ == "__main__":
    main()
