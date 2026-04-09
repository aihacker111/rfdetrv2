# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

"""
train_one_epoch — single epoch supervised training step.

Supports:
  - Gradient accumulation
  - AMP (bfloat16)
  - Multi-scale resizing
  - EMA model update
  - Drop-path / dropout schedules
  - Prototype alignment loss logging
  - Distributed training
"""

from __future__ import annotations

import logging
import math
import random
from typing import Callable, DefaultDict, Iterable, List

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F

import rfdetrv2.util.misc as utils
from rfdetrv2.datasets.coco_album import compute_multi_scale_scales
from rfdetrv2.util.misc import NestedTensor

try:
    from torch.amp import GradScaler, autocast
    _LEGACY_AMP = False
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore[no-redef]
    _LEGACY_AMP = True


def _autocast_args(args) -> dict:
    if _LEGACY_AMP:
        return {"enabled": args.amp, "dtype": torch.bfloat16}
    return {"device_type": "cuda", "enabled": args.amp, "dtype": torch.bfloat16}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0.0,
    ema_m: torch.nn.Module = None,
    schedules: dict = None,
    num_training_steps_per_epoch: int = None,
    vit_encoder_num_layers: int = None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
    postprocess=None,
) -> dict:
    """
    Run one full training epoch.

    Args:
        model:       The detection model (may be DDP-wrapped).
        criterion:   Loss function (returns a dict of loss components).
        lr_scheduler: Step-level LR scheduler — stepped once per sub-batch.
        data_loader: Iterable of ``(samples, targets)`` batches.
        optimizer:   AdamW or similar.
        device:      Target device.
        epoch:       Current epoch index (0-based).
        batch_size:  Effective batch size (``batch_size × grad_accum_steps``).
        max_norm:    Gradient clipping max norm (0 = disabled).
        ema_m:       EMA model wrapper (or None).
        schedules:   Drop schedules ``{"dp": [...], "do": [...]}``.
        num_training_steps_per_epoch: Total steps this epoch.
        vit_encoder_num_layers: Used for drop-path schedule updates.
        args:        Flat training args namespace.
        callbacks:   Dict of event → list of callables.
        postprocess: Unused; kept for API compatibility.

    Returns:
        Dict of metric averages (loss, class_error, lr, …).
    """
    if schedules is None:
        schedules = {}
    if callbacks is None:
        callbacks = DefaultDict(list)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))

    start_steps = epoch * num_training_steps_per_epoch

    scaler_kwargs = {"enabled": args.amp, "growth_interval": 100}
    scaler = GradScaler(**scaler_kwargs) if _LEGACY_AMP else GradScaler("cuda", **scaler_kwargs)

    assert batch_size % args.grad_accum_steps == 0, (
        f"batch_size ({batch_size}) must be divisible by grad_accum_steps ({args.grad_accum_steps})"
    )
    sub_batch = batch_size // args.grad_accum_steps

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, 10, f"Epoch [{epoch}]")
    ):
        it = start_steps + data_iter_step

        # --- Fire batch-start callbacks ---
        for cb in callbacks["on_train_batch_start"]:
            cb({"step": it, "model": model, "epoch": epoch})

        # --- Apply drop-path / dropout schedules ---
        _update_drop_schedules(model, schedules, it, args, vit_encoder_num_layers)

        # --- Multi-scale resize ---
        if args.multi_scale and not args.do_random_resize_via_padding:
            samples = _multi_scale_resize(samples, args, it)

        # --- Gradient accumulation loop ---
        loss_dict = None
        weight_dict = None

        for i in range(args.grad_accum_steps):
            s_start, s_end = i * sub_batch, (i + 1) * sub_batch
            sub_samples = NestedTensor(
                samples.tensors[s_start:s_end],
                samples.mask[s_start:s_end],
            ).to(device)
            sub_targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets[s_start:s_end]
            ]

            with autocast(**_autocast_args(args)):
                outputs = model(sub_samples, sub_targets)
                loss_dict = criterion(outputs, sub_targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict
                    if k in weight_dict
                )
            del outputs
            scaler.scale(losses).backward()

        # --- Optimizer step ---
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_value = sum(loss_dict_scaled.values()).item()

        if not math.isfinite(loss_value):
            logger.warning(
                "Non-finite loss (%s) at step %d — skipping batch and zeroing gradients.",
                loss_value, it,
            )
            optimizer.zero_grad()
            scaler.update()  # keep scaler state consistent
            continue

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        if ema_m is not None:
            ema_m.update(model)

        # --- Logging ---
        loss_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=loss_value, **loss_dict_scaled, **loss_unscaled)
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Superposition-aware prototype losses
        for _pk in ("loss_proto_pull", "loss_proto_ortho", "loss_proto_disambig", "loss_proto_sparse"):
            if _pk in loss_dict_reduced:
                metric_logger.update(**{_pk: loss_dict_reduced[_pk].item()})

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _update_drop_schedules(model, schedules, it, args, vit_encoder_num_layers):
    """Apply per-step drop-path and dropout schedule updates."""
    if "dp" in schedules:
        val = schedules["dp"][it]
        if args.distributed:
            model.module.update_drop_path(val, vit_encoder_num_layers)
        else:
            model.update_drop_path(val, vit_encoder_num_layers)

    if "do" in schedules:
        val = schedules["do"][it]
        if args.distributed:
            model.module.update_dropout(val)
        else:
            model.update_dropout(val)


def _multi_scale_resize(samples: NestedTensor, args, it: int) -> NestedTensor:
    """Randomly resize the batch to one of the multi-scale target resolutions."""
    scales = compute_multi_scale_scales(
        args.resolution, args.expanded_scales,
        args.patch_size, args.num_windows,
    )
    random.seed(it)
    scale = random.choice(scales)
    with torch.no_grad():
        samples.tensors = F.interpolate(
            samples.tensors, size=scale, mode="bilinear", align_corners=False
        )
        samples.mask = F.interpolate(
            samples.mask.unsqueeze(1).float(), size=scale, mode="nearest"
        ).squeeze(1).bool()
    return samples
