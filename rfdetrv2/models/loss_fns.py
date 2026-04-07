# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Detection loss functions.

Exports
-------
sigmoid_focal_loss         — standard focal loss
sigmoid_varifocal_loss     — VariFocal loss (quality-aware)
position_supervised_loss   — position-supervised focal loss
dice_loss / dice_loss_jit  — dice loss for segmentation
sigmoid_ce_loss / sigmoid_ce_loss_jit — BCE for segmentation masks
"""

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t     = prob * targets + (1 - prob) * (1 - targets)
    loss    = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss    = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def sigmoid_varifocal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    prob         = inputs.sigmoid()
    focal_weight = (
        targets * (targets > 0.0).float()
        + (1 - alpha) * (prob - targets).abs().pow(gamma) * (targets <= 0.0).float()
    )
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return (ce_loss * focal_weight).mean(1).sum() / num_boxes


def position_supervised_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: float,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    prob    = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss    = ce_loss * (torch.abs(targets - prob) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * (targets > 0.0).float() + (1 - alpha) * (targets <= 0.0).float()
        loss    = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
) -> torch.Tensor:
    inputs      = inputs.sigmoid().flatten(1)
    numerator   = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss        = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)
