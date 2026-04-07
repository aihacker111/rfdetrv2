# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Backward-compatibility shim.

All implementation has been moved to focused sub-modules:
  rfdetrv2.models.prototype_memory  → PrototypeMemory
  rfdetrv2.models.criterion         → SetCriterion
  rfdetrv2.models.detector          → LWDETR, PostProcess, MLP
  rfdetrv2.models.loss_fns          → sigmoid_focal_loss, …
  rfdetrv2.models.builder           → build_model, build_criterion_and_postprocessors

Import from those modules directly in new code.
"""

from rfdetrv2.models.builder import build_criterion_and_postprocessors, build_model
from rfdetrv2.models.criterion import SetCriterion
from rfdetrv2.models.detector import LWDETR, MLP, PostProcess
from rfdetrv2.models.loss_fns import (
    dice_loss,
    dice_loss_jit,
    position_supervised_loss,
    sigmoid_ce_loss,
    sigmoid_ce_loss_jit,
    sigmoid_focal_loss,
    sigmoid_varifocal_loss,
)
from rfdetrv2.models.prototype_memory import PrototypeMemory

__all__ = [
    "PrototypeMemory",
    "LWDETR",
    "SetCriterion",
    "PostProcess",
    "MLP",
    "sigmoid_focal_loss",
    "sigmoid_varifocal_loss",
    "position_supervised_loss",
    "dice_loss",
    "dice_loss_jit",
    "sigmoid_ce_loss",
    "sigmoid_ce_loss_jit",
    "build_model",
    "build_criterion_and_postprocessors",
]
