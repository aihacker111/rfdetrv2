# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
build_rf_model — construct the full RF-DETR detection model from a ModelConfig.

Kept as a standalone function (not a class) so it's easy to mock in tests
and to call from any pipeline without importing the heavier ``main.py``.
"""

from __future__ import annotations

from rfdetrv2.config import ModelConfig, pydantic_dump
from rfdetrv2.main import populate_args
from rfdetrv2.models import build_model


def build_rf_model(model_config: ModelConfig):
    """
    Build and return the RF-DETR detection model.

    Converts a typed ``ModelConfig`` Pydantic model into the flat
    ``argparse.Namespace`` expected by the legacy ``build_model`` function,
    then builds and returns the ``nn.Module``.

    Args:
        model_config: A ``ModelConfig`` (or subclass) instance.

    Returns:
        The constructed ``nn.Module`` (on CPU, not yet moved to device).
    """
    args = populate_args(**pydantic_dump(model_config))
    return build_model(args)
