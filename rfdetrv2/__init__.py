# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
rfdetrv2 — RF-DETR v2 object detection library.

Model shortcuts::

    from rfdetrv2 import RFDETRV2Base, RFDETRV2Small, RFDETRV2Nano, RFDETRV2Large

``RFDETRBase`` / ``RFDETRSmall`` / … remain as aliases for backward compatibility.

Pipeline API::

    from rfdetrv2.pipeline import (
        TrainingPipeline, FinetunePipeline, InferencePipeline, ExportPipeline,
    )
"""

from __future__ import annotations

import os
from typing import Any

# Enable MPS fallback ops so macOS users don't need to set this manually.
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

__all__ = [
    "RFDETRV2",
    "RFDETRV2Base",
    "RFDETRV2Small",
    "RFDETRV2Nano",
    "RFDETRV2Large",
    # Deprecated aliases (same objects as RFDETRV2*)
    "RFDETRBase",
    "RFDETRSmall",
    "RFDETRNano",
    "RFDETRLarge",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from rfdetrv2 import detr as _detr
        return getattr(_detr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
