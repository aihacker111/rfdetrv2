# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from rfdetrv2.detr import (
    RFDETRV2,
    RFDETRBase,
    RFDETRLarge,
    RFDETRNano,
    RFDETRSmall,
)
from rfdetrv2.runner import (
    HOSTED_MODELS,
    Pipeline,
    download_pretrain_weights,
    evaluate,
    load_config,
    predict_detections,
    resolve_pretrain_weights_path,
    train_one_epoch,
)

__all__ = [
    "HOSTED_MODELS",
    "Pipeline",
    "RFDETRBase",
    "RFDETRLarge",
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRV2",
    "download_pretrain_weights",
    "evaluate",
    "load_config",
    "predict_detections",
    "resolve_pretrain_weights_path",
    "train_one_epoch",
]
