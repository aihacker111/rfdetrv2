# ------------------------------------------------------------------------
# Pydantic configuration presets (model + train).
# ------------------------------------------------------------------------

from rfdetrv2.schemas.model import (
    DEVICE,
    ModelConfig,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
)
from rfdetrv2.schemas.train import SegmentationTrainConfig, TrainConfig

__all__ = [
    "DEVICE",
    "ModelConfig",
    "RFDETRBaseConfig",
    "RFDETRLargeConfig",
    "RFDETRNanoConfig",
    "RFDETRSmallConfig",
    "TrainConfig",
    "SegmentationTrainConfig",
]
