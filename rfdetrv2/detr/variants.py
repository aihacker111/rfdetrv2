# ------------------------------------------------------------------------
# RF-DETR — preset wrapper classes (Nano / Small / Base / Large).
# ------------------------------------------------------------------------

from rfdetrv2.detr.core import RFDETRV2
from rfdetrv2.schemas import (
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    TrainConfig,
)


class RFDETRBase(RFDETRV2):
    """
    Train an RF-DETR Base model (29M parameters).
    """
    size = "rfdetr-base"
    def get_model_config(self, **kwargs):
        return RFDETRBaseConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRNano(RFDETRV2):
    """
    RF-DETR Nano: DINOv3 ViT-S (dinov3_vits16, 21M).
    """
    size = "rfdetr-nano"

    def get_model_config(self, **kwargs):
        return RFDETRNanoConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRSmall(RFDETRV2):
    """
    RF-DETR Small: DINOv3 ViT-S+ (dinov3_vits16plus, 29M).
    """
    size = "rfdetr-small"

    def get_model_config(self, **kwargs):
        return RFDETRSmallConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETRLarge(RFDETRV2):
    size = "rfdetr-large"
    def get_model_config(self, **kwargs):
        return RFDETRLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)
