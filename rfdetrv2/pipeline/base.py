# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
Base pipeline abstraction.

All pipelines (train, finetune, inference, export) inherit from BasePipeline,
which owns the model lifecycle: build → load weights → device placement.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from logging import getLogger
from typing import Optional

import torch

from rfdetrv2.config import ModelConfig
from rfdetrv2.core.model import build_rf_model

logger = getLogger(__name__)


class BasePipeline(ABC):
    """
    Abstract base for all RF-DETR pipelines.

    Sub-classes only need to implement ``run()``.  Everything else —
    model construction, weight loading, device placement — is handled here.
    """

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.device = torch.device(model_config.device)

        logger.info("Building model with config: %s", model_config)
        self.model = build_rf_model(model_config)
        self.model.to(self.device)

        if model_config.pretrain_weights is not None:
            self._load_pretrain_weights(model_config.pretrain_weights)

    # ------------------------------------------------------------------
    # Class-method constructors  (convenient entry points)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, model_config: ModelConfig, **kwargs) -> "BasePipeline":
        """Build a pipeline directly from a ``ModelConfig`` object."""
        return cls(model_config, **kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_config: ModelConfig, **kwargs) -> "BasePipeline":
        """
        Build a pipeline and restore weights from a training checkpoint.

        The checkpoint must have been saved by this codebase (keys: ``model``, ``args``).
        """
        pipeline = cls(model_config, **kwargs)
        pipeline._load_checkpoint(checkpoint_path)
        return pipeline

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, **kwargs):
        """Execute the pipeline (train / evaluate / predict / export)."""
        ...

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _load_pretrain_weights(self, path: str) -> None:
        """Load pretrained weights, handling class-head size mismatches."""
        logger.info("Loading pretrained weights from: %s", path)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pretrained weights not found: {path}. "
                "Pass the correct path via model_config.pretrain_weights."
            )

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        # Handle class-head size mismatch gracefully
        head_key = "class_embed.bias"
        if head_key in state:
            ckpt_num_classes = state[head_key].shape[0]
            model_num_classes = self.model_config.num_classes + 1
            if ckpt_num_classes != model_num_classes:
                logger.warning(
                    "Class-head mismatch: checkpoint has %d classes, model expects %d. "
                    "Re-initialising detection head.",
                    ckpt_num_classes,
                    model_num_classes,
                )
                self.model.reinitialize_detection_head(ckpt_num_classes)

        # Clip query embeddings to the expected size
        num_desired_queries = getattr(self.model_config, "num_queries", 300) * getattr(
            self.model_config, "group_detr", 13
        )
        for name in list(state.keys()):
            if name.endswith(("refpoint_embed.weight", "query_feat.weight")):
                state[name] = state[name][:num_desired_queries]

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.debug("Missing keys: %s", missing)
        if unexpected:
            logger.debug("Unexpected keys: %s", unexpected)

        logger.info("Pretrained weights loaded successfully.")

    def _load_checkpoint(self, path: str) -> None:
        """Restore a full training checkpoint (model weights + optional metadata)."""
        logger.info("Restoring checkpoint from: %s", path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        self.model.load_state_dict(state, strict=True)

        # Restore class names if saved
        if "args" in ckpt and hasattr(ckpt["args"], "class_names"):
            self.class_names = ckpt["args"].class_names

        logger.info("Checkpoint restored (epoch %s).", ckpt.get("epoch", "?"))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze the DINOv3 backbone — no gradient updates on the encoder."""
        backbone = self.model.backbone
        if hasattr(backbone, "__getitem__") and len(backbone) > 0:
            encoder = getattr(backbone[0], "encoder", None)
            if encoder is not None:
                for p in encoder.parameters():
                    p.requires_grad_(False)
                logger.info("DINOv3 encoder frozen.")

    def unfreeze_encoder(self) -> None:
        """Unfreeze the DINOv3 backbone."""
        backbone = self.model.backbone
        if hasattr(backbone, "__getitem__") and len(backbone) > 0:
            encoder = getattr(backbone[0], "encoder", None)
            if encoder is not None:
                for p in encoder.parameters():
                    p.requires_grad_(True)
                logger.info("DINOv3 encoder unfrozen.")
