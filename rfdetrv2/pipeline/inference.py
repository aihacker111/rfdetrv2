# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
InferencePipeline — run object detection on images.

Usage:
    from rfdetrv2.pipeline import InferencePipeline
    from rfdetrv2.config import RFDETRV2BaseConfig

    pipeline = InferencePipeline.from_checkpoint(
        checkpoint_path="output/checkpoint_best_total.pth",
        model_config=RFDETRV2BaseConfig(),
    )

    detections = pipeline.run("photo.jpg", threshold=0.5)
"""

from __future__ import annotations

from copy import deepcopy
from logging import getLogger
from typing import List, Optional, Union

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from rfdetrv2.config import ModelConfig
from rfdetrv2.models import PostProcess
from rfdetrv2.pipeline.base import BasePipeline

logger = getLogger(__name__)

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

ImageInput = Union[str, Image.Image, np.ndarray, torch.Tensor]


class InferencePipeline(BasePipeline):
    """
    Inference-only pipeline for RF-DETR object detection.

    Accepts a wide variety of image inputs, handles batch preprocessing,
    runs the model forward pass, and returns ``supervision.Detections``.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.resolution: int = model_config.resolution
        self.postprocess = PostProcess(num_select=getattr(model_config, "num_select", 300))
        self.class_names: dict = {}

        self._optimised_model: Optional[torch.nn.Module] = None
        self._opt_resolution: Optional[int] = None
        self._opt_batch_size: Optional[int] = None
        self._opt_dtype: Optional[torch.dtype] = None
        self._is_compiled: bool = False

        self.model.eval()

    def run(
        self,
        images: Union[ImageInput, List[ImageInput]],
        threshold: float = 0.5,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """Run object detection on one or more images."""
        single = not isinstance(images, list)
        if single:
            images = [images]

        batch, orig_sizes = self._preprocess(images)

        with torch.no_grad():
            predictions = self._forward(batch)

        target_sizes = torch.tensor(orig_sizes, device=self.device)
        results = self.postprocess(predictions, target_sizes=target_sizes)

        detections = [self._to_detections(r, threshold) for r in results]
        return detections[0] if single else detections

    def optimize(
        self,
        compile: bool = True,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Optimise the model for repeated inference (optional JIT trace)."""
        self._remove_optimised()

        self._optimised_model = deepcopy(self.model)
        self._optimised_model.eval()
        self._optimised_model.export()
        self._optimised_model = self._optimised_model.to(dtype=dtype)

        self._opt_resolution = self.resolution
        self._opt_dtype = dtype

        if compile:
            dummy = torch.randn(
                batch_size, 3, self.resolution, self.resolution,
                device=self.device, dtype=dtype,
            )
            self._optimised_model = torch.jit.trace(self._optimised_model, dummy)
            self._is_compiled = True
            self._opt_batch_size = batch_size
            logger.info(
                "Model compiled for batch_size=%d, dtype=%s, resolution=%d.",
                batch_size, dtype, self.resolution,
            )
        else:
            logger.info("Model optimised (not compiled).")

    def remove_optimization(self) -> None:
        """Discard the optimised model and revert to the original."""
        self._remove_optimised()

    def _preprocess(self, images: List[ImageInput]):
        """Decode, normalise, and resize all images; return (batch_tensor, orig_sizes)."""
        processed, orig_sizes = [], []

        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            if isinstance(img, (Image.Image, np.ndarray)):
                img = TF.to_tensor(img)

            if not isinstance(img, torch.Tensor):
                raise TypeError(f"Unsupported image type: {type(img)}")
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(
                    f"Expected image tensor of shape (3, H, W), got {tuple(img.shape)}."
                )
            if img.max() > 1.0:
                raise ValueError(
                    "Image has pixel values > 1.  Please normalise to [0, 1] first."
                )

            orig_sizes.append((img.shape[1], img.shape[2]))

            img = img.to(self.device)
            img = TF.normalize(img, _MEAN, _STD)
            img = TF.resize(img, [self.resolution, self.resolution])
            processed.append(img)

        return torch.stack(processed), orig_sizes

    def _forward(self, batch: torch.Tensor) -> dict:
        """Run the forward pass, returning a predictions dict."""
        if self._optimised_model is not None:
            self._check_optimised_compatibility(batch)
            raw = self._optimised_model(batch.to(dtype=self._opt_dtype))
        else:
            raw = self.model(batch)

        if isinstance(raw, tuple):
            out = {"pred_logits": raw[1], "pred_boxes": raw[0]}
            if len(raw) == 3:
                out["pred_masks"] = raw[2]
            return out
        return raw

    def _to_detections(self, result: dict, threshold: float) -> sv.Detections:
        """Filter by threshold and wrap in ``supervision.Detections``."""
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]
        keep = scores > threshold

        kwargs = dict(
            xyxy=boxes[keep].float().cpu().numpy(),
            confidence=scores[keep].float().cpu().numpy(),
            class_id=labels[keep].cpu().numpy(),
        )

        if "masks" in result:
            masks = result["masks"][keep]
            kwargs["mask"] = masks.squeeze(1).cpu().numpy()

        return sv.Detections(**kwargs)

    def _check_optimised_compatibility(self, batch: torch.Tensor) -> None:
        if self._opt_resolution != batch.shape[2]:
            raise ValueError(
                f"Resolution mismatch: optimised for {self._opt_resolution}, "
                f"got {batch.shape[2]}.  Call remove_optimization() first."
            )
        if self._is_compiled and self._opt_batch_size != batch.shape[0]:
            raise ValueError(
                f"Batch size mismatch: compiled for {self._opt_batch_size}, "
                f"got {batch.shape[0]}.  Re-optimise with the correct batch size."
            )

    def _remove_optimised(self) -> None:
        self._optimised_model = None
        self._opt_resolution = None
        self._opt_batch_size = None
        self._opt_dtype = None
        self._is_compiled = False
