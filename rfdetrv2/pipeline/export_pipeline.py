# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
ExportPipeline — export a trained RF-DETR to ONNX (with optional simplification).
"""

from __future__ import annotations

import os
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Optional, Tuple

import torch

from rfdetrv2.config import ModelConfig
from rfdetrv2.pipeline.base import BasePipeline

logger = getLogger(__name__)


class ExportPipeline(BasePipeline):
    """ONNX export pipeline for RF-DETR models."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.resolution: int = model_config.resolution

    def run(
        self,
        output_dir: str = "output",
        shape: Optional[Tuple[int, int]] = None,
        batch_size: int = 1,
        backbone_only: bool = False,
        simplify: bool = False,
        force_simplify: bool = False,
        opset_version: int = 17,
        verbose: bool = True,
        infer_dir: Optional[str] = None,
    ) -> str:
        """Export the model to ONNX and return the path to the exported file."""
        self._check_export_deps()

        from rfdetrv2.deploy.export import export_onnx, make_infer_image, onnx_simplify

        if shape is None:
            shape = (self.resolution, self.resolution)
        self._validate_shape(shape)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)

        model = deepcopy(self.model).cpu()
        model.eval()

        input_tensor = make_infer_image(infer_dir, shape, batch_size, torch.device("cpu"))
        input_names = ["input"]
        output_names = ["features"] if backbone_only else ["dets", "labels"]

        self._dry_run(model, input_tensor, backbone_only)

        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensor,
            output_names=output_names,
            dynamic_axes=None,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version,
        )
        logger.info("ONNX model saved to: %s", output_file)

        if simplify:
            simplified = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensor,
                force=force_simplify,
            )
            logger.info("Simplified ONNX saved to: %s", simplified)
            return str(simplified)

        return str(output_file)

    @staticmethod
    def _check_export_deps() -> None:
        try:
            import onnx  # noqa: F401
        except ImportError:
            raise ImportError(
                "ONNX export requires additional dependencies. "
                "Install them with: pip install rfdetrv2[onnxexport]"
            )

    @staticmethod
    def _validate_shape(shape: Tuple[int, int]) -> None:
        if shape[0] % 14 != 0 or shape[1] % 14 != 0:
            raise ValueError(
                f"Export shape {shape} must be divisible by 14 (patch size of DINOv3)."
            )

    @staticmethod
    def _dry_run(model: torch.nn.Module, input_tensor: torch.Tensor, backbone_only: bool) -> None:
        """Forward pass to surface shape errors before tracing."""
        with torch.no_grad():
            out = model(input_tensor)
        if backbone_only:
            logger.info("Backbone-only export — output shape: %s", tuple(out.shape))
        else:
            logger.info(
                "Full model export — boxes: %s, logits: %s",
                tuple(out["pred_boxes"].shape),
                tuple(out["pred_logits"].shape),
            )
