# ------------------------------------------------------------------------
# Shared detection inference (used by ``Pipeline.predict`` and ``RFDETRV2.predict``).
# ------------------------------------------------------------------------

from __future__ import annotations

from logging import getLogger
from typing import List, Sequence, Union

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
from PIL import Image

logger = getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def predict_detections(
    run: object,
    images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
    threshold: float = 0.5,
    *,
    means: Sequence[float] | None = None,
    stds: Sequence[float] | None = None,
) -> Union[sv.Detections, List[sv.Detections]]:
    """
    Run batched detection on one or many images.

    ``run`` must provide: ``model``, ``device``, ``resolution``, ``postprocess``,
    optional ``inference_model``, and optimization flags (see ``Pipeline``).
    """
    means_l = list(means) if means is not None else list(getattr(run, "imagenet_mean", IMAGENET_MEAN))
    stds_l = list(stds) if stds is not None else list(getattr(run, "imagenet_std", IMAGENET_STD))

    if not getattr(run, "_is_optimized_for_inference", False) and not getattr(
        run, "_has_warned_about_not_being_optimized_for_inference", False
    ):
        logger.warning(
            "Model is not optimized for inference. "
            "Latency may be higher than expected. "
            "Call optimize_for_inference() on the training pipeline to trace/compile."
        )
        run._has_warned_about_not_being_optimized_for_inference = True
        run.model.eval()

    if not isinstance(images, list):
        images = [images]

    orig_sizes: list[tuple[int, int]] = []
    processed_images: list[torch.Tensor] = []

    for img in images:
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)

        if (img > 1).any():
            raise ValueError(
                "Image has pixel values above 1. Ensure the image is normalized to [0, 1]."
            )
        if img.shape[0] != 3:
            raise ValueError(f"Expected 3 RGB channels, got {img.shape[0]}.")

        h, w = int(img.shape[1]), int(img.shape[2])
        orig_sizes.append((h, w))

        img_tensor = img.to(run.device)
        img_tensor = F.normalize(img_tensor, means_l, stds_l)
        img_tensor = F.resize(img_tensor, (run.resolution, run.resolution))
        processed_images.append(img_tensor)

    batch_tensor = torch.stack(processed_images)

    if getattr(run, "_is_optimized_for_inference", False):
        if run._optimized_resolution != batch_tensor.shape[2]:
            raise ValueError(
                f"Resolution mismatch: optimized for {run._optimized_resolution}, "
                f"got {batch_tensor.shape[2]}. Call remove_optimized_model() or re-optimize."
            )
        if getattr(run, "_optimized_has_been_compiled", False):
            if run._optimized_batch_size != batch_tensor.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: optimized for {run._optimized_batch_size}, "
                    f"got {batch_tensor.shape[0]}."
                )

    with torch.no_grad():
        if getattr(run, "_is_optimized_for_inference", False) and run.inference_model is not None:
            predictions = run.inference_model(batch_tensor.to(dtype=run._optimized_dtype))
        else:
            predictions = run.model(batch_tensor)

        if isinstance(predictions, tuple):
            return_predictions = {
                "pred_logits": predictions[1],
                "pred_boxes": predictions[0],
            }
            if len(predictions) == 3:
                return_predictions["pred_masks"] = predictions[2]
            predictions = return_predictions

        target_sizes = torch.tensor(orig_sizes, device=run.device)
        results = run.postprocess(predictions, target_sizes=target_sizes)

    detections_list: list[sv.Detections] = []
    for result in results:
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]

        keep = scores > threshold
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        if "masks" in result:
            masks = result["masks"][keep]
            detections = sv.Detections(
                xyxy=boxes.float().cpu().numpy(),
                confidence=scores.float().cpu().numpy(),
                class_id=labels.cpu().numpy(),
                mask=masks.squeeze(1).cpu().numpy(),
            )
        else:
            detections = sv.Detections(
                xyxy=boxes.float().cpu().numpy(),
                confidence=scores.float().cpu().numpy(),
                class_id=labels.cpu().numpy(),
            )
        detections_list.append(detections)

    return detections_list if len(detections_list) > 1 else detections_list[0]
