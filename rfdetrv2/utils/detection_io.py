# ------------------------------------------------------------------------
# Label helpers for visualization / CLI (used by ``scripts/inference*.py``).
# ------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
import supervision as sv
import torch
from PIL import Image

from rfdetrv2.utils.coco_classes import COCO_CLASSES

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[misc, assignment]


def resolve_class_label(cls_id: int, class_names: dict, *, use_coco_fallback: bool = True) -> str:
    """Map a raw class id to a string label (handles RF-DETR / COCO-style dicts)."""
    raw = int(cls_id)
    for key in (raw, raw + 1):
        val = class_names.get(key)
        if isinstance(val, str):
            return val
    if use_coco_fallback:
        if raw in COCO_CLASSES:
            return COCO_CLASSES[raw]
        if (raw + 1) in COCO_CLASSES:
            return COCO_CLASSES[raw + 1]
    return str(raw)


def merge_overlapping_detections(
    detections: sv.Detections,
    labels: list[str],
    box_eps: float = 2.0,
) -> tuple[sv.Detections, list[str]]:
    """Merge rows with identical boxes so stacked labels stay readable."""
    if len(detections) == 0:
        return detections, labels
    xyxy = np.asarray(detections.xyxy)
    conf = np.asarray(detections.confidence)
    cls_id = np.asarray(detections.class_id)
    merged_xyxy, merged_conf, merged_cls, merged_labels = [], [], [], []
    used = [False] * len(detections)
    for i in range(len(detections)):
        if used[i]:
            continue
        box = xyxy[i]
        group = [i]
        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            if np.all(np.abs(xyxy[j] - box) < box_eps):
                group.append(j)
                used[j] = True
        used[i] = True
        merged_xyxy.append(box)
        merged_conf.append(conf[group[0]])
        merged_cls.append(cls_id[group[0]])
        merged_labels.append("\n".join(labels[k] for k in group))
    return (
        sv.Detections(
            xyxy=np.array(merged_xyxy),
            confidence=np.array(merged_conf),
            class_id=np.array(merged_cls),
        ),
        merged_labels,
    )


def _image_to_bgr(image: Union[str, np.ndarray, Image.Image, torch.Tensor]) -> np.ndarray:
    """Load or convert an image to BGR uint8 for OpenCV drawing."""
    if cv2 is None:
        raise ImportError("save_prediction_images requires opencv-python (cv2).")

    if isinstance(image, str):
        bgr = cv2.imread(image)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return bgr

    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image, torch.Tensor):
        t = image.detach().float().cpu()
        if t.dim() != 3 or t.shape[0] != 3:
            raise ValueError(f"Expected CHW RGB tensor with C=3, got shape {tuple(t.shape)}")
        if float(t.max()) <= 1.0 + 1e-3:
            t = (t * 255.0).clamp(0, 255)
        rgb = t.permute(1, 2, 0).numpy().astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if isinstance(image, np.ndarray):
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 ndarray, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if ndarray_is_bgr:
            return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    raise TypeError(f"Unsupported image type: {type(image)}")


def save_prediction_images(
    images: Union[str, np.ndarray, Image.Image, torch.Tensor, list],
    detections: Union[sv.Detections, List[sv.Detections]],
    save_path: Union[str, Path],
    *,
    class_names: dict | None = None,
    box_eps: float = 2.0,
    ndarray_is_bgr: bool = False,
) -> list[Path]:
    """
    Draw ``detections`` on ``images`` and write to disk.

    * ``save_path`` ending with ``.png`` / ``.jpg`` / … → single output (one image only).
    * Otherwise → treated as a directory; writes ``pred_000.png``, … (or ``<stem>_pred.png`` when one path input).
    """
    if cv2 is None:
        raise ImportError("save_prediction_images requires opencv-python (cv2).")

    save_path = Path(save_path)
    cn = class_names if class_names is not None else COCO_CLASSES

    img_list: list = images if isinstance(images, list) else [images]
    det_list: List[sv.Detections] = detections if isinstance(detections, list) else [detections]

    if len(img_list) != len(det_list):
        raise ValueError(
            f"images and detections length mismatch: {len(img_list)} vs {len(det_list)}"
        )

    suffixes = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    single_file = save_path.suffix.lower() in suffixes

    if single_file:
        if len(img_list) != 1:
            raise ValueError("save_path is a file path but multiple images were provided; pass a directory instead.")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out_paths = [save_path]
    else:
        save_path.mkdir(parents=True, exist_ok=True)
        out_paths = []
        for i in range(len(img_list)):
            stem = Path(img_list[i]).stem if isinstance(img_list[i], str) else f"pred_{i:03d}"
            out_paths.append(save_path / f"{stem}_pred.png")

    written: list[Path] = []
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(
        text_scale=0.35,
        text_thickness=1,
        text_padding=2,
    )

    for img_src, det, out_p in zip(img_list, det_list, out_paths):
        bgr = _image_to_bgr(img_src)
        labels = [
            f"{resolve_class_label(int(cid), cn)} {float(conf):.2f}"
            for conf, cid in zip(det.confidence, det.class_id)
        ]
        draw_det, draw_lbl = merge_overlapping_detections(det, labels, box_eps=box_eps)
        annotated = box_annotator.annotate(scene=bgr.copy(), detections=draw_det)
        annotated = label_annotator.annotate(
            scene=annotated, detections=draw_det, labels=draw_lbl
        )
        cv2.imwrite(str(out_p), annotated)
        written.append(out_p.resolve())

    return written
