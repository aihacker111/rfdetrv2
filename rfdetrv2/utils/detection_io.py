# ------------------------------------------------------------------------
# Label helpers for visualization / CLI (used by ``scripts/inference*.py``).
# ------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import supervision as sv

from rfdetrv2.utils.coco_classes import COCO_CLASSES


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
