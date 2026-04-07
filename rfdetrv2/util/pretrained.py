"""
Single import surface for RF-DETR pretrained file resolution.

* **RF-DETR COCO checkpoints** — :func:`resolve_rfdetr_coco_checkpoint`

DINOv3 encoder weights are now resolved automatically inside
``DinoV3.__init__`` via ``torch.hub.download_url_to_file`` — no manual
helper is needed.
"""
from __future__ import annotations

from rfdetrv2.util.rfdetr_pretrained import resolve_rfdetr_coco_checkpoint

__all__ = [
    "resolve_rfdetr_coco_checkpoint",
]
