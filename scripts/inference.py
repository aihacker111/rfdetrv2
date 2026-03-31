#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Single-image (or path list) inference: ``Pipeline.from_pretrained`` + ``predict``.
# ------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import _common  # noqa: E402
from rfdetrv2.runner.trainer import Pipeline  # noqa: E402
from rfdetrv2.utils.coco_classes import COCO_CLASSES  # noqa: E402
from rfdetrv2.utils.detection_io import resolve_class_label  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RF-DETRv2 detection on image(s) via Pipeline.predict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _common.add_config_args(parser, mode="inference")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint .pth")
    parser.add_argument("--image", type=str, required=True, help="Input image path.")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="If set, write annotated image (path ends with .png/.jpg) or directory of PNGs.",
    )
    args = parser.parse_args()
    cfg_path = _common.resolve_yaml("inference", args.variant, args.config)
    kw: dict = {}
    if args.device is not None:
        kw["device"] = args.device

    pipe = Pipeline.from_pretrained(args.weights, config=str(cfg_path), **kw)
    dets = pipe.predict(args.image, threshold=args.threshold, save_path=args.save)

    names = getattr(pipe.cfg, "class_names", None) or getattr(pipe, "class_names", None)
    if isinstance(names, list):
        class_names = {i + 1: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        class_names = names
    else:
        class_names = COCO_CLASSES

    print(f"Detections: {len(dets)}")
    for i, (xyxy, conf, cls_id) in enumerate(
        zip(dets.xyxy, dets.confidence, dets.class_id),
        start=1,
    ):
        cls_int = int(cls_id)
        label = resolve_class_label(cls_int, class_names)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        print(
            f"{i:03d} | {label} (id={cls_int}) | conf={float(conf):.4f} "
            f"| box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )


if __name__ == "__main__":
    main()
