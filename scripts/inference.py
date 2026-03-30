from pathlib import Path
import argparse
import os
import sys

import cv2
import supervision as sv

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfdetrv2.detr._util import pydantic_dump
from rfdetrv2.runner import Pipeline
from rfdetrv2.schemas import RFDETRBaseConfig, RFDETRLargeConfig, RFDETRNanoConfig, RFDETRSmallConfig
from rfdetrv2.utils.coco_classes import COCO_CLASSES
from rfdetrv2.utils.detection_io import merge_overlapping_detections, resolve_class_label
from rfdetrv2.utils.dinov3_pretrained import resolve_pretrained_encoder_path

DINO_WEIGHTS_BY_SIZE = {
    "nano": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "small": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}
DEFAULT_PRETRAINED = os.environ.get("PRETRAINED_ENCODER")

_CONFIGS = {
    "nano": RFDETRNanoConfig,
    "small": RFDETRSmallConfig,
    "base": RFDETRBaseConfig,
    "large": RFDETRLargeConfig,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RF-DETRV2 inference on one image.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["nano", "small", "base", "large"],
        default="base",
        help="Model size: nano (21M), small (29M), base, large.",
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=str,
        default=DEFAULT_PRETRAINED,
        help="Path to DINOv3 .pth. If unset, uses dinov3_pretrained/ or project root, or downloads official weights.",
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--threshold", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--save", type=str, default=None, help="Optional output image path with drawn boxes.")
    parser.add_argument(
        "--classes-file",
        type=str,
        default=None,
        help="Optional text file with one class name per line (overrides checkpoint).",
    )
    args = parser.parse_args()

    explicit = args.pretrained_encoder or DEFAULT_PRETRAINED
    pretrained = resolve_pretrained_encoder_path(
        project_root,
        args.model_size,
        explicit=explicit if explicit else None,
        weights_by_size=DINO_WEIGHTS_BY_SIZE,
    )

    model_cfg = _CONFIGS[args.model_size](
        pretrain_weights=args.weights,
        pretrained_encoder=pretrained,
        device=args.device,
    )
    pipe = Pipeline(**pydantic_dump(model_cfg))
    detections = pipe.predict(args.image, threshold=args.threshold)
    class_names = getattr(pipe, "class_names", None) or COCO_CLASSES
    if args.classes_file:
        with open(args.classes_file) as f:
            names = [line.strip() for line in f if line.strip()]
        class_names = {i + 1: n for i, n in enumerate(names)}

    if class_names and class_names is not COCO_CLASSES:
        sample = dict(list(class_names.items())[:3])
        val_types = {type(v).__name__ for v in list(class_names.values())[:5]}
        print(f"[debug] class_names sample={sample} value_types={val_types}")

    print(f"Detections: {len(detections)}")
    for i, (xyxy, conf, cls_id) in enumerate(
        zip(detections.xyxy, detections.confidence, detections.class_id),
        start=1,
    ):
        cls_int = int(cls_id)
        cls_name = resolve_class_label(cls_int, class_names)
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        print(
            f"{i:03d} | class={cls_name} (id={cls_int}) | conf={float(conf):.4f} "
            f"| box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

    if args.save is not None:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read input image: {args.image}")

        labels = [
            f"{resolve_class_label(int(cid), class_names)} {float(conf):.2f}"
            for conf, cid in zip(detections.confidence, detections.class_id)
        ]
        draw_detections, draw_labels = merge_overlapping_detections(detections, labels)

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(
            text_scale=0.35,
            text_thickness=1,
            text_padding=2,
        )
        annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=draw_detections)
        annotated = label_annotator.annotate(scene=annotated, detections=draw_detections, labels=draw_labels)

        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated image to: {output_path}")


if __name__ == "__main__":
    main()
