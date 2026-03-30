#!/usr/bin/env python3
"""
Simple video inference with RF-DETR v2.
Reads video, runs detection on each frame, writes annotated output.
"""
from pathlib import Path
import argparse
import sys

import cv2
import supervision as sv

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from rfdetrv2.detr._util import pydantic_dump
from rfdetrv2.runner import Pipeline
from rfdetrv2.schemas import RFDETRBaseConfig, RFDETRSmallConfig
from rfdetrv2.utils.coco_classes import COCO_CLASSES
from rfdetrv2.utils.detection_io import resolve_class_label
from rfdetrv2.utils.dinov3_pretrained import resolve_pretrained_encoder_path

DINO_WEIGHTS_BY_SIZE = {
    "small": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RF-DETR v2 inference on video.")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default: input_video_det.mp4)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--model-size", choices=["small", "base"], default="base")
    parser.add_argument("--show", action="store_true", help="Display output in window (press 'q' to quit)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Process every Nth frame (0 = all)")
    args = parser.parse_args()

    cfg_cls = RFDETRSmallConfig if args.model_size == "small" else RFDETRBaseConfig
    pretrained = resolve_pretrained_encoder_path(
        project_root,
        args.model_size,
        explicit=None,
        weights_by_size=DINO_WEIGHTS_BY_SIZE,
    )
    model_cfg = cfg_cls(
        pretrain_weights=args.weights,
        pretrained_encoder=pretrained,
        device=args.device,
    )
    pipe = Pipeline(**pydantic_dump(model_cfg))
    class_names = getattr(pipe, "class_names", None) or COCO_CLASSES

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.output or str(Path(args.video).parent / (Path(args.video).stem + "_det.mp4"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_idx = 0
    skip = max(1, args.skip_frames) if args.skip_frames else 1

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % skip != 0:
                writer.write(frame_bgr)
                frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            detections = pipe.predict(frame_rgb, threshold=args.threshold)

            labels = [
                f"{resolve_class_label(int(cid), class_names)} {float(conf):.2f}"
                for conf, cid in zip(detections.confidence, detections.class_id)
            ]

            annotated = box_annotator.annotate(scene=frame_bgr.copy(), detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            writer.write(annotated)

            if args.show:
                cv2.imshow("RF-DETR", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Saved output to {out_path}")


if __name__ == "__main__":
    main()
