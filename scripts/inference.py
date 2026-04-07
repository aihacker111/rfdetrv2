#!/usr/bin/env python3
"""
RF-DETR v2 — inference on a single image or video.

Model size, num_classes, and class names are read **automatically** from
the checkpoint — no need to specify them manually.  This works correctly
for any fine-tuned checkpoint (custom dataset, different class counts).

Usage
-----
  # Image
  python inference.py --weights runs/exp1/checkpoint_best_total.pth --image photo.jpg
  python inference.py --weights runs/exp1/checkpoint_best_total.pth --image photo.jpg --save out.jpg

  # Optional DINOv3 backbone weights (override checkpoint / auto-resolve):
  python inference.py --weights ckpt.pth --image a.jpg --pretrained-encoder /path/to/dinov3.pth
  # Or: PRETRAINED_ENCODER=/path/to/dinov3.pth python inference.py --weights ckpt.pth --image a.jpg

  # Video
  python inference.py --weights runs/exp1/checkpoint_best_total.pth --video clip.mp4
  python inference.py --weights runs/exp1/checkpoint_best_total.pth --video clip.mp4 --output det.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rfdetrv2 import RFDETRBase, RFDETRLarge, RFDETRNano, RFDETRSmall
from rfdetrv2.util.pretrained import model_size_from_encoder, resolve_pretrained_encoder_path

_DEFAULT_PRETRAINED_ENV = os.environ.get("PRETRAINED_ENCODER")


# ─────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────

# Maps (encoder_name, resolution) → model class.
# Both Base and Large share dinov3_base — distinguish by resolution.
_ENCODER_TO_MODEL = {
    "dinov3_nano":  RFDETRNano,
    "dinov3_small": RFDETRSmall,
    "dinov3_base":  RFDETRBase,   # resolution <= 560 → Base
    "dinov3_large": RFDETRLarge,
}


def _load_checkpoint_meta(
    weights: str,
) -> tuple[type, int, dict[int, str], str, int, str | None]:
    """
    Read the checkpoint and return:
      - model_class  : the right RFDETRxxx class
      - num_classes  : number of foreground classes
      - class_names  : {1: "Caption", 2: "Footnote", …}
      - encoder      : e.g. ``dinov3_base``
      - resolution   : training input size
      - pretrained_encoder_saved : path from ``args.pretrained_encoder`` if any

    Values come from ``ckpt['args']`` when present.  Falls back to detection-head
    tensor shape for ``num_classes`` on older checkpoints.
    """
    path = Path(weights)
    if not path.exists():
        sys.exit(f"[error] Checkpoint not found: {weights}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        sys.exit("[error] Checkpoint must be a .pth dict saved by the rfdetrv2 trainer.")

    args = ckpt.get("args")

    # ── num_classes ──────────────────────────────────────────────
    # Primary source: args.num_classes (always present after fine-tuning)
    num_classes = getattr(args, "num_classes", None) if args is not None else None

    # Fallback: read from detection-head tensor shape (K+1 logits)
    if num_classes is None:
        state = ckpt.get("model", {})
        bias = state.get("class_embed.bias")
        if bias is not None:
            num_classes = int(bias.shape[0]) - 1  # subtract background

    if num_classes is None or num_classes < 1:
        sys.exit(
            "[error] Cannot determine num_classes from checkpoint.\n"
            "  → Re-save with a recent fine-tune run, or use checkpoint_best_total.pth."
        )

    # ── class_names ──────────────────────────────────────────────
    # args.class_names is a list ["Caption", "Footnote", …] after fine-tuning
    raw_names = getattr(args, "class_names", None) if args is not None else None
    if isinstance(raw_names, (list, tuple)) and raw_names:
        # COCO convention: category ids start at 1
        class_names = {i + 1: str(n) for i, n in enumerate(raw_names)}
    elif isinstance(raw_names, dict) and raw_names:
        class_names = {int(k): str(v) for k, v in raw_names.items()}
    else:
        # No names stored — use generic labels
        class_names = {i + 1: f"class_{i}" for i in range(num_classes)}
        print(
            f"[warning] No class names found in checkpoint.  "
            f"Using generic labels class_0 … class_{num_classes - 1}."
        )

    # ── model class (auto-detect from encoder + resolution) ──────
    encoder = getattr(args, "encoder", "dinov3_base") if args is not None else "dinov3_base"
    resolution = int(getattr(args, "resolution", 560) if args is not None else 560)
    pretrained_encoder_saved = (
        getattr(args, "pretrained_encoder", None) if args is not None else None
    )
    if isinstance(pretrained_encoder_saved, str) and pretrained_encoder_saved.strip():
        pretrained_encoder_saved = os.path.realpath(
            os.path.expanduser(pretrained_encoder_saved.strip())
        )
    else:
        pretrained_encoder_saved = None

    if encoder == "dinov3_base" and resolution > 560:
        model_class = RFDETRLarge
    else:
        model_class = _ENCODER_TO_MODEL.get(encoder, RFDETRBase)

    print(
        f"[info] Checkpoint: {path.name}\n"
        f"  model      = {model_class.__name__}  (encoder={encoder}, resolution={resolution})\n"
        f"  num_classes= {num_classes}\n"
        f"  classes    = {list(class_names.values())}"
    )
    return (
        model_class,
        num_classes,
        class_names,
        encoder,
        resolution,
        pretrained_encoder_saved,
    )


# ─────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────

def load_model(
    weights: str,
    device: str,
    *,
    pretrained_encoder: str | None = None,
):
    """Build and return (model, class_names) ready for inference."""
    (
        model_class,
        num_classes,
        class_names,
        encoder,
        resolution,
        pe_ckpt,
    ) = _load_checkpoint_meta(weights)

    explicit = pretrained_encoder or _DEFAULT_PRETRAINED_ENV or pe_ckpt
    if explicit:
        explicit = os.path.realpath(os.path.expanduser(explicit))

    model_size = model_size_from_encoder(encoder, resolution)
    pe_resolved = resolve_pretrained_encoder_path(_ROOT, model_size, explicit=explicit)
    print(f"[info] pretrained_encoder = {pe_resolved}")

    model = model_class(
        pretrain_weights=weights,
        pretrained_encoder=pe_resolved,
        num_classes=num_classes,
        device=device,
    )
    # Make class_names available for display
    model._class_names = class_names
    return model, class_names


# ─────────────────────────────────────────────────────────────────
# Draw helpers
# ─────────────────────────────────────────────────────────────────

# 20 visually distinct BGR colors
_PALETTE_BGR = [
    ( 75,  25, 230), ( 75, 180,  60), ( 25, 225, 255), (200, 130,   0),
    ( 48, 130, 245), (180,  30, 145), (240, 240,  70), (230,  50, 240),
    ( 60, 245, 210), (212, 190, 250), (128, 128,   0), (255, 190, 220),
    ( 40, 110, 170), (200, 250, 255), (  0,   0, 128), (195, 255, 170),
    (  0, 128, 128), (180, 215, 255), (128,   0,   0), (128, 128, 128),
]

def _color(cat_id: int):
    return _PALETTE_BGR[int(cat_id) % len(_PALETTE_BGR)]


def draw_boxes(
    image_bgr: np.ndarray,
    xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    class_names: dict[int, str],
    thickness: int = 2,
    alpha: float = 0.20,
) -> np.ndarray:
    """Draw filled + outlined bounding boxes with labels onto a BGR image."""
    canvas  = image_bgr.copy()
    overlay = canvas.copy()

    for box, score, cid in zip(xyxy, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        color = _color(cid)
        label = f"{class_names.get(int(cid) + 1, class_names.get(int(cid), str(cid)))} {score:.2f}"

        # Filled rectangle on overlay (for alpha blend)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        # Solid border
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Label background + text
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, min(0.55, (x2 - x1) / 200))
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(canvas, (x1, ty - th - bl - 2), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(canvas, label, (x1 + 2, ty - bl),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    return canvas


# ─────────────────────────────────────────────────────────────────
# Image inference
# ─────────────────────────────────────────────────────────────────

def run_image(args: argparse.Namespace) -> None:
    model, class_names = load_model(
        args.weights,
        args.device,
        pretrained_encoder=args.pretrained_encoder,
    )

    detections = model.predict(args.image, threshold=args.threshold)
    print(f"\n[result] {len(detections)} detection(s)")

    for i, (box, conf, cid) in enumerate(
        zip(detections.xyxy, detections.confidence, detections.class_id), start=1
    ):
        name = class_names.get(int(cid) + 1, class_names.get(int(cid), str(cid)))
        x1, y1, x2, y2 = map(lambda v: round(float(v), 1), box)
        print(f"  {i:03d}  {name:<20} conf={float(conf):.3f}  box=[{x1},{y1},{x2},{y2}]")

    if args.save:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            sys.exit(f"[error] Cannot read image: {args.image}")

        annotated = draw_boxes(
            image_bgr,
            xyxy=detections.xyxy,
            scores=detections.confidence,
            class_ids=detections.class_id,
            class_names=class_names,
        )
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        print(f"\n[saved] {out}")


# ─────────────────────────────────────────────────────────────────
# Video inference
# ─────────────────────────────────────────────────────────────────

def run_video(args: argparse.Namespace) -> None:
    model, class_names = load_model(
        args.weights,
        args.device,
        pretrained_encoder=args.pretrained_encoder,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[error] Cannot open video: {args.video}")

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.output or str(
        Path(args.video).with_stem(Path(args.video).stem + "_det")
    )
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    skip  = max(1, args.skip_frames or 1)
    frame_idx = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % skip == 0:
                frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                detections = model.predict(frame_rgb, threshold=args.threshold)
                frame_bgr  = draw_boxes(
                    frame_bgr,
                    xyxy=detections.xyxy,
                    scores=detections.confidence,
                    class_ids=detections.class_id,
                    class_names=class_names,
                )

            writer.write(frame_bgr)

            if args.show:
                cv2.imshow("RF-DETR", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 200 == 0:
                print(f"  processed {frame_idx}/{total} frames …")

    finally:
        cap.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"[saved] {out_path}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="RF-DETR v2 inference — model size and classes auto-detected from checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",   required=True,  help="Path to .pth checkpoint.")
    p.add_argument("--device",    default="cuda",  choices=["cuda", "cpu", "mps"])
    p.add_argument("--threshold", type=float, default=0.35, help="Confidence threshold.")
    p.add_argument(
        "--pretrained-encoder",
        default=None,
        metavar="PATH",
        help=(
            "DINOv3 encoder .pth (or repo::weights). "
            "Default: checkpoint args, else PRETRAINED_ENCODER env, else dinov3_pretrained/ / download."
        ),
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", help="Path to an input image.")
    mode.add_argument("--video", help="Path to an input video.")

    # Image options
    p.add_argument("--save",   default=None, help="[image] Save annotated image to this path.")

    # Video options
    p.add_argument("--output",      default=None, help="[video] Output video path.")
    p.add_argument("--skip-frames", type=int, default=0,
                   help="[video] Run detection every N frames (0 = every frame).")
    p.add_argument("--show",        action="store_true",
                   help="[video] Show preview window (press q to quit).")

    args = p.parse_args()

    if args.image:
        run_image(args)
    else:
        run_video(args)


if __name__ == "__main__":
    main()