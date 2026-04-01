#!/usr/bin/env python3
"""
RF-DETR v2 — inference on a single image or video.

``num_classes`` and display names are taken **only** from the checkpoint:

  * **K** (foreground classes): detection-head tensor shapes, else ``len(class_names)``,
    else ``args.num_classes``.
  * **Names**: top-level ``class_names`` (list or dict ``{1: "Caption", …}``), else
    ``args.class_names`` (list or dict), else ``class_0`` … for missing keys.

Usage
-----
  python scripts/inference.py --weights out/checkpoint_best_total.pth --image a.png --save out.png
  python scripts/inference.py --weights model.pth --video clip.mp4 --output clip_det.mp4
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rfdetrv2 import RFDETRV2Base, RFDETRV2Large, RFDETRV2Nano, RFDETRV2Small
from rfdetrv2.util.dinov3_pretrained import resolve_pretrained_encoder_path

DEFAULT_PRETRAINED = os.environ.get("PRETRAINED_ENCODER")

DINO_WEIGHTS_BY_SIZE = {
    "nano": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "small": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "large": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}

_MODELS = {
    "nano": RFDETRV2Nano,
    "small": RFDETRV2Small,
    "base": RFDETRV2Base,
    "large": RFDETRV2Large,
}


def _infer_k_from_state(state: dict) -> int | None:
    """Foreground count K from class-head biases (K+1 logits including background)."""
    n_outs: list[int] = []
    for key, tensor in state.items():
        if not isinstance(key, str) or not hasattr(tensor, "shape") or tensor.ndim < 1:
            continue
        if not (
            key.endswith("class_embed.bias")
            or re.search(r"enc_out_class_embed\.\d+\.bias$", key)
        ):
            continue
        n = int(tensor.shape[0])
        if n >= 2:
            n_outs.append(n)
    if not n_outs:
        return None
    n_out, _ = Counter(n_outs).most_common(1)[0]
    return n_out - 1


def _resolve_state_dict(ckpt: dict) -> dict | None:
    if not isinstance(ckpt, dict):
        return None
    for name in ("model", "ema_model", "model_ema", "state_dict"):
        inner = ckpt.get(name)
        if isinstance(inner, dict) and _infer_k_from_state(inner) is not None:
            return inner
    tensors = {k: v for k, v in ckpt.items() if isinstance(k, str) and isinstance(v, torch.Tensor)}
    if _infer_k_from_state(tensors) is not None:
        return tensors
    return None


def _names_from_checkpoint(ckpt: dict) -> dict[int, str]:
    """Build ``{1: name, …}`` from checkpoint metadata only (may be empty)."""
    raw = ckpt.get("class_names")
    if isinstance(raw, dict) and raw:
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)) and raw:
        return {i + 1: str(n) for i, n in enumerate(raw)}
    a = ckpt.get("args")
    if a is None:
        return {}
    cn = getattr(a, "class_names", None)
    if isinstance(cn, (list, tuple)) and cn:
        return {i + 1: str(n) for i, n in enumerate(cn)}
    if isinstance(cn, dict) and cn:
        return {int(k): str(v) for k, v in cn.items()}
    return {}


def _parse_checkpoint_meta(ckpt: dict) -> tuple[int, dict[int, str], int]:
    """
    Return ``(K, {1..K: name})`` using only checkpoint contents.

    **K** (architecture, must match saved tensors): weights first, else name list length,
    else ``args.num_classes``.

    **Labels**: use every name stored in the checkpoint for keys ``1 .. len(names)``; any
    missing key up to ``K`` gets ``class_{j}`` (0-based index). This covers the common
    case where metadata lists 11 DocLayNet classes but the file still has a 90-class
    COCO head — your 11 names are kept for logits ``0..10``.
    """
    state = _resolve_state_dict(ckpt)
    k_w = _infer_k_from_state(state) if state else None
    raw = _names_from_checkpoint(ckpt)
    k_n = len(raw) if raw else None
    a = ckpt.get("args")
    k_a = None
    if a is not None and getattr(a, "num_classes", None) is not None:
        k_a = int(a.num_classes)

    if k_w is not None and k_w > 0:
        k = k_w
    elif k_n is not None and k_n > 0:
        k = k_n
    elif k_a is not None and k_a > 0:
        k = k_a
    else:
        raise SystemExit(
            "Cannot infer num_classes from checkpoint. Need detection-head tensors, "
            "a ``class_names`` list, or ``args.num_classes``. "
            "Re-save with a recent train/finetune run, or use checkpoint_best_total.pth."
        )

    if k_w and k_n and k_n != k_w:
        print(
            f"[inference] Warning: weights imply K={k_w} classes but checkpoint lists "
            f"{k_n} name(s) (e.g. fine-tune metadata + COCO-sized head). "
            f"Using K={k} for loading; showing stored names for classes 1..{k_n}, "
            "generic labels for the rest. Re-fine-tune so the head matches your dataset.",
            file=sys.stderr,
        )

    names: dict[int, str] = {}
    for i in range(1, k + 1):
        if i in raw:
            names[i] = raw[i]
        else:
            names[i] = f"class_{i - 1}"

    n_meta = len(raw) if raw else 0
    return k, names, n_meta


def _label_name(cls_id: int, class_names: dict[int, str]) -> str:
    raw = int(cls_id)
    for key in (raw, raw + 1):
        val = class_names.get(key)
        if isinstance(val, str):
            return val
    return str(raw)


def _merge_overlapping_detections(detections: sv.Detections, labels: list, box_eps: float = 2.0):
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


def _load_model(weights: str, model_size: str, device: str, pretrained_encoder: str | None):
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(weights)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit("Checkpoint must be a dict (.pth with model/args or training format).")

    k, class_names, n_meta = _parse_checkpoint_meta(ckpt)
    if n_meta and n_meta < k:
        print(
            f"[inference] num_classes={k} (from checkpoint head); "
            f"{n_meta} name(s) from metadata, {k - n_meta} generic (class indices {n_meta}..{k - 1}).",
            file=sys.stderr,
        )
    else:
        print(
            f"[inference] num_classes={k} (from checkpoint); {len(class_names)} label(s).",
            file=sys.stderr,
        )

    enc = pretrained_encoder or DEFAULT_PRETRAINED
    pretrained = resolve_pretrained_encoder_path(
        _PROJECT_ROOT,
        model_size,
        explicit=enc if enc else None,
        weights_by_size=DINO_WEIGHTS_BY_SIZE,
    )
    model = _MODELS[model_size](
        pretrain_weights=weights,
        pretrained_encoder=pretrained,
        device=device,
        num_classes=k,
    )
    model._class_names = class_names
    return model, class_names


def _run_image(args: argparse.Namespace) -> None:
    model, class_names = _load_model(
        args.weights, args.model_size, args.device, args.pretrained_encoder
    )
    detections = model.predict(args.image, threshold=args.threshold)

    print(f"Detections: {len(detections)}")
    for i, (xyxy, conf, cls_id) in enumerate(
        zip(detections.xyxy, detections.confidence, detections.class_id),
        start=1,
    ):
        cls_int = int(cls_id)
        cls_name = _label_name(cls_int, class_names)
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        print(
            f"{i:03d} | class={cls_name} (id={cls_int}) | conf={float(conf):.4f} "
            f"| box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

    if args.save:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        labels = [
            f"{_label_name(int(cid), class_names)} {float(conf):.2f}"
            for conf, cid in zip(detections.confidence, detections.class_id)
        ]
        draw_detections, draw_labels = _merge_overlapping_detections(detections, labels)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(
            text_scale=0.35,
            text_thickness=1,
            text_padding=2,
        )
        annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=draw_detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=draw_detections, labels=draw_labels
        )
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), annotated)
        print(f"Saved: {out}")


def _run_video(args: argparse.Namespace) -> None:
    model, class_names = _load_model(
        args.weights, args.model_size, args.device, args.pretrained_encoder
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.output or str(
        Path(args.video).parent / (Path(args.video).stem + "_det.mp4")
    )
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
            detections = model.predict(frame_rgb, threshold=args.threshold)
            labels = [
                f"{_label_name(int(cid), class_names)} {float(conf):.2f}"
                for conf, cid in zip(detections.confidence, detections.class_id)
            ]
            annotated = box_annotator.annotate(scene=frame_bgr.copy(), detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="RF-DETR v2 image or video inference.")
    parser.add_argument("--weights", type=str, required=True, help="Checkpoint .pth")
    parser.add_argument(
        "--model-size",
        choices=list(_MODELS.keys()),
        default="base",
    )
    parser.add_argument("--pretrained-encoder", type=str, default=DEFAULT_PRETRAINED)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--threshold", type=float, default=0.35)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str, help="Single image path")
    mode.add_argument("--video", type=str, help="Input video path")

    parser.add_argument("--save", type=str, default=None, help="[image] Annotated output path")
    parser.add_argument("--output", type=str, default=None, help="[video] Output video path")
    parser.add_argument("--skip-frames", type=int, default=0, help="[video] Process every Nth frame (0=all)")
    parser.add_argument("--show", action="store_true", help="[video] Preview window (q to quit)")

    args = parser.parse_args()

    if args.image:
        _run_image(args)
    else:
        _run_video(args)


if __name__ == "__main__":
    main()
