#!/usr/bin/env python3
"""
RF-DETR v2 — inference on a single image or a video.

Usage
-----
  python scripts/inference.py --weights model.pth --image photo.jpg --save out.jpg
  python scripts/inference.py --weights model.pth --video clip.mp4 --output clip_det.mp4

**Number of classes:** the script reads your fine-tuned head from the .pth
(``class_embed.bias`` length or ``args.num_classes``) and passes ``num_classes=…``
into the model so it matches the checkpoint. Without this, the default is 90 (COCO)
and weights may not load into the head — you would see generic ``class_0..class_89``.

**Class names** (no extra files required when the .pth embeds them):

  1. ``class_names`` in the .pth
  2. ``args.class_names`` in the .pth
  3. Else generic ``class_0`` … from head size

Optional: ``--num-classes``, ``--classes-file``, ``--coco-json``, ``--dataset-dir``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rfdetrv2 import RFDETRV2Base, RFDETRV2Large, RFDETRV2Nano, RFDETRV2Small
from rfdetrv2.util.coco_classes import COCO_CLASSES, infer_classes_from_dataset_dir
from rfdetrv2.util.coco_classes import ordered_class_names_from_coco_json
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


def _get_class_name(cls_id: int, class_names: dict, use_coco_fallback: bool = True) -> str:
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


def _list_to_name_dict(names: list[str] | tuple) -> dict[int, str]:
    return {i + 1: str(n) for i, n in enumerate(names)}


def _infer_foreground_class_count(state: dict) -> int | None:
    """Foreground class count K from ``class_embed.bias`` (K+1 logits including background)."""
    for key, tensor in state.items():
        if not key.endswith("class_embed.bias") or not hasattr(tensor, "shape"):
            continue
        n = int(tensor.shape[0])
        if n >= 2:
            return n - 1
    return None


def _num_foreground_classes_for_model_build(ckpt: dict) -> int | None:
    """
    How many **foreground** classes the saved ``model`` was trained with.
    Prefer the actual ``class_embed`` size; fall back to ``args`` / ``class_names``.
    """
    state = ckpt.get("model")
    if isinstance(state, dict):
        k = _infer_foreground_class_count(state)
        if k is not None and k > 0:
            return k
    args_obj = ckpt.get("args")
    if args_obj is not None:
        nc = getattr(args_obj, "num_classes", None)
        if nc is not None:
            return int(nc)
    raw = ckpt.get("class_names")
    if isinstance(raw, (list, tuple)) and raw:
        return len(raw)
    return None


def _class_names_dict_from_checkpoint(
    weights_path: str,
    *,
    ckpt: dict | None = None,
) -> tuple[dict[int, str], str]:
    """
    Build ``{1..K: name}`` from checkpoint. Returns (mapping, source_tag) for logging.

    source_tag: ``"checkpoint.class_names"``, ``"checkpoint.args"``, ``"head_generic"``, or
    ``"none"`` (caller may fall back to COCO).
    """
    if ckpt is None:
        path = Path(weights_path)
        if not path.is_file():
            return {}, "none"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    raw = ckpt.get("class_names")
    if isinstance(raw, (list, tuple)) and raw:
        return _list_to_name_dict(raw), "checkpoint.class_names"

    args_obj = ckpt.get("args")
    if args_obj is not None:
        cn = getattr(args_obj, "class_names", None)
        if isinstance(cn, (list, tuple)) and cn:
            return _list_to_name_dict(cn), "checkpoint.args"
        if isinstance(cn, dict) and cn:
            return {int(k): str(v) for k, v in cn.items()}, "checkpoint.args"

    state = ckpt.get("model")
    if isinstance(state, dict):
        k = _infer_foreground_class_count(state)
        if k is not None and k > 0:
            return {i + 1: f"class_{i}" for i in range(k)}, "head_generic"

    return {}, "none"


def _load_model(args: argparse.Namespace):
    explicit = args.pretrained_encoder or DEFAULT_PRETRAINED
    pretrained = resolve_pretrained_encoder_path(
        _PROJECT_ROOT,
        args.model_size,
        explicit=explicit if explicit else None,
        weights_by_size=DINO_WEIGHTS_BY_SIZE,
    )
    cls = _MODELS[args.model_size]

    path = Path(args.weights)
    ckpt: dict | None = None
    if path.is_file():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    n_fg = getattr(args, "num_classes", None)
    if n_fg is not None:
        n_fg = int(n_fg)
    elif isinstance(ckpt, dict):
        n_fg = _num_foreground_classes_for_model_build(ckpt)

    model_kw: dict = dict(
        pretrain_weights=args.weights,
        pretrained_encoder=pretrained,
        device=args.device,
    )
    if n_fg is not None:
        model_kw["num_classes"] = n_fg
        if args.num_classes is None:
            print(
                f"[inference] Building model with num_classes={n_fg} "
                "(from checkpoint; default 90 would mismatch a fine-tuned head).",
                file=sys.stderr,
            )

    model = cls(**model_kw)

    ckpt_names, src = (
        _class_names_dict_from_checkpoint(args.weights, ckpt=ckpt)
        if isinstance(ckpt, dict)
        else _class_names_dict_from_checkpoint(args.weights)
    )
    if ckpt_names:
        model._class_names = ckpt_names
        if src == "head_generic":
            print(
                "[inference] No class names in checkpoint — using generic labels "
                f"class_0..class_{len(ckpt_names) - 1} from detection head size.",
                file=sys.stderr,
            )
    return model


def _resolve_class_names(args: argparse.Namespace, model) -> dict[int, str]:
    """Label map ``{1..K} -> name`` aligned with model indices (same as training)."""
    if args.classes_file:
        with open(args.classes_file, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if not names:
            raise ValueError(f"No non-empty lines in --classes-file {args.classes_file!r}")
        return {i + 1: n for i, n in enumerate(names)}
    if args.coco_json:
        names = ordered_class_names_from_coco_json(args.coco_json)
        if not names:
            raise ValueError(f"No categories in --coco-json {args.coco_json!r}")
        return {i + 1: n for i, n in enumerate(names)}
    if args.dataset_dir:
        id_to_name = infer_classes_from_dataset_dir(args.dataset_dir)
        if not id_to_name:
            raise ValueError(
                f"No COCO JSON with categories found under --dataset-dir {args.dataset_dir!r}"
            )
        ordered = [id_to_name[k] for k in sorted(id_to_name.keys())]
        return {i + 1: n for i, n in enumerate(ordered)}
    inner = getattr(model, "_class_names", None)
    if inner:
        return inner
    print(
        "[inference] Warning: could not read class names or head size from checkpoint; "
        "using MS-COCO names (likely wrong for a custom fine-tuned model).",
        file=sys.stderr,
    )
    return COCO_CLASSES


def _run_image(args: argparse.Namespace) -> None:
    model = _load_model(args)
    detections = model.predict(args.image, threshold=args.threshold)

    class_names = _resolve_class_names(args, model)
    coco_fb = class_names is COCO_CLASSES

    print(f"Detections: {len(detections)}")
    for i, (xyxy, conf, cls_id) in enumerate(
        zip(detections.xyxy, detections.confidence, detections.class_id),
        start=1,
    ):
        cls_int = int(cls_id)
        cls_name = _get_class_name(cls_int, class_names, use_coco_fallback=coco_fb)
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
            f"{_get_class_name(int(cid), class_names, use_coco_fallback=coco_fb)} {float(conf):.2f}"
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
    model = _load_model(args)
    class_names = _resolve_class_names(args, model)
    coco_fb = class_names is COCO_CLASSES

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
                f"{_get_class_name(int(cid), class_names, use_coco_fallback=coco_fb)} {float(conf):.2f}"
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
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Override foreground class count K (default: read from checkpoint).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str, help="Single image path")
    mode.add_argument("--video", type=str, help="Input video path")

    parser.add_argument("--save", type=str, default=None, help="[image] Annotated output path")
    parser.add_argument(
        "--classes-file",
        type=str,
        default=None,
        help="Optional override: one class name per line (index 0 = first class).",
    )
    parser.add_argument(
        "--coco-json",
        type=str,
        default=None,
        help="Optional override: COCO JSON with categories (sorted by id = class order).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Optional override: dataset root to locate a COCO JSON for category names.",
    )
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
