"""
Visualize DocumentAugmentation on samples from a COCO dataset directory.

Dataset structure expected:
    dataset_dir/
        images/         (or train/ val/)
        annotations/    _annotations.coco.json  or  instances_train2017.json

Usage:
    python viz_augment.py --dataset path/to/dataset --n 10
    python viz_augment.py --dataset path/to/dataset --n 20 --out results/ --no_spatial
"""
import argparse
import json
import random
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# ── try project import, fall back to inline ──────────────────────────────────
try:
    import sys
    sys.path.insert(0, ".")
    from rfdetrv2.datasets.transforms import DocumentAugmentation
except ImportError:
    _BBOX_PARAMS = A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels", "orig_indices"],
        min_area=1.0,
        min_visibility=0.0,
        check_each_transform=True,
    )

    class DocumentAugmentation:
        def __init__(self, apply_spatial=True, p=1.0):
            spatial_ops = ([
                A.OneOf([
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                                       rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.2,
                                     border_mode=cv2.BORDER_CONSTANT, p=1.0),
                    A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, crop_border=True, p=1.0),
                ], p=0.4),
            ] if apply_spatial else [])
            background_ops = [A.OneOf([
                A.RandomShadow(shadow_roi=(0,0,1,1), num_shadows_lower=1,
                               num_shadows_upper=2, shadow_dimension=4, p=1.0),
                A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, alpha_coef=0.1, p=1.0),
                A.RandomRain(slant_lower=-5, slant_upper=5, drop_length=8, drop_width=1,
                             drop_color=(180,180,180), blur_value=1,
                             brightness_coefficient=0.9, rain_type=None, p=1.0),
                A.GridDropout(ratio=0.03, unit_size_min=1, unit_size_max=2,
                              holes_number_x=1, random_offset=False, p=1.0),
            ], p=0.35)]
            color_ops = [A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            ], p=0.4)]
            degradation_ops = [A.OneOf([
                A.Defocus(radius=(1, 3), alias_blur=0.1, p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=1.0),
                A.Morphological(scale=(1, 3), operation="erosion", p=0.5),
                A.Morphological(scale=(1, 3), operation="dilation", p=0.5),
            ], p=0.35)]
            all_ops = spatial_ops + background_ops + color_ops + degradation_ops
            self._compose = A.Compose(
                [A.SomeOf(all_ops, n=min(2, len(all_ops)), replace=False, p=p)],
                bbox_params=_BBOX_PARAMS,
            )

        def __call__(self, img, bboxes, class_labels):
            orig_indices = list(range(len(bboxes)))
            result = self._compose(
                image=img, bboxes=bboxes,
                class_labels=class_labels, orig_indices=orig_indices, masks=[],
            )
            return result["image"], result["bboxes"], result["class_labels"]


# ── dataset loader ────────────────────────────────────────────────────────────

def find_annotation_file(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "train" / "_annotations.coco.json",
        dataset_dir / "annotations" / "instances_train2017.json",
        dataset_dir / "annotations" / "train.json",
        dataset_dir / "_annotations.coco.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    ann_dir = dataset_dir / "annotations"
    if ann_dir.is_dir():
        for p in sorted(ann_dir.glob("*.json")):
            return p
    raise FileNotFoundError(f"No annotation JSON found under {dataset_dir}")


def find_image_dir(dataset_dir: Path) -> Path:
    for name in ("images", "train", "val"):
        p = dataset_dir / name
        if p.is_dir():
            return p
    return dataset_dir


def load_coco(dataset_dir: Path):
    ann_file = find_annotation_file(dataset_dir)
    img_dir  = find_image_dir(dataset_dir)
    print(f"Annotation : {ann_file}")
    print(f"Images dir : {img_dir}")

    with open(ann_file) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
    ann_by_img = {}
    for ann in coco.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    samples = []
    for img_info in coco["images"]:
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            img_path = img_dir / Path(img_info["file_name"]).name
        if not img_path.exists():
            continue
        samples.append({
            "path":    img_path,
            "anns":    ann_by_img.get(img_info["id"], []),
            "cat_map": cat_map,
        })
    print(f"Found {len(samples)} images with matched paths.\n")
    return samples


# ── draw ──────────────────────────────────────────────────────────────────────

COLORS = plt.cm.get_cmap("tab10").colors


def draw_image(ax, img, bboxes, labels, cat_map, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cid   = int(labels[i])
        color = COLORS[cid % len(COLORS)]
        name  = cat_map.get(cid, str(cid))
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=color, facecolor="none",
        ))
        ax.text(x1 + 2, y1 + 12, name, fontsize=7,
                color="white", backgroundcolor=color)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True,           help="Dataset root directory")
    parser.add_argument("--n",          type=int, default=10,    help="Number of sample images")
    parser.add_argument("--out",        default="aug_output",    help="Output folder")
    parser.add_argument("--no_spatial", action="store_true",     help="Disable spatial transforms")
    parser.add_argument("--p",          type=float, default=1.0, help="Augmentation probability")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_coco(Path(args.dataset))
    if not samples:
        raise RuntimeError("No images found in dataset.")

    chosen = random.sample(samples, min(args.n, len(samples)))
    aug    = DocumentAugmentation(apply_spatial=not args.no_spatial, p=args.p)

    for idx, sample in enumerate(chosen):
        img_bgr = cv2.imread(str(sample["path"]))
        if img_bgr is None:
            print(f"  skip (unreadable): {sample['path']}")
            continue
        img     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img.shape[:2]
        cat_map = sample["cat_map"]

        # parse annotations → xyxy
        bboxes, class_labels = [], []
        for ann in sample["anns"]:
            x, y, bw, bh = ann["bbox"]
            x2, y2 = min(x + bw, w), min(y + bh, h)
            if x2 > x and y2 > y:
                bboxes.append([x, y, x2, y2])
                class_labels.append(ann["category_id"])

        aug_img, aug_boxes, aug_labels = aug(img.copy(), bboxes, class_labels)

        # side-by-side: original | augmented
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
        draw_image(ax_l, img,     bboxes,    class_labels, cat_map, "original")
        draw_image(ax_r, aug_img, aug_boxes, aug_labels,   cat_map, "augmented")
        fig.suptitle(Path(sample["path"]).name, fontsize=11, y=1.01)
        plt.tight_layout()

        out_path = out_dir / f"{idx:03d}_{Path(sample['path']).stem}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=130)
        plt.close(fig)
        print(f"[{idx+1}/{len(chosen)}] saved → {out_path}")

    print(f"\nDone. {len(chosen)} images saved to '{out_dir}/'")


if __name__ == "__main__":
    main()