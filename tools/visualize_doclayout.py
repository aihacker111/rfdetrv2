"""
DocLayNet COCO Dataset Visualizer
==================================
Visualize images with bounding boxes and class labels from a COCO-format dataset.
Each image is saved as its own file under save_dir.

Usage
-----
    # Save every image individually
    python visualize_dataset.py --data_dir /path/to/doclaynet --split train --save_dir ./viz

    # Random sample of N images
    python visualize_dataset.py --data_dir /path/to/doclaynet --num_images 20 --save_dir ./viz

    # Single image by ID
    python visualize_dataset.py --data_dir /path/to/doclaynet --image_id 42 --save_dir ./viz

    # Filter by category
    python visualize_dataset.py --data_dir /path/to/doclaynet --category "Text" --save_dir ./viz

    # Dataset-level statistics chart
    python visualize_dataset.py --data_dir /path/to/doclaynet --stats --save_dir ./viz

Expected layout
---------------
    doclaynet/
    ├── images/train/  (or flat images/)
    └── annotations/train.json
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ─────────────────────── color palette ───────────────────────────

# 20 visually distinct colors (BGR for OpenCV, RGB for matplotlib)
PALETTE = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]

def get_color(cat_id: int):
    """Return a consistent RGB color for a category id."""
    return PALETTE[cat_id % len(PALETTE)]


# ─────────────────────── loader ──────────────────────────────────

class COCOLoader:
    """Minimal COCO loader — no external dependencies."""

    def __init__(self, ann_file: Path, images_dir: Path):
        with open(ann_file) as f:
            data = json.load(f)

        self.images_dir = images_dir
        self.categories  = {c["id"]: c["name"] for c in data["categories"]}
        self.cat_name_to_id = {v: k for k, v in self.categories.items()}

        self.images = {img["id"]: img for img in data["images"]}

        # Build image_id → list of annotations
        self.anns_by_image = defaultdict(list)
        for ann in data["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        self.image_ids = list(self.images.keys())

    def find_image_path(self, file_name: str) -> Path:
        direct = self.images_dir / file_name
        if direct.exists():
            return direct
        stem = Path(file_name).name
        for p in self.images_dir.rglob(stem):
            return p
        raise FileNotFoundError(f"Image not found: {file_name}")

    def get_by_category(self, cat_name: str):
        """Return image IDs that have at least one annotation of cat_name."""
        cat_id = self.cat_name_to_id.get(cat_name)
        if cat_id is None:
            raise ValueError(f"Category '{cat_name}' not found. "
                             f"Available: {sorted(self.categories.values())}")
        return list({ann["image_id"]
                     for ann in sum(self.anns_by_image.values(), [])
                     if ann["category_id"] == cat_id})


# ─────────────────────── draw one image ──────────────────────────

def draw_image(
    img_array: np.ndarray,
    annotations: list,
    categories: dict,
    show_labels: bool = True,
    alpha: float = 0.25,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes + filled overlays on a copy of img_array.

    Returns a uint8 RGB numpy array.
    """
    canvas = img_array.copy().astype(np.uint8)
    overlay = canvas.copy()

    for ann in annotations:
        cat_id = ann["category_id"]
        color  = get_color(cat_id)
        x, y, w, h = [int(v) for v in ann["bbox"]]
        x2, y2 = x + w, y + h

        # Filled rectangle on overlay
        cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)
        # Solid border on canvas
        cv2.rectangle(canvas, (x, y), (x2, y2), color, line_thickness)

        if show_labels:
            label = categories.get(cat_id, str(cat_id))
            conf  = ann.get("score", ann.get("confidence", None))
            text  = f"{label} {conf:.2f}" if conf is not None else label

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.35, min(0.6, w / 200))
            thickness  = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Background pill for text
            tx, ty = x, max(y - 4, th + 4)
            cv2.rectangle(canvas, (tx, ty - th - baseline - 2), (tx + tw + 4, ty + 2), color, -1)
            cv2.putText(canvas, text, (tx + 2, ty - baseline),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Blend overlay
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    return canvas


# ─────────────────────── stats panel ─────────────────────────────

def draw_stats_panel(annotations: list, categories: dict, figsize_w: int = 4) -> plt.Figure:
    """Small bar chart showing class distribution for a single image."""
    counts = defaultdict(int)
    for ann in annotations:
        counts[categories.get(ann["category_id"], str(ann["category_id"]))] += 1

    if not counts:
        return None

    names  = sorted(counts, key=lambda x: -counts[x])
    values = [counts[n] for n in names]
    colors = [
        tuple(c / 255 for c in get_color(categories.get(n) or list(categories.keys())[0]))
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(figsize_w, max(2, len(names) * 0.4)))
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xlabel("count", fontsize=8)
    ax.set_title("Annotations per class", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


# ─────────────────────── visualizer ──────────────────────────────

class DatasetVisualizer:

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        ann_subdir: str = "annotations",
        img_subdir: str = "images",
    ):
        data_dir = Path(data_dir)
        ann_file = data_dir / ann_subdir / f"{split}.json"
        img_dir  = data_dir / img_subdir / split
        if not img_dir.exists():
            img_dir = data_dir / img_subdir   # flat layout

        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        self.loader = COCOLoader(ann_file, img_dir)
        print(f"Loaded {len(self.loader.image_ids)} images, "
              f"{sum(len(v) for v in self.loader.anns_by_image.values())} annotations, "
              f"{len(self.loader.categories)} categories.")
        print(f"Categories: {', '.join(sorted(self.loader.categories.values()))}")

    # ── public methods ──────────────────────────────────────────

    def show(
        self,
        image_ids: list = None,
        num_images: int = 9,
        show_labels: bool = True,
        save_dir: str = None,
    ):
        """
        Save (or display) each annotated image as its own file.

        Args:
            image_ids:   Specific IDs to process. None = random sample.
            num_images:  How many images when image_ids is None.
            show_labels: Draw class name on each box.
            save_dir:    Directory to save individual PNGs. None = display interactively.
        """
        if image_ids is None:
            image_ids = random.sample(
                self.loader.image_ids,
                min(num_images, len(self.loader.image_ids)),
            )

        total = len(image_ids)
        print(f"Processing {total} images...")

        for i, img_id in enumerate(image_ids, 1):
            try:
                canvas   = self._render_image(img_id, show_labels)
                img_meta = self.loader.images[img_id]
                # Use the original filename stem so the output is easy to trace back
                stem = Path(img_meta["file_name"]).stem
                filename = f"{stem}_id{img_id}.png"

                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(canvas)
                ax.axis("off")
                ax.set_title(
                    f"{img_meta['file_name']}  |  "
                    f"{len(self.loader.anns_by_image[img_id])} annotations",
                    fontsize=9,
                )
                fig.tight_layout()
                self._save_or_show(fig, save_dir, filename)

                if save_dir:
                    print(f"  [{i}/{total}] saved → {filename}")

            except Exception as e:
                print(f"  [{i}/{total}] ERROR image_id={img_id}: {e}")

    def show_single(
        self,
        image_id: int,
        show_stats: bool = True,
        save_dir: str = None,
    ):
        """
        Display a single image with full-size bounding boxes + class legend + stats bar.

        Args:
            image_id:   COCO image id.
            show_stats: Show class distribution bar chart alongside.
            save_dir:   Save to file instead of displaying.
        """
        canvas = self._render_image(image_id, show_labels=True)
        anns   = self.loader.anns_by_image[image_id]

        if show_stats and anns:
            fig, (ax_img, ax_stat) = plt.subplots(
                1, 2, figsize=(16, 8),
                gridspec_kw={"width_ratios": [3, 1]},
            )
        else:
            fig, ax_img = plt.subplots(1, 1, figsize=(14, 8))
            ax_stat = None

        ax_img.imshow(canvas)
        ax_img.axis("off")

        # Legend
        used_cats = {ann["category_id"] for ann in anns}
        patches = [
            mpatches.Patch(
                color=tuple(c / 255 for c in get_color(cid)),
                label=self.loader.categories.get(cid, str(cid)),
            )
            for cid in sorted(used_cats)
        ]
        ax_img.legend(handles=patches, loc="lower left", fontsize=8,
                      framealpha=0.7, ncol=max(1, len(patches) // 10))

        img_meta = self.loader.images[image_id]
        ax_img.set_title(
            f"{img_meta['file_name']}   "
            f"[{img_meta.get('width','?')}×{img_meta.get('height','?')}]   "
            f"{len(anns)} annotations",
            fontsize=9,
        )

        if ax_stat is not None:
            counts = defaultdict(int)
            for ann in anns:
                counts[self.loader.categories.get(ann["category_id"], str(ann["category_id"]))] += 1
            names  = sorted(counts, key=lambda x: -counts[x])
            values = [counts[n] for n in names]
            colors = [tuple(c / 255 for c in get_color(
                self.loader.cat_name_to_id.get(n, 0))) for n in names]
            bars = ax_stat.barh(names[::-1], values[::-1], color=colors[::-1])
            ax_stat.bar_label(bars, padding=3, fontsize=8)
            ax_stat.set_xlabel("count", fontsize=8)
            ax_stat.set_title("Class distribution", fontsize=9)
            ax_stat.tick_params(labelsize=8)

        img_meta = self.loader.images[image_id]
        stem = Path(img_meta["file_name"]).stem
        filename = f"{stem}_id{image_id}.png"
        fig.tight_layout()
        self._save_or_show(fig, save_dir, filename)

    def show_category(
        self,
        category_name: str,
        num_images: int = 9,
        save_dir: str = None,
    ):
        """Show / save individual images that contain a specific category."""
        ids = self.loader.get_by_category(category_name)
        print(f"Found {len(ids)} images with category '{category_name}'.")
        sample = random.sample(ids, min(num_images, len(ids)))
        self.show(image_ids=sample, save_dir=save_dir)

    def dataset_stats(self, save_dir: str = None):
        """Plot dataset-level statistics: class distribution + annotations per image."""
        # ── class distribution ──
        cat_counts = defaultdict(int)
        ann_per_image = []

        for img_id, anns in self.loader.anns_by_image.items():
            ann_per_image.append(len(anns))
            for ann in anns:
                cat_counts[self.loader.categories.get(ann["category_id"],
                            str(ann["category_id"]))] += 1

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Class bar chart
        names  = sorted(cat_counts, key=lambda x: -cat_counts[x])
        values = [cat_counts[n] for n in names]
        colors = [tuple(c / 255 for c in get_color(
                  self.loader.cat_name_to_id.get(n, 0))) for n in names]
        bars = axes[0].barh(names[::-1], values[::-1], color=colors[::-1])
        axes[0].bar_label(bars, padding=3, fontsize=8)
        axes[0].set_xlabel("annotation count")
        axes[0].set_title(f"Class distribution  ({len(names)} classes)")
        axes[0].tick_params(labelsize=8)

        # Annotations-per-image histogram
        axes[1].hist(ann_per_image, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
        axes[1].axvline(np.mean(ann_per_image), color="red", linestyle="--",
                        label=f"mean={np.mean(ann_per_image):.1f}")
        axes[1].axvline(np.median(ann_per_image), color="orange", linestyle="--",
                        label=f"median={np.median(ann_per_image):.1f}")
        axes[1].set_xlabel("annotations per image")
        axes[1].set_ylabel("image count")
        axes[1].set_title("Annotations per image distribution")
        axes[1].legend(fontsize=8)

        total_anns = sum(values)
        fig.suptitle(
            f"Dataset stats  —  {len(self.loader.image_ids)} images  |  "
            f"{total_anns} annotations  |  {len(names)} classes",
            fontsize=11,
        )
        fig.tight_layout()
        self._save_or_show(fig, save_dir, "dataset_stats.png")

    # ── internal ────────────────────────────────────────────────

    def _render_image(self, image_id: int, show_labels: bool = True) -> np.ndarray:
        img_meta = self.loader.images[image_id]
        img_path = self.loader.find_image_path(img_meta["file_name"])
        img      = np.array(Image.open(img_path).convert("RGB"))
        anns     = self.loader.anns_by_image[image_id]
        return draw_image(img, anns, self.loader.categories, show_labels=show_labels)

    @staticmethod
    def _save_or_show(fig: plt.Figure, save_dir: str, filename: str):
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            out = Path(save_dir) / filename
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved → {out}")
            plt.close(fig)
        else:
            plt.show()


# ─────────────────────────── CLI ─────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Visualize a DocLayNet / COCO-format dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",    required=True,  help="Dataset root directory.")
    p.add_argument("--split",       default="train", help="Split name (train / val / test).")
    p.add_argument("--num_images",  type=int, default=9,  help="Number of images to process.")
    p.add_argument("--image_id",    type=int, default=None, help="Process a single image by ID.")
    p.add_argument("--category",    default=None, help="Filter to a specific category name.")
    p.add_argument("--stats",       action="store_true", help="Show dataset-level statistics.")
    p.add_argument("--save_dir",    default=None, help="Directory to save individual output images.")
    p.add_argument("--no_labels",   action="store_true", help="Hide class labels on boxes.")
    args = p.parse_args()

    viz = DatasetVisualizer(data_dir=args.data_dir, split=args.split)

    if args.stats:
        viz.dataset_stats(save_dir=args.save_dir)

    elif args.image_id is not None:
        viz.show_single(args.image_id, save_dir=args.save_dir)

    elif args.category:
        viz.show_category(args.category, num_images=args.num_images, save_dir=args.save_dir)

    else:
        viz.show(num_images=args.num_images,
                 show_labels=not args.no_labels,
                 save_dir=args.save_dir)


if __name__ == "__main__":
    main()