"""
Visualize DocumentAugmentation on a sample image.
Usage:
    python viz_augment.py --image path/to/image.jpg
    python viz_augment.py --image path/to/image.jpg --rows 3 --cols 4
"""
import argparse
import random

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── paste or import your DocumentAugmentation here ──────────────────────────
# from rfdetrv2.datasets.transforms import DocumentAugmentation
# or copy the class inline below:

import sys
sys.path.insert(0, ".")
try:
    from rfdetrv2.datasets.transforms import DocumentAugmentation
except ImportError:
    # fallback: inline minimal version for standalone testing
    import albumentations as A
    import cv2

    _BBOX_PARAMS = A.BboxParams(
        format="pascal_voc",
        label_fields=["class_labels", "orig_indices"],
        min_area=1.0,
        min_visibility=0.0,
        check_each_transform=True,
    )

    class DocumentAugmentation:
        def __init__(self, apply_spatial=True, p=0.5):
            self.apply_spatial = apply_spatial
            self.p = p
            spatial_ops = ([
                A.OneOf([
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=0,
                                       border_mode=cv2.BORDER_CONSTANT, p=1.0),
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
                A.RandomGamma(gamma_limit=(70,130), p=1.0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            ], p=0.4)]
            degradation_ops = [A.OneOf([
                A.Defocus(radius=(1,3), alias_blur=0.1, p=1.0),
                A.MotionBlur(blur_limit=(3,7), p=1.0),
                A.GaussianBlur(blur_limit=(3,5), sigma_limit=0, p=1.0),
                A.Morphological(scale=(1,3), operation="erosion", p=0.5),
                A.Morphological(scale=(1,3), operation="dilation", p=0.5),
            ], p=0.35)]
            all_ops = spatial_ops + background_ops + color_ops + degradation_ops
            self._compose = A.Compose(
                [A.SomeOf(all_ops, n=min(2, len(all_ops)), replace=False, p=self.p)],
                bbox_params=_BBOX_PARAMS,
            )

        def __call__(self, img, target=None):
            was_pil = isinstance(img, Image.Image)
            img_np = np.asarray(img) if was_pil else img
            bboxes = target["bboxes"] if target and "bboxes" in target else []
            class_labels = target["class_labels"] if target and "class_labels" in target else []
            orig_indices = list(range(len(bboxes)))
            result = self._compose(
                image=img_np, bboxes=bboxes,
                class_labels=class_labels, orig_indices=orig_indices, masks=[],
            )
            out = Image.fromarray(result["image"]) if was_pil else result["image"]
            out_target = None
            if target is not None:
                out_target = {
                    "bboxes": result["bboxes"],
                    "class_labels": result["class_labels"],
                }
            return out, out_target


# ── helpers ──────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_boxes(ax, img: np.ndarray, bboxes, labels, title: str) -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis("off")
    colors = plt.cm.get_cmap("tab10").colors
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        c = colors[int(labels[i]) % len(colors)] if labels else "red"
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.2, edgecolor=c, facecolor="none",
        )
        ax.add_patch(rect)


def make_dummy_boxes(img: np.ndarray):
    """Generate a few plausible bounding boxes when no annotation is provided."""
    h, w = img.shape[:2]
    boxes = [
        [int(w * 0.05), int(h * 0.05), int(w * 0.45), int(h * 0.25)],
        [int(w * 0.05), int(h * 0.30), int(w * 0.90), int(h * 0.50)],
        [int(w * 0.05), int(h * 0.55), int(w * 0.60), int(h * 0.75)],
    ]
    labels = [0, 1, 2]
    return boxes, labels


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--rows", type=int, default=3, help="Grid rows")
    parser.add_argument("--cols", type=int, default=4, help="Grid cols")
    parser.add_argument("--p", type=float, default=1.0,
                        help="Augmentation probability (1.0 = always apply)")
    parser.add_argument("--no_spatial", action="store_true",
                        help="Disable spatial transforms (layout-analysis mode)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    img = load_image(args.image)
    bboxes, labels = make_dummy_boxes(img)

    aug = DocumentAugmentation(apply_spatial=not args.no_spatial, p=args.p)

    n_aug = args.rows * args.cols - 1          # first cell = original
    fig, axes = plt.subplots(args.rows, args.cols, figsize=(args.cols * 3, args.rows * 3))
    axes = axes.flatten()

    # ── cell 0: original ──────────────────────────────────────────────────────
    draw_boxes(axes[0], img, bboxes, labels, "original")

    # ── remaining cells: augmented ────────────────────────────────────────────
    category_labels = ["spatial", "background", "color", "degradation"]
    for i in range(1, len(axes)):
        target = {"bboxes": bboxes, "class_labels": labels}
        aug_img, aug_target = aug(img.copy(), target)
        aug_np = np.asarray(aug_img) if isinstance(aug_img, Image.Image) else aug_img
        aug_boxes = aug_target["bboxes"] if aug_target else []
        aug_labels = aug_target["class_labels"] if aug_target else []
        draw_boxes(axes[i], aug_np, aug_boxes, aug_labels, f"aug #{i}")

    fig.suptitle(
        f"DocumentAugmentation  |  spatial={'on' if not args.no_spatial else 'off'}  p={args.p}",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, bbox_inches="tight", dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()