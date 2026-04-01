"""
DocLayNet COCO Dataset Cleaner
==============================

Layout expected:
    doclaynet/
    ├── images/          <-- all images flat in here
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json

Output (saved next to originals):
    doclaynet/
    └── annotations/
        ├── train_clean.json
        ├── val_clean.json
        └── test_clean.json

Usage:
    python clean_doclaynet.py --data_dir /path/to/doclaynet
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  → Saved: {path}")


def get_images_on_disk(images_dir: Path) -> set:
    """Return a set of all image filenames found in images_dir."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return {f.name for f in images_dir.iterdir() if f.suffix.lower() in exts}


def clean_coco(data: dict, images_on_disk: set, json_name: str) -> dict:
    images      = data.get("images", [])
    annotations = data.get("annotations", [])
    categories  = data.get("categories", [])

    print(f"\n{'─'*55}")
    print(f"  File : {json_name}")
    print(f"{'─'*55}")
    print(f"  Before → images: {len(images)}, "
          f"annotations: {len(annotations)}, "
          f"categories: {len(categories)}")

    issues = defaultdict(list)

    # ── 1. Duplicate image IDs ─────────────────────────────────
    seen_img_ids = set()
    clean_images = []
    for img in images:
        if img["id"] in seen_img_ids:
            issues["duplicate_image_id"].append(img["id"])
        else:
            seen_img_ids.add(img["id"])
            clean_images.append(img)
    images = clean_images

    # ── 2. Images missing on disk ──────────────────────────────
    clean_images = []
    for img in images:
        fname = Path(img["file_name"]).name     # strip sub-path prefixes if any
        if fname not in images_on_disk:
            issues["image_missing_on_disk"].append(img["file_name"])
        else:
            img["file_name"] = fname            # normalise to flat filename
            clean_images.append(img)
    images = clean_images

    # ── 3. Valid ID sets ───────────────────────────────────────
    valid_image_ids = {img["id"] for img in images}
    valid_cat_ids   = {cat["id"] for cat in categories}

    # ── 4. Duplicate annotation IDs ───────────────────────────
    seen_ann_ids = set()
    clean_anns = []
    for ann in annotations:
        if ann["id"] in seen_ann_ids:
            issues["duplicate_ann_id"].append(ann["id"])
        else:
            seen_ann_ids.add(ann["id"])
            clean_anns.append(ann)
    annotations = clean_anns

    # ── 5. Annotations referencing missing image_id ────────────
    clean_anns = []
    for ann in annotations:
        if ann["image_id"] not in valid_image_ids:
            issues["ann_missing_image_id"].append(
                {"ann_id": ann["id"], "image_id": ann["image_id"]}
            )
        else:
            clean_anns.append(ann)
    annotations = clean_anns

    # ── 6. Annotations with out-of-range category_id ──────────
    clean_anns = []
    for ann in annotations:
        if ann["category_id"] not in valid_cat_ids:
            issues["ann_invalid_category_id"].append(
                {"ann_id": ann["id"], "category_id": ann["category_id"],
                 "valid_ids": sorted(valid_cat_ids)}
            )
        else:
            clean_anns.append(ann)
    annotations = clean_anns

    # ── 7. Invalid bounding boxes ──────────────────────────────
    img_sizes = {
        img["id"]: (img.get("width", 0), img.get("height", 0))
        for img in images
    }
    clean_anns = []
    clamped = 0
    for ann in annotations:
        bbox = ann.get("bbox", [])

        if len(bbox) != 4:
            issues["bbox_wrong_length"].append({"ann_id": ann["id"], "bbox": bbox})
            continue

        x, y, w, h = [float(v) for v in bbox]

        if x < 0 or y < 0:
            issues["bbox_negative_origin"].append({"ann_id": ann["id"], "bbox": [x, y, w, h]})
            continue

        if w <= 0 or h <= 0:
            issues["bbox_zero_size"].append({"ann_id": ann["id"], "bbox": [x, y, w, h]})
            continue

        if w * h < 1.0:
            issues["bbox_tiny_area"].append({"ann_id": ann["id"], "area": round(w * h, 3)})
            continue

        # Clamp to image boundary when size info is available
        img_w, img_h = img_sizes.get(ann["image_id"], (0, 0))
        if img_w > 0 and img_h > 0:
            new_w = round(min(x + w, img_w) - x, 2)
            new_h = round(min(y + h, img_h) - y, 2)
            if new_w <= 0 or new_h <= 0:
                issues["bbox_zero_after_clamp"].append({"ann_id": ann["id"]})
                continue
            if new_w != w or new_h != h:
                ann["bbox"] = [x, y, new_w, new_h]
                clamped += 1

        clean_anns.append(ann)
    annotations = clean_anns

    # ── 8. Remap IDs to be contiguous (1…N) ───────────────────
    old_to_new_img_id = {}
    for new_id, img in enumerate(images, start=1):
        old_to_new_img_id[img["id"]] = new_id
        img["id"] = new_id

    for new_id, ann in enumerate(annotations, start=1):
        ann["image_id"] = old_to_new_img_id.get(ann["image_id"], ann["image_id"])
        ann["id"] = new_id

    # ── Print summary ──────────────────────────────────────────
    print(f"  After  → images: {len(images)}, "
          f"annotations: {len(annotations)}, "
          f"categories: {len(categories)}")

    if clamped:
        print(f"  Clamped {clamped} bboxes to image boundary.")

    total = sum(len(v) for v in issues.values())
    if total == 0:
        print("  ✓ No issues found.")
    else:
        print(f"  Issues fixed: {total}")
        for issue_type, items in issues.items():
            print(f"    • {issue_type}: {len(items)}")

    return {
        **{k: v for k, v in data.items() if k not in ("images", "annotations", "categories")},
        "images":      images,
        "annotations": annotations,
        "categories":  categories,
    }


def main():
    parser = argparse.ArgumentParser(description="Clean DocLayNet COCO dataset.")
    parser.add_argument("--data_dir", required=True,
                        help="Root folder containing images/ and annotations/.")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    ann_dir    = data_dir / "annotations"
    images_dir = data_dir / "images"

    if not ann_dir.exists():
        raise FileNotFoundError(f"annotations/ not found in {data_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"images/ not found in {data_dir}")

    images_on_disk = get_images_on_disk(images_dir)
    print(f"Found {len(images_on_disk)} images on disk in {images_dir}")

    # Process every .json in annotations/ except already-cleaned ones
    json_files = sorted(
        f for f in ann_dir.glob("*.json")
        if not f.stem.endswith("_clean")
    )

    if not json_files:
        print("No JSON files found in annotations/.")
        return

    for json_path in json_files:
        data    = load_json(json_path)
        cleaned = clean_coco(data, images_on_disk, json_path.name)
        out     = ann_dir / f"{json_path.stem}_clean.json"
        save_json(cleaned, out)

    print("\nDone.")


if __name__ == "__main__":
    main()