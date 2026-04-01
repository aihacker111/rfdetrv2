# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
evaluate — validation / test loop with COCO metrics.

Also exports:
  coco_extended_metrics   — precision/recall via confidence threshold sweep
  map_eval_labels_to_coco — label → COCO category_id remapping
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

import rfdetrv2.util.misc as utils
from rfdetrv2.datasets.coco_eval import CocoEvaluator

try:
    from torch.amp import autocast
    _LEGACY_AMP = False
except ImportError:
    from torch.cuda.amp import autocast  # type: ignore[no-redef]
    _LEGACY_AMP = True


def _autocast_args(args) -> dict:
    if args is None:
        return {"enabled": False, "dtype": torch.bfloat16}
    if _LEGACY_AMP:
        return {"enabled": getattr(args, "amp", False), "dtype": torch.bfloat16}
    return {"device_type": "cuda", "enabled": getattr(args, "amp", False), "dtype": torch.bfloat16}


# ------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------

def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    """
    Run the validation loop and return COCO metrics.

    Args:
        model:        The detection model (eval mode is set internally).
        criterion:    Loss function (eval mode set internally).
        postprocess:  ``PostProcess`` instance for converting logits → boxes.
        data_loader:  Validation DataLoader.
        base_ds:      COCO API ground-truth dataset.
        device:       Target device.
        args:         Flat training args namespace.

    Returns:
        Tuple ``(stats_dict, coco_evaluator)``.
        ``stats_dict`` contains averaged losses + ``coco_eval_bbox``
        (and optionally ``coco_eval_masks``) arrays, plus ``results_json``.
    """
    model.eval()
    criterion.eval()
    if args is not None and getattr(args, "fp16_eval", False):
        model.half()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))

    iou_types = ("bbox",) if not getattr(args, "segmentation_head", False) else ("bbox", "segm")
    max_dets = getattr(args, "eval_max_dets", 500) if args is not None else 500
    coco_evaluator = CocoEvaluator(base_ds, iou_types, max_dets)

    label_to_cat_id = getattr(args, "label_to_cat_id", None) if args is not None else None

    for samples, targets in metric_logger.log_every(data_loader, 10, "Eval:"):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args is not None and getattr(args, "fp16_eval", False):
            samples.tensors = samples.tensors.half()

        with autocast(**_autocast_args(args)):
            outputs = model(samples)

        # Cast FP16 outputs back to FP32 for loss computation
        if args is not None and getattr(args, "fp16_eval", False):
            outputs = _cast_outputs_to_float(outputs)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}

        metric_logger.update(loss=sum(loss_scaled.values()), **loss_scaled, **loss_unscaled)
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocess(outputs, orig_sizes)

        if label_to_cat_id is not None:
            results = map_eval_labels_to_coco(results, label_to_cat_id)

        res = {t["image_id"].item(): r for t, r in zip(targets, results)}
        coco_evaluator.update(res)

    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["results_json"] = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])

    if utils.is_main_process():
        print("\nPer-class AP table (bbox):")
        print(stats["results_json"]["ap_per_class_markdown"])

    if "bbox" in iou_types:
        stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

    if "segm" in iou_types:
        seg_metrics = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
        if utils.is_main_process():
            print("\nPer-class AP table (segm):")
            print(seg_metrics["ap_per_class_markdown"])
        stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    return stats, coco_evaluator


# ------------------------------------------------------------------
# Extended metrics (precision/recall via confidence sweep)
# ------------------------------------------------------------------

def coco_extended_metrics(coco_eval) -> dict:
    """
    Compute per-class and macro precision/recall by sweeping confidence thresholds.

    Finds the threshold that maximises macro-F1, then reports precision/recall
    at that threshold alongside standard AP50 and AP50:95.

    Returns a dict with keys:
      ``class_map``, ``ap_per_class_table``, ``ap_per_class_markdown``,
      ``map``, ``precision``, ``recall``.
    """
    iou50_idx = np.argwhere(np.isclose(coco_eval.params.iouThrs, 0.50)).item()
    cat_ids = coco_eval.params.catIds
    area_idx, maxdet_idx = 0, 2

    # Unflatten evalImgs into {cat_id: {area_rng: {img_id: e}}}
    evalImgs_flat = {}
    for e in coco_eval.evalImgs:
        if e is None:
            continue
        evalImgs_flat.setdefault(e["category_id"], {}).setdefault(
            tuple(e["aRng"]), {}
        )[e["image_id"]] = e

    area_rng_all = tuple(coco_eval.params.areaRng[area_idx])

    # Gather per-class detection data
    per_class_data = []
    for cid in cat_ids:
        scores, matches, ignores, total_gt = [], [], [], 0
        for img_id in coco_eval.params.imgIds:
            e = evalImgs_flat.get(cid, {}).get(area_rng_all, {}).get(img_id)
            if e is None:
                continue
            total_gt += sum(1 for ig in e["gtIgnore"] if not ig)
            for d in range(len(e["dtIds"])):
                scores.append(e["dtScores"][d])
                matches.append(e["dtMatches"][iou50_idx, d])
                ignores.append(e["dtIgnore"][iou50_idx, d])
        per_class_data.append({
            "scores": np.array(scores),
            "matches": np.array(matches),
            "ignore": np.array(ignores, dtype=bool),
            "total_gt": total_gt,
        })

    classes_with_gt = [k for k, d in enumerate(per_class_data) if d["total_gt"] > 0]
    conf_sweep = _sweep_confidence(per_class_data, np.linspace(0, 1, 101), classes_with_gt)
    best = max(conf_sweep, key=lambda x: x["macro_f1"])

    map_50_95 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])

    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    per_class = []
    for k, cid in enumerate(cat_ids):
        p_slice = coco_eval.eval["precision"][:, :, k, area_idx, maxdet_idx]
        p_masked = np.where(p_slice > -1, p_slice, np.nan)
        ap_50_95 = float(np.nanmean(np.nanmean(p_masked, axis=1)))
        ap_50 = float(np.nanmean(p_masked[iou50_idx]))
        if any(np.isnan(v) for v in [ap_50_95, ap_50, best["per_class_prec"][k], best["per_class_rec"][k]]):
            continue
        per_class.append({
            "class": cat_id_to_name[int(cid)],
            "map@50:95": ap_50_95,
            "map@50": ap_50,
            "precision": best["per_class_prec"][k],
            "recall": best["per_class_rec"][k],
        })

    per_class.append({
        "class": "all",
        "map@50:95": map_50_95,
        "map@50": map_50,
        "precision": best["macro_precision"],
        "recall": best["macro_recall"],
    })

    # Build AP table
    rows_sorted = sorted(
        [r for r in per_class if r["class"] != "all"],
        key=lambda x: x["map@50:95"], reverse=True,
    )
    ap_rows = [{"rank": i + 1, "class": r["class"], "ap50_95": r["map@50:95"], "ap50": r["map@50"]}
               for i, r in enumerate(rows_sorted)]
    summary = next((r for r in per_class if r["class"] == "all"), None)
    if summary:
        ap_rows.append({"rank": "all", "class": "all", "ap50_95": summary["map@50:95"], "ap50": summary["map@50"]})

    md_lines = ["| rank | class | ap50_95 | ap50 |", "| --- | --- | ---: | ---: |"]
    md_lines += [f"| {r['rank']} | {r['class']} | {r['ap50_95']:.4f} | {r['ap50']:.4f} |" for r in ap_rows]

    return {
        "class_map": per_class,
        "ap_per_class_table": {"columns": ["rank", "class", "ap50_95", "ap50"], "rows": ap_rows},
        "ap_per_class_markdown": "\n".join(md_lines),
        "map": map_50,
        "precision": best["macro_precision"],
        "recall": best["macro_recall"],
    }


def _sweep_confidence(per_class_data: list, thresholds: np.ndarray, classes_with_gt: list) -> list:
    """Sweep confidence thresholds and compute macro precision/recall/F1 at each."""
    results = []
    for thresh in thresholds:
        prec_list, rec_list, f1_list = [], [], []
        for d in per_class_data:
            valid = (d["scores"] >= thresh) & ~d["ignore"]
            tp = np.sum(d["matches"][valid] != 0)
            fp = np.sum(d["matches"][valid] == 0)
            fn = d["total_gt"] - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            prec_list.append(p)
            rec_list.append(r)
            f1_list.append(f1)
        macro_prec = np.mean([prec_list[k] for k in classes_with_gt]) if classes_with_gt else 0.0
        macro_rec = np.mean([rec_list[k] for k in classes_with_gt]) if classes_with_gt else 0.0
        macro_f1 = np.mean([f1_list[k] for k in classes_with_gt]) if classes_with_gt else 0.0
        results.append({
            "confidence_threshold": thresh,
            "macro_f1": macro_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "per_class_prec": np.array(prec_list),
            "per_class_rec": np.array(rec_list),
        })
    return results


# ------------------------------------------------------------------
# Label remapping
# ------------------------------------------------------------------

def map_eval_labels_to_coco(results: List[dict], label_to_cat_id: Sequence[int]) -> List[dict]:
    """
    Remap model class indices (0…K-1) to COCO ``category_id`` values.

    Required when the dataset uses non-contiguous category IDs (common in
    Roboflow and custom COCO datasets).
    """
    if not label_to_cat_id:
        return results
    out = []
    for res in results:
        if "labels" not in res:
            out.append(res)
            continue
        lbl = res["labels"]
        mapped = torch.tensor(
            [label_to_cat_id[int(x.item())] for x in lbl],
            dtype=lbl.dtype, device=lbl.device,
        )
        out.append({**res, "labels": mapped})
    return out


# Backward-compatible name used elsewhere in the codebase
map_eval_labels_to_coco_category_ids = map_eval_labels_to_coco


# ------------------------------------------------------------------
# FP16 output cast helper
# ------------------------------------------------------------------

def _cast_outputs_to_float(outputs: dict) -> dict:
    """Cast all FP16 tensors in the outputs dict back to float32."""
    def _cast(obj):
        if isinstance(obj, torch.Tensor):
            return obj.float()
        if isinstance(obj, dict):
            return {k: _cast(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_cast(v) for v in obj]
        return obj
    return _cast(outputs)
