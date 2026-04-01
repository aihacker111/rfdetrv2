# rfdetrv2.core — internal building blocks used by all pipelines
#
# Import trainer + evaluator before model: ``build_rf_model`` pulls in ``main``,
# and ``main`` imports ``evaluate`` / ``train_one_epoch`` from this package — if
# ``model`` were imported first, ``core`` would still be partially initialized.
from rfdetrv2.core.trainer import train_one_epoch
from rfdetrv2.core.evaluator import (
    evaluate,
    coco_extended_metrics,
    map_eval_labels_to_coco,
    map_eval_labels_to_coco_category_ids,
)
from rfdetrv2.core.model import build_rf_model

__all__ = [
    "train_one_epoch",
    "evaluate",
    "coco_extended_metrics",
    "map_eval_labels_to_coco",
    "map_eval_labels_to_coco_category_ids",
    "build_rf_model",
]
