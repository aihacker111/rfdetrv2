# ------------------------------------------------------------------------
# Supervised training: one ``Pipeline`` class plus loop primitives.
# ------------------------------------------------------------------------

from rfdetrv2.cfg.loader import load_run_config
from rfdetrv2.runner.inference import IMAGENET_MEAN, IMAGENET_STD, predict_detections
from rfdetrv2.runner.loops import evaluate, train_one_epoch
from rfdetrv2.runner.trainer import HOSTED_MODELS, Pipeline, download_pretrain_weights
from rfdetrv2.utils.rfdetr_pretrained import resolve_pretrain_weights_path


def load_config(path=None, **kwargs):
    """Merge packaged defaults, optional YAML at ``path``, and ``kwargs`` into a run namespace."""
    return load_run_config(path, **kwargs)


__all__ = [
    "HOSTED_MODELS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "Pipeline",
    "download_pretrain_weights",
    "evaluate",
    "load_config",
    "predict_detections",
    "resolve_pretrain_weights_path",
    "train_one_epoch",
]
