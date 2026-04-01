# ------------------------------------------------------------------------
# RF-DETR  —  pipeline package
# ------------------------------------------------------------------------

"""
``rfdetrv2.pipeline`` — high-level, one-call interfaces for every workflow.

``InferencePipeline`` is imported lazily so environments without ``supervision``
can still use training / export pipelines.
"""

from __future__ import annotations

from typing import Any

from rfdetrv2.pipeline.base import BasePipeline
from rfdetrv2.pipeline.train import TrainingPipeline
from rfdetrv2.pipeline.finetune import FinetunePipeline
from rfdetrv2.pipeline.export_pipeline import ExportPipeline

__all__ = [
    "BasePipeline",
    "TrainingPipeline",
    "FinetunePipeline",
    "InferencePipeline",
    "ExportPipeline",
]


def __getattr__(name: str) -> Any:
    if name == "InferencePipeline":
        from rfdetrv2.pipeline.inference import InferencePipeline as _IP
        return _IP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
