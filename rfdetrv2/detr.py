# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
High-level model classes: RFDETRV2, RFDETRV2Base, RFDETRV2Small, RFDETRV2Nano, RFDETRV2Large.
"""

from __future__ import annotations

import glob
import json
import os
from logging import getLogger
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from rfdetrv2.config import (
    FineTuneConfig,
    ModelConfig,
    RFDETRV2BaseConfig,
    RFDETRV2LargeConfig,
    RFDETRV2NanoConfig,
    RFDETRV2SmallConfig,
    TrainConfig,
    pydantic_dump,
)
from rfdetrv2.pipeline import (
    ExportPipeline,
    FinetunePipeline,
    InferencePipeline,
    TrainingPipeline,
)
from rfdetrv2.util.coco_classes import COCO_CLASSES
from rfdetrv2.util.metrics import MetricsPlotSink, MetricsTensorBoardSink, MetricsWandBSink

logger = getLogger(__name__)


class RFDETRV2:
    """
    Base RF-DETR model façade.

    Owns a single ``ModelConfig`` and lazily creates the appropriate pipeline
    on the first call to ``train()``, ``finetune()``, ``predict()``, or ``export()``.
    """

    size: Optional[str] = None

    def __init__(self, **kwargs):
        self.model_config: ModelConfig = self._default_model_config(**kwargs)
        self._class_names: Optional[dict] = None
        self._inference_pipeline: Optional[InferencePipeline] = None

    def _default_model_config(self, **kwargs) -> ModelConfig:
        return ModelConfig(**kwargs)

    def train(self, **kwargs) -> None:
        train_config = TrainConfig(**kwargs)
        train_config = self._prepare_class_info(train_config)

        pipeline = TrainingPipeline(self.model_config)
        self._register_logging_callbacks(pipeline, train_config)

        pipeline.run(train_config)
        self._inference_pipeline = None

    def finetune(self, **kwargs) -> None:
        tune_config = FineTuneConfig(**kwargs)
        tune_config = self._prepare_class_info(tune_config)

        pipeline = FinetunePipeline(self.model_config)
        self._register_logging_callbacks(pipeline, tune_config)

        pipeline.run(tune_config)
        self._inference_pipeline = None

    def predict(
        self,
        images: Union[str, Image.Image, np.ndarray, torch.Tensor,
                      List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
        threshold: float = 0.5,
    ) -> Union[Any, List[Any]]:
        pipeline = self._get_or_build_inference_pipeline()
        return pipeline.run(images, threshold=threshold)

    def optimize_for_inference(
        self,
        compile: bool = True,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        pipeline = self._get_or_build_inference_pipeline()
        pipeline.optimize(compile=compile, batch_size=batch_size, dtype=dtype)

    def remove_optimization(self) -> None:
        if self._inference_pipeline is not None:
            self._inference_pipeline.remove_optimization()

    def export(self, **kwargs) -> str:
        pipeline = ExportPipeline(self.model_config)
        return pipeline.run(**kwargs)

    def deploy_to_roboflow(
        self,
        workspace: str,
        project_id: str,
        version: str,
        api_key: Optional[str] = None,
        size: Optional[str] = None,
        output_dir: str = "output",
    ) -> None:
        import shutil
        from roboflow import Roboflow

        api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if api_key is None:
            raise ValueError("Provide api_key= or set the ROBOFLOW_API_KEY environment variable.")

        size = size or self.size
        if size is None:
            raise ValueError("Must set size= for custom model architectures.")

        pipeline = self._get_or_build_inference_pipeline()
        model_module = pipeline.model

        tmp_dir = os.path.join(output_dir, ".roboflow_upload")
        os.makedirs(tmp_dir, exist_ok=True)
        weights_path = os.path.join(tmp_dir, "weights.pt")

        torch.save({"model": model_module.state_dict()}, weights_path)

        rf = Roboflow(api_key=api_key)
        rf.workspace(workspace).project(project_id).version(version).deploy(
            model_type=size,
            model_path=tmp_dir,
            filename="weights.pt",
        )
        shutil.rmtree(tmp_dir)
        logger.info("Model deployed to Roboflow project %s/%s.", workspace, project_id)

    @property
    def class_names(self) -> dict:
        if self._class_names:
            return self._class_names
        return COCO_CLASSES

    @class_names.setter
    def class_names(self, value):
        self._class_names = value

    def _get_or_build_inference_pipeline(self) -> InferencePipeline:
        if self._inference_pipeline is None:
            self._inference_pipeline = InferencePipeline(self.model_config)
            self._inference_pipeline.class_names = self.class_names
        return self._inference_pipeline

    def _prepare_class_info(self, train_config: TrainConfig) -> TrainConfig:
        """Infer class names, count, and optional COCO id remap from the dataset."""
        import yaml

        dataset_dir = train_config.dataset_dir
        class_names: Optional[List[str]] = None
        num_classes: Optional[int] = None
        label_to_cat_id: Optional[List[int]] = None

        if train_config.dataset_file == "roboflow":
            coco_ann = os.path.join(dataset_dir, "train", "_annotations.coco.json")
            if os.path.exists(coco_ann):
                with open(coco_ann) as f:
                    anns = json.load(f)
                cats = sorted(
                    [c for c in anns["categories"] if c.get("supercategory") != "none"],
                    key=lambda c: int(c["id"]),
                )
                class_names = [c["name"] for c in cats]
                label_to_cat_id = [int(c["id"]) for c in cats]
                num_classes = len(class_names)
            else:
                yaml_files = glob.glob(os.path.join(dataset_dir, "data*.yaml")) + \
                             glob.glob(os.path.join(dataset_dir, "data*.yml"))
                if len(yaml_files) != 1:
                    raise FileNotFoundError(
                        f"Could not find class names in {dataset_dir}. "
                        "Expected COCO (train/_annotations.coco.json) or YOLO (single data*.yaml)."
                    )
                with open(yaml_files[0]) as f:
                    data = yaml.safe_load(f)
                names = data.get("names", {})
                if isinstance(names, dict):
                    class_names = [names[i] for i in sorted(names)]
                else:
                    class_names = list(names)
                num_classes = len(class_names)

        elif train_config.dataset_file == "coco":
            from rfdetrv2.datasets.coco import infer_coco_num_classes_and_names

            root = train_config.coco_path or dataset_dir
            inferred = infer_coco_num_classes_and_names(root)
            if inferred is None:
                raise RuntimeError(
                    f"Could not read class names from COCO JSON under {root}."
                )
            class_names, num_classes, label_to_cat_id = inferred

        elif train_config.dataset_file == "o365":
            return train_config

        else:
            raise ValueError(f"Unsupported dataset_file: {train_config.dataset_file}")

        if class_names is None or num_classes is None:
            return train_config

        self._class_names = {i + 1: n for i, n in enumerate(class_names)}

        if self.model_config.num_classes != num_classes:
            logger.info(
                "Updating model num_classes: %d → %d.",
                self.model_config.num_classes, num_classes,
            )
            self.model_config = self.model_config.model_copy(update={"num_classes": num_classes}) \
                if hasattr(self.model_config, "model_copy") \
                else self.model_config.copy(update={"num_classes": num_classes})

        updates: dict = {"class_names": class_names}
        if label_to_cat_id is not None:
            updates["label_to_cat_id"] = label_to_cat_id
        if hasattr(train_config, "model_copy"):
            return train_config.model_copy(update=updates)
        return train_config.copy(update=updates)

    def _register_logging_callbacks(
        self,
        pipeline: TrainingPipeline,
        train_config: TrainConfig,
    ) -> None:
        metrics_plot = MetricsPlotSink(output_dir=train_config.output_dir)
        pipeline.add_callback("on_fit_epoch_end", metrics_plot.update)
        pipeline.add_callback("on_train_end", metrics_plot.save)

        if train_config.tensorboard:
            tb_sink = MetricsTensorBoardSink(output_dir=train_config.output_dir)
            pipeline.add_callback("on_fit_epoch_end", tb_sink.update)
            pipeline.add_callback("on_train_end", tb_sink.close)

        if train_config.wandb:
            wb_sink = MetricsWandBSink(
                output_dir=train_config.output_dir,
                project=train_config.project,
                run=train_config.run,
                config=pydantic_dump(train_config),
            )
            pipeline.add_callback("on_fit_epoch_end", wb_sink.update)
            pipeline.add_callback("on_train_end", wb_sink.close)

        if train_config.early_stopping:
            from rfdetrv2.util.early_stopping import EarlyStoppingCallback
            early_stop_cb = EarlyStoppingCallback(
                model=pipeline,
                patience=train_config.early_stopping_patience,
                min_delta=train_config.early_stopping_min_delta,
                use_ema=train_config.early_stopping_use_ema,
                segmentation_head=train_config.segmentation_head,
            )
            pipeline.add_callback("on_fit_epoch_end", early_stop_cb.update)


class RFDETRV2Base(RFDETRV2):
    """RF-DETR Base — DINOv3-Base backbone."""
    size = "rfdetr-base"

    def _default_model_config(self, **kwargs) -> RFDETRV2BaseConfig:
        return RFDETRV2BaseConfig(**kwargs)


class RFDETRV2Small(RFDETRV2):
    """RF-DETR Small — DINOv3 ViT-S+ backbone."""
    size = "rfdetr-small"

    def _default_model_config(self, **kwargs) -> RFDETRV2SmallConfig:
        return RFDETRV2SmallConfig(**kwargs)


class RFDETRV2Nano(RFDETRV2):
    """RF-DETR Nano — DINOv3 ViT-S backbone."""
    size = "rfdetr-nano"

    def _default_model_config(self, **kwargs) -> RFDETRV2NanoConfig:
        return RFDETRV2NanoConfig(**kwargs)


class RFDETRV2Large(RFDETRV2):
    """RF-DETR Large — DINOv3-Base at 640px."""
    size = "rfdetr-large"

    def _default_model_config(self, **kwargs) -> RFDETRV2LargeConfig:
        return RFDETRV2LargeConfig(**kwargs)


# Backward-compatible aliases (prefer RFDETRV2*)
RFDETRBase = RFDETRV2Base
RFDETRSmall = RFDETRV2Small
RFDETRNano = RFDETRV2Nano
RFDETRLarge = RFDETRV2Large
