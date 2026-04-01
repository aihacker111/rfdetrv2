# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
FinetunePipeline — fine-tune a pretrained RF-DETR on a custom dataset.

Usage:
    from rfdetrv2.pipeline import FinetunePipeline
    from rfdetrv2.config import RFDETRV2BaseConfig, FineTuneConfig

    model_cfg  = RFDETRV2BaseConfig(pretrain_weights="rf-detr-base.pth")
    tune_cfg   = FineTuneConfig(
        dataset_dir="./data",
        epochs=20,
        freeze_encoder=True,
        unfreeze_at_epoch=5,
        lr=1e-4,
        lr_encoder=1e-5,
    )

    pipeline = FinetunePipeline(model_cfg)
    pipeline.run(tune_cfg)
"""

from __future__ import annotations

from logging import getLogger
from typing import Optional

from rfdetrv2.config import FineTuneConfig, ModelConfig
from rfdetrv2.pipeline.train import TrainingPipeline

logger = getLogger(__name__)


class FinetunePipeline(TrainingPipeline):
    """
    Fine-tuning pipeline with optional two-phase training:

      Phase 1 (warm-up):  encoder frozen, only decoder + head updated.
      Phase 2 (finetune): encoder unfrozen, full-model update with lower LR.

    Both phases are controlled via ``FineTuneConfig``.
    If ``unfreeze_at_epoch`` is None, the encoder stays frozen for all epochs
    when ``freeze_encoder`` is True.
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

    def run(self, train_config: FineTuneConfig, **extra_kwargs) -> None:  # type: ignore[override]
        """
        Execute the fine-tuning loop.

        Extends ``TrainingPipeline.run()`` by:
          1. Injecting an epoch-level callback that unfreezes the encoder at
             ``train_config.unfreeze_at_epoch`` if it was initially frozen.
          2. Using LoRA on the backbone if ``backbone_lora=True``.
        """
        unfreeze_epoch: Optional[int] = getattr(train_config, "unfreeze_at_epoch", None)

        if train_config.freeze_encoder and unfreeze_epoch is not None:
            self._register_unfreeze_callback(unfreeze_epoch)
            logger.info(
                "Fine-tune: encoder frozen for epochs 0..%d, then unfrozen.", unfreeze_epoch - 1
            )

        if getattr(train_config, "backbone_lora", False):
            self._apply_lora()

        super().run(train_config, **extra_kwargs)

    # ------------------------------------------------------------------
    # LoRA helper
    # ------------------------------------------------------------------

    def _apply_lora(self) -> None:
        """Wrap the DINOv3 backbone encoder with LoRA adapters."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "backbone_lora=True requires the `peft` package. "
                "Install it with: pip install peft"
            ) from exc

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            use_dora=True,
            target_modules=["q_proj", "v_proj", "k_proj", "qkv", "query", "key", "value"],
        )
        self.model.backbone[0].encoder = get_peft_model(
            self.model.backbone[0].encoder, lora_config
        )
        logger.info("LoRA adapters applied to DINOv3 backbone.")

    # ------------------------------------------------------------------
    # Epoch-level unfreeze callback
    # ------------------------------------------------------------------

    def _register_unfreeze_callback(self, unfreeze_epoch: int) -> None:
        """Register a callback that unfreezes the encoder at a given epoch."""

        pipeline_ref = self

        def _maybe_unfreeze(log_stats: dict) -> None:
            if log_stats.get("epoch") == unfreeze_epoch:
                pipeline_ref.unfreeze_encoder()
                logger.info("Fine-tune phase 2: encoder unfrozen at epoch %d.", unfreeze_epoch)

        self.add_callback("on_fit_epoch_end", _maybe_unfreeze)
