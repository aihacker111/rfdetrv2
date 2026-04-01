# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
TrainingPipeline — full training loop from scratch or from pretrained weights.

Usage:
    from rfdetrv2.pipeline import TrainingPipeline
    from rfdetrv2.config import RFDETRV2BaseConfig, TrainConfig

    model_cfg = RFDETRV2BaseConfig(pretrain_weights="rf-detr-base.pth")
    train_cfg = TrainConfig(dataset_dir="./data", epochs=50)

    pipeline = TrainingPipeline(model_cfg)
    pipeline.run(train_cfg)
"""

from __future__ import annotations

import datetime
import json
import math
import multiprocessing
import os
import random
import shutil
import time
import warnings
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Callable, DefaultDict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import rfdetrv2.util.misc as utils
from rfdetrv2.config import ModelConfig, TrainConfig, pydantic_dump
from rfdetrv2.core.trainer import train_one_epoch
from rfdetrv2.core.evaluator import evaluate
from rfdetrv2.datasets import build_dataset, get_coco_api_from_dataset
from rfdetrv2.models import PostProcess, build_criterion_and_postprocessors
from rfdetrv2.pipeline.base import BasePipeline
from rfdetrv2.util.drop_scheduler import drop_scheduler
from rfdetrv2.util.get_param_dicts import get_param_dict
from rfdetrv2.util.utils import BestMetricHolder, ModelEma, clean_state_dict

logger = getLogger(__name__)


class TrainingPipeline(BasePipeline):
    """
    End-to-end training pipeline for RF-DETR models.

    Handles:
      - Dataset loading (COCO / YOLO / Roboflow layouts)
      - Optimizer & LR scheduler construction
      - EMA model management
      - Checkpoint saving (best + periodic)
      - Distributed training (DDP)
      - Logging (TensorBoard / W&B)
      - Callbacks (on_fit_epoch_end, on_train_end, on_train_batch_start)
    """

    SUPPORTED_CALLBACKS = frozenset([
        "on_fit_epoch_end",
        "on_train_batch_start",
        "on_train_end",
    ])

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.postprocess = PostProcess(num_select=getattr(model_config, "num_select", 300))
        self.ema_model: Optional[ModelEma] = None
        self.stop_early: bool = False
        self.callbacks: DefaultDict[str, List[Callable]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, train_config: TrainConfig, **extra_kwargs) -> None:
        """
        Execute the full training loop.

        Args:
            train_config: A ``TrainConfig`` instance with all training hyper-parameters.
            **extra_kwargs: Any extra kwargs forwarded to internal helpers
                            (e.g. ``min_batches`` for tiny datasets).
        """
        self._validate_callbacks()
        args = self._build_args(train_config)
        # Checkpoint ``args`` must match the built model / dataset (avoid stale e.g. num_classes=90).
        args.num_classes = self.model_config.num_classes
        if getattr(train_config, "class_names", None) is not None:
            args.class_names = train_config.class_names
        if getattr(train_config, "label_to_cat_id", None) is not None:
            args.label_to_cat_id = list(train_config.label_to_cat_id)

        utils.init_distributed_mode(args)
        self._seed(args)

        criterion, postprocess = build_criterion_and_postprocessors(args)

        if getattr(args, "freeze_encoder", False):
            self.freeze_encoder()

        model = self.model.to(self.device)
        model, model_without_ddp = self._wrap_ddp(model, args)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %d", n_params)

        self._scale_lr_for_multi_gpu(args)

        optimizer = self._build_optimizer(args, model_without_ddp)
        data_loaders = self._build_data_loaders(args, extra_kwargs)
        lr_scheduler = self._build_lr_scheduler(optimizer, args, len(data_loaders["train"].dataset))

        if args.use_ema:
            self.ema_model = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)

        if args.resume:
            self._resume_from_checkpoint(args, model_without_ddp, optimizer, lr_scheduler)

        if args.eval:
            self._run_eval_only(model, criterion, postprocess, data_loaders, args)
            return

        self._train_loop(
            model, model_without_ddp, criterion, postprocess,
            optimizer, lr_scheduler, data_loaders, args, n_params,
        )

    def request_early_stop(self) -> None:
        """Signal the pipeline to stop after the current epoch completes."""
        self.stop_early = True
        logger.info("Early stop requested — will finish the current epoch.")

    def add_callback(self, event: str, fn: Callable) -> None:
        """Register a callback for a training event."""
        if event not in self.SUPPORTED_CALLBACKS:
            raise ValueError(
                f"Unknown callback event '{event}'. "
                f"Supported: {sorted(self.SUPPORTED_CALLBACKS)}"
            )
        self.callbacks[event].append(fn)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        model, model_without_ddp, criterion, postprocess,
        optimizer, lr_scheduler, data_loaders, args, n_params,
    ):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        effective_batch = args.batch_size * args.grad_accum_steps
        total_batch = effective_batch * utils.get_world_size()
        steps_per_epoch = max(1, (len(data_loaders["train"].dataset) + total_batch - 1) // total_batch)

        schedules = self._build_drop_schedules(args, steps_per_epoch)

        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        best_map_5095, best_map_ema_5095 = 0.0, 0.0
        test_stats = ema_test_stats = {}

        logger.info("Starting training for %d epochs.", args.epochs)
        start_time = time.time()

        for epoch in range(args.start_epoch, args.epochs):
            epoch_start = time.time()

            if args.distributed:
                # With BatchSampler, loader.sampler is SequentialSampler over batch indices;
                # DistributedSampler lives on batch_sampler.sampler.
                self._set_distributed_sampler_epoch(data_loaders["train"], epoch)

            model.train()
            criterion.train()

            train_stats = train_one_epoch(
                model=model,
                criterion=criterion,
                lr_scheduler=lr_scheduler,
                data_loader=data_loaders["train"],
                optimizer=optimizer,
                device=self.device,
                epoch=epoch,
                batch_size=effective_batch,
                max_norm=args.clip_max_norm,
                ema_m=self.ema_model,
                schedules=schedules,
                num_training_steps_per_epoch=steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers,
                args=args,
                callbacks=self.callbacks,
            )

            epoch_time_str = str(datetime.timedelta(seconds=int(time.time() - epoch_start)))

            self._save_checkpoint(
                output_dir, epoch, args,
                model_without_ddp, optimizer, lr_scheduler,
            )

            with torch.no_grad():
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocess,
                    data_loaders["val"], data_loaders["val_ds"],
                    self.device, args=args,
                )

            map_key = "coco_eval_masks" if args.segmentation_head else "coco_eval_bbox"
            map_regular = test_stats[map_key][0]
            if best_map_holder.update(map_regular, epoch, is_ema=False):
                best_map_5095 = max(best_map_5095, map_regular)
                self._save_best_checkpoint(
                    output_dir / "checkpoint_best_regular.pth",
                    epoch, args, model_without_ddp, optimizer, lr_scheduler,
                )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_params,
                "train_epoch_time": epoch_time_str,
                "epoch_time": str(datetime.timedelta(seconds=int(time.time() - epoch_start))),
            }

            if args.use_ema:
                ema_test_stats, _ = evaluate(
                    self.ema_model.module, criterion, postprocess,
                    data_loaders["val"], data_loaders["val_ds"],
                    self.device, args=args,
                )
                log_stats.update({f"ema_test_{k}": v for k, v in ema_test_stats.items()})
                map_ema = ema_test_stats[map_key][0]
                best_map_ema_5095 = max(best_map_ema_5095, map_ema)
                if best_map_holder.update(map_ema, epoch, is_ema=True):
                    self._save_best_checkpoint(
                        output_dir / "checkpoint_best_ema.pth",
                        epoch, args, self.ema_model.module, optimizer, lr_scheduler,
                    )

            log_stats.update(best_map_holder.summary())

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            for cb in self.callbacks["on_fit_epoch_end"]:
                cb(log_stats)

            if self.stop_early:
                logger.info("Early stop triggered at epoch %d.", epoch)
                break

        self._finalize(output_dir, best_map_5095, best_map_ema_5095, test_stats, ema_test_stats, args)
        self._run_test_split(model, criterion, postprocess, data_loaders, output_dir, args)

        total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info("Training complete. Total time: %s", total_time)

        for cb in self.callbacks["on_train_end"]:
            cb()

    @staticmethod
    def _set_distributed_sampler_epoch(train_loader: DataLoader, epoch: int) -> None:
        """Advance epoch on DistributedSampler (plain or wrapped by BatchSampler)."""
        bs = getattr(train_loader, "batch_sampler", None)
        if bs is not None:
            inner = getattr(bs, "sampler", None)
            if inner is not None and hasattr(inner, "set_epoch"):
                inner.set_epoch(epoch)
                return
        sampler = getattr(train_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_args(self, train_config: TrainConfig):
        """Merge ModelConfig + TrainConfig into a flat argparse.Namespace."""
        from rfdetrv2.main import populate_args

        model_dict = pydantic_dump(self.model_config)
        train_dict = pydantic_dump(train_config)

        # TrainConfig fields win over ModelConfig on overlap
        merged = {**model_dict, **train_dict}

        # Ensure coco_path fallback
        if merged.get("coco_path") is None:
            merged["coco_path"] = merged.get("dataset_dir")

        return populate_args(**merged)

    def _build_optimizer(self, args, model_without_ddp):
        param_dicts = get_param_dict(args, model_without_ddp)
        param_dicts = [p for p in param_dicts if p["params"].requires_grad]
        return torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    def _build_lr_scheduler(self, optimizer, args, n_train_samples: int):
        total_batch = args.batch_size * utils.get_world_size() * args.grad_accum_steps
        steps_per_epoch = max(1, (n_train_samples + total_batch - 1) // total_batch)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(steps_per_epoch * args.warmup_epochs)
        lr_gamma = getattr(args, "lr_gamma", 0.1)
        milestones = [m * steps_per_epoch for m in getattr(args, "lr_milestones", [80, 160])]

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            sched = args.lr_scheduler
            min_f = getattr(args, "lr_min_factor", 0.05)
            if sched == "cosine":
                return min_f + (1 - min_f) * 0.5 * (1 + math.cos(math.pi * progress))
            if sched == "linear":
                return 1.0 - progress * (1.0 - min_f)
            if sched == "step":
                return 1.0 if step < args.lr_drop * steps_per_epoch else lr_gamma
            if sched == "multistep":
                return lr_gamma ** sum(1 for ms in milestones if step >= ms)
            if sched == "wsd":
                stable = getattr(args, "lr_stable_ratio", 0.7)
                if progress < stable:
                    return 1.0
                dp = (progress - stable) / max(1e-8, 1.0 - stable)
                return min_f + (1 - min_f) * 0.5 * (1 + math.cos(math.pi * dp))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _build_data_loaders(self, args, extra_kwargs: dict) -> dict:
        dataset_train = build_dataset("train", args, args.resolution)
        dataset_val = build_dataset("val", args, args.resolution)
        image_set_test = "test" if args.dataset_file == "roboflow" else "val"
        dataset_test = build_dataset(image_set_test, args, args.resolution)

        debug_limit = int(getattr(args, "debug_data_limit", 0) or 0)
        if debug_limit > 0:
            train_n = min(debug_limit, len(dataset_train))
            val_n = min(debug_limit, len(dataset_val))
            test_n = min(debug_limit, len(dataset_test))
            dataset_train = torch.utils.data.Subset(dataset_train, range(train_n))
            dataset_val = torch.utils.data.Subset(dataset_val, range(val_n))
            dataset_test = torch.utils.data.Subset(dataset_test, range(test_n))
            logger.warning(
                "debug_data_limit=%d → using train=%d, val=%d, test=%d images (smoke test).",
                debug_limit,
                train_n,
                val_n,
                test_n,
            )

        # Tiny dataset guard
        effective_bs = args.batch_size * args.grad_accum_steps
        min_batches = extra_kwargs.get("min_batches", 5)

        num_workers = self._resolve_num_workers(args.num_workers)

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if len(dataset_train) < effective_bs * min_batches:
            logger.warning(
                "Dataset is very small (%d samples). Using replacement sampler.", len(dataset_train)
            )
            if args.distributed:
                # Keep DistributedSampler so set_epoch / sharding stay correct (no RandomSampler).
                train_sampler = DistributedSampler(dataset_train, shuffle=True)
            else:
                train_sampler = torch.utils.data.RandomSampler(
                    dataset_train, replacement=True, num_samples=effective_bs * min_batches
                )
            loader_train = DataLoader(
                dataset_train, batch_size=effective_bs,
                sampler=train_sampler, collate_fn=utils.collate_fn,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=num_workers > 0,
            )
        else:
            batch_sampler = torch.utils.data.BatchSampler(sampler_train, effective_bs, drop_last=True)
            loader_train = DataLoader(
                dataset_train, batch_sampler=batch_sampler,
                collate_fn=utils.collate_fn, num_workers=num_workers,
                pin_memory=True, persistent_workers=num_workers > 0,
            )

        loader_val = DataLoader(
            dataset_val, args.batch_size, sampler=sampler_val,
            drop_last=False, collate_fn=utils.collate_fn,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        loader_test = DataLoader(
            dataset_test, args.batch_size,
            sampler=torch.utils.data.SequentialSampler(dataset_test),
            drop_last=False, collate_fn=utils.collate_fn,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        return {
            "train": loader_train,
            "val": loader_val,
            "test": loader_test,
            "val_ds": get_coco_api_from_dataset(dataset_val),
            "test_ds": get_coco_api_from_dataset(dataset_test),
        }

    @staticmethod
    def _build_drop_schedules(args, steps_per_epoch: int) -> dict:
        schedules = {}
        if getattr(args, "dropout", 0) > 0:
            schedules["do"] = drop_scheduler(
                args.dropout, args.epochs, steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule,
            )
        if getattr(args, "drop_path", 0) > 0:
            schedules["dp"] = drop_scheduler(
                args.drop_path, args.epochs, steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule,
            )
        return schedules

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, output_dir, epoch, args, model_without_ddp, optimizer, lr_scheduler):
        if not utils.is_main_process():
            return
        paths = [output_dir / "checkpoint.pth"]
        if (epoch + 1) % args.checkpoint_interval == 0:
            paths.append(output_dir / f"checkpoint{epoch:04d}.pth")
        weights = self._build_weights_dict(epoch, args, model_without_ddp, optimizer, lr_scheduler)
        for path in paths:
            utils.save_on_master(weights, path)

    def _save_best_checkpoint(self, path, epoch, args, model_without_ddp, optimizer, lr_scheduler):
        if not utils.is_main_process() or args.dont_save_weights:
            return
        weights = self._build_weights_dict(epoch, args, model_without_ddp, optimizer, lr_scheduler)
        utils.save_on_master(weights, path)

    def _build_weights_dict(self, epoch, args, model_without_ddp, optimizer, lr_scheduler) -> dict:
        weights = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if self.ema_model is not None:
            weights["ema_model"] = self.ema_model.module.state_dict()
        cn = getattr(args, "class_names", None)
        if cn:
            if isinstance(cn, (list, tuple)):
                weights["class_names"] = list(cn)
            elif isinstance(cn, dict):
                keys = sorted(int(k) for k in cn.keys())
                weights["class_names"] = [str(cn[k]) for k in keys]
        lt = getattr(args, "label_to_cat_id", None)
        if lt:
            weights["label_to_cat_id"] = list(lt)
        return weights

    def _finalize(self, output_dir, best_map_5095, best_map_ema_5095, test_stats, ema_test_stats, args):
        if not utils.is_main_process():
            return
        best_is_ema = best_map_ema_5095 > best_map_5095
        src = "checkpoint_best_ema.pth" if best_is_ema else "checkpoint_best_regular.pth"
        dst = output_dir / "checkpoint_best_total.pth"
        shutil.copy2(output_dir / src, dst)
        utils.strip_checkpoint(str(dst))

        results = (ema_test_stats if best_is_ema else test_stats).get("results_json", {})
        results["class_map"] = {"valid": results.get("class_map", [])}
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f)

        logger.info("Best checkpoint saved to: %s", dst)
        logger.info("Results saved to: %s", output_dir / "results.json")

        # Update self.model to best weights
        if best_is_ema and self.ema_model is not None:
            self.model = self.ema_model.module
        self.model.eval()

    def _run_test_split(self, model, criterion, postprocess, data_loaders, output_dir, args):
        if not getattr(args, "run_test", False):
            return
        best_state = torch.load(
            output_dir / "checkpoint_best_total.pth",
            map_location="cpu", weights_only=False,
        )["model"]
        model.load_state_dict(best_state)
        model.eval()
        test_stats, _ = evaluate(
            model, criterion, postprocess,
            data_loaders["test"], data_loaders["test_ds"],
            self.device, args=args,
        )
        with open(output_dir / "results.json", "r") as f:
            results = json.load(f)
        results["class_map"]["test"] = test_stats["results_json"]["class_map"]
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _validate_callbacks(self):
        for key in self.callbacks:
            if key not in self.SUPPORTED_CALLBACKS:
                raise ValueError(
                    f"Unknown callback '{key}'. Supported: {sorted(self.SUPPORTED_CALLBACKS)}"
                )

    def _seed(self, args):
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _wrap_ddp(self, model, args):
        model_without_ddp = model
        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=False
            )
            model_without_ddp = model.module
        return model, model_without_ddp

    def _scale_lr_for_multi_gpu(self, args):
        world_size = utils.get_world_size()
        if world_size <= 1:
            return
        mode = getattr(args, "lr_scale_mode", "sqrt")
        scale = math.sqrt(world_size) if mode == "sqrt" else float(world_size)
        args.lr *= scale
        args.lr_encoder *= scale
        logger.info("Multi-GPU LR scaled by %.3f → lr=%.2e, lr_encoder=%.2e", scale, args.lr, args.lr_encoder)

    def _resume_from_checkpoint(self, args, model_without_ddp, optimizer, lr_scheduler):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(ckpt["model"], strict=True)
        if args.use_ema:
            if "ema_model" in ckpt:
                self.ema_model.module.load_state_dict(clean_state_dict(ckpt["ema_model"]))
            else:
                self.ema_model = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)
        if not args.eval and all(k in ckpt for k in ("optimizer", "lr_scheduler", "epoch")):
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            args.start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from epoch %d.", ckpt.get("epoch", 0))

    def _run_eval_only(self, model, criterion, postprocess, data_loaders, args):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats, coco_eval = evaluate(
            model, criterion, postprocess,
            data_loaders["val"], data_loaders["val_ds"],
            self.device, args=args,
        )
        key = "coco_eval_masks" if args.segmentation_head else "coco_eval_bbox"
        utils.save_on_master(coco_eval.coco_eval[key.split("_")[-1]].eval, output_dir / "eval.pth")

    @staticmethod
    def _resolve_num_workers(requested: int) -> int:
        if requested > 0 and multiprocessing.get_start_method(allow_none=True) == "spawn":
            import __main__
            if not getattr(__main__, "__file__", None) or __main__.__name__ != "__main__":
                warnings.warn(
                    "Setting num_workers=0 because the script is not wrapped in "
                    "`if __name__ == '__main__':`. Required for multiprocessing with 'spawn'.",
                    RuntimeWarning,
                )
                return 0
        return requested
