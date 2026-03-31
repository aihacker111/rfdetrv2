# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Training pipeline: builds ``LWDETR``, dataloaders, full fit loop, checkpointing, ONNX export.
"""
import copy
import datetime
import json
import math
import multiprocessing
import os
import random
import shutil
import time
import warnings
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Callable, DefaultDict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import rfdetrv2.utils.misc as utils
from rfdetrv2.cfg.loader import load_run_config, merge_namespace, namespace_to_dict
from rfdetrv2.data import build_dataset, get_coco_api_from_dataset
from rfdetrv2.models import PostProcess, build_criterion_and_postprocessors, build_model
from rfdetrv2.runner.inference import IMAGENET_MEAN, IMAGENET_STD, predict_detections
from rfdetrv2.runner.loops import evaluate, train_one_epoch
from rfdetrv2.utils.benchmark import benchmark
from rfdetrv2.utils.drop_scheduler import drop_scheduler
from rfdetrv2.utils.get_param_dicts import get_param_dict
from rfdetrv2.utils.rfdetr_pretrained import (
    HF_BASE_URL,
    RFDETR_COCO_CHECKPOINT_BY_SIZE,
    resolve_pretrain_weights_path,
)
from rfdetrv2.utils.utils import BestMetricHolder, ModelEma, clean_state_dict

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

logger = getLogger(__name__)

# HuggingFace COCO checkpoints (https://huggingface.co/myn0908/rfdetrv2) — for docs / tooling.
HOSTED_MODELS: dict[str, str] = {
    **{
        f"rfdetrv2_{k}": f"{HF_BASE_URL.rstrip('/')}/{v}?download=true"
        for k, v in RFDETR_COCO_CHECKPOINT_BY_SIZE.items()
    },
    **{
        v: f"{HF_BASE_URL.rstrip('/')}/{v}?download=true"
        for v in RFDETR_COCO_CHECKPOINT_BY_SIZE.values()
    },
}


def download_pretrain_weights(pretrain_weights: str, redownload: bool = False) -> str:
    """Resolve *pretrain_weights* to a local ``.pth`` (download HF / URL if needed)."""
    proj = Path(__file__).resolve().parents[2]
    return resolve_pretrain_weights_path(pretrain_weights, proj, redownload=redownload)


class Pipeline:
    def __init__(
        self,
        config: str | Path | None = None,
        cfg=None,
        **kwargs,
    ):
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = load_run_config(config, **kwargs)
        cfg = self.cfg
        self.resolution = cfg.resolution
        self.model = build_model(cfg)
        self.device = torch.device(cfg.device)

        if cfg.pretrain_weights is not None:
            print("Loading pretrain weights")
            weight_path = download_pretrain_weights(cfg.pretrain_weights, redownload=False)
            try:
                checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Failed to load pretrain weights: {e}")
                print("Failed to load pretrain weights, re-downloading")
                weight_path = download_pretrain_weights(cfg.pretrain_weights, redownload=True)
                checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
            try:
                setattr(cfg, "pretrain_weights", weight_path)
            except Exception:
                pass

            ck = checkpoint.get("args") or checkpoint.get("cfg")
            if ck is not None and hasattr(ck, "class_names"):
                self.cfg.class_names = ck.class_names
                self.class_names = ck.class_names
            elif isinstance(checkpoint.get("cfg"), dict):
                cn = checkpoint["cfg"].get("class_names")
                if cn is not None:
                    self.cfg.class_names = cn
                    self.class_names = cn

            checkpoint_num_classes = checkpoint['model']['class_embed.bias'].shape[0]
            if checkpoint_num_classes != cfg.num_classes + 1:
                self.reinitialize_detection_head(checkpoint_num_classes)

            if cfg.pretrain_exclude_keys is not None:
                assert isinstance(cfg.pretrain_exclude_keys, list)
                for exclude_key in cfg.pretrain_exclude_keys:
                    checkpoint['model'].pop(exclude_key)

            if cfg.pretrain_keys_modify_to_load is not None:
                from rfdetrv2.utils.obj365_to_coco_model import get_coco_pretrain_from_obj365
                assert isinstance(cfg.pretrain_keys_modify_to_load, list)
                model_wo = self.model
                for modify_key_to_load in cfg.pretrain_keys_modify_to_load:
                    try:
                        checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                            model_wo.state_dict()[modify_key_to_load],
                            checkpoint['model'][modify_key_to_load]
                        )
                    except Exception:
                        print(f"Failed to load {modify_key_to_load}, deleting from checkpoint")
                        checkpoint['model'].pop(modify_key_to_load)

            num_desired_queries = cfg.num_queries * cfg.group_detr
            query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
            for name, state in checkpoint['model'].items():
                if any(name.endswith(x) for x in query_param_names):
                    checkpoint['model'][name] = state[:num_desired_queries]

            self.model.load_state_dict(checkpoint['model'], strict=False)

        if cfg.backbone_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as exc:
                raise ImportError(
                    "backbone_lora=True requires `peft` and compatible `transformers`/`torch` versions. "
                    "Please install compatible packages or disable backbone_lora."
                ) from exc
            print("Applying LORA to backbone")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                use_dora=True,
                target_modules=[
                    "q_proj", "v_proj", "k_proj",
                    "qkv",
                    "query", "key", "value", "cls_token", "register_tokens",
                ]
            )
            self.model.backbone[0].encoder = get_peft_model(self.model.backbone[0].encoder, lora_config)

        self.model = self.model.to(self.device)
        self.postprocess = PostProcess(num_select=cfg.num_select)
        self.stop_early = False

        self.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size: int | None = None
        self._optimized_resolution: int | None = None
        self._optimized_dtype = None
        self.imagenet_mean = list(IMAGENET_MEAN)
        self.imagenet_std = list(IMAGENET_STD)

    @classmethod
    def from_pretrained(
        cls,
        weights: str | Path,
        config: str | Path | None = None,
        **kwargs,
    ) -> "Pipeline":
        """Build from a ``.pth`` checkpoint (sets ``pretrain_weights`` for load)."""
        return cls(config=config, pretrain_weights=str(weights), **kwargs)

    def _sync_frozen_params_from_cfg(self, cfg) -> None:
        """Apply ``freeze_encoder`` / ``freeze_decoder`` / ``freeze_detection_head`` from ``cfg`` to ``self.model``."""
        m = self.model
        if getattr(cfg, "freeze_encoder", False):
            backbone = m.backbone
            if hasattr(backbone, "__getitem__") and len(backbone) > 0 and hasattr(backbone[0], "encoder"):
                for p in backbone[0].encoder.parameters():
                    p.requires_grad = False
                if utils.is_main_process():
                    logger.info("Encoder frozen (freeze_encoder=True).")
        if getattr(cfg, "freeze_decoder", False):
            for n, p in m.named_parameters():
                if "transformer.decoder" in n:
                    p.requires_grad = False
            if utils.is_main_process():
                logger.info("Transformer decoder frozen (freeze_decoder=True).")
        if getattr(cfg, "freeze_detection_head", False):
            head_keys = (
                "class_embed",
                "bbox_embed",
                "refpoint_embed",
                "query_feat",
                "enc_out_bbox_embed",
                "enc_out_class_embed",
            )
            for n, p in m.named_parameters():
                if any(k in n for k in head_keys):
                    p.requires_grad = False
            if utils.is_main_process():
                logger.info("Detection head frozen (freeze_detection_head=True).")

    def apply_finetune_freeze(
        self,
        *,
        freeze_encoder: bool | None = None,
        freeze_decoder: bool | None = None,
        freeze_detection_head: bool | None = None,
    ) -> "Pipeline":
        """Update config and ``requires_grad`` for common detection fine-tuning patterns."""
        updates = {
            k: v
            for k, v in (
                ("freeze_encoder", freeze_encoder),
                ("freeze_decoder", freeze_decoder),
                ("freeze_detection_head", freeze_detection_head),
            )
            if v is not None
        }
        if updates:
            self.cfg = merge_namespace(self.cfg, **updates)
        self._sync_frozen_params_from_cfg(self.cfg)
        return self

    def predict(
        self,
        images,
        threshold: float = 0.5,
        **kwargs,
    ):
        """Run detection (same preprocessing as ``RFDETRV2.predict``)."""
        return predict_detections(self, images, threshold, **kwargs)

    def optimize_for_inference(self, compile=True, batch_size=1, dtype=torch.float32):
        """Deep-copy detector, call ``export()`` mode, optional ``torch.jit.trace``."""
        self.remove_optimized_model()
        self.inference_model = deepcopy(self.model)
        self.inference_model.eval()
        self.inference_model.export()
        self._optimized_resolution = self.resolution
        self._is_optimized_for_inference = True
        self.inference_model = self.inference_model.to(dtype=dtype)
        self._optimized_dtype = dtype
        if compile:
            self.inference_model = torch.jit.trace(
                self.inference_model,
                torch.randn(
                    batch_size,
                    3,
                    self.resolution,
                    self.resolution,
                    device=self.device,
                    dtype=dtype,
                ),
            )
            self._optimized_has_been_compiled = True
            self._optimized_batch_size = batch_size

    def remove_optimized_model(self) -> None:
        self.inference_model = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    def finetune(self, callbacks: DefaultDict[str, List[Callable]], **kwargs):
        """Fine-tuning entry point; same as :meth:`fit` (pass ``freeze_encoder`` etc. in ``kwargs``)."""
        return self._run_training(callbacks, **kwargs)

    def reinitialize_detection_head(self, num_classes):
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self):
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")

    def fit(self, callbacks: DefaultDict[str, List[Callable]], **kwargs):
        """Full supervised training (datasets, optimizer, epochs, checkpoints)."""
        return self._run_training(callbacks, **kwargs)

    def train(self, callbacks: DefaultDict[str, List[Callable]], **kwargs):
        """Same as :meth:`fit` (kept for callers that use ``.train(...)``)."""
        return self._run_training(callbacks, **kwargs)

    def _run_training(self, callbacks: DefaultDict[str, List[Callable]], **kwargs):
        currently_supported_callbacks = ["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]
        for key in callbacks.keys():
            if key not in currently_supported_callbacks:
                raise ValueError(
                    f"Callback {key} is not currently supported, please file an issue if you need it!\n"
                    f"Currently supported callbacks: {currently_supported_callbacks}"
                )

        if kwargs:
            self.cfg = merge_namespace(self.cfg, **kwargs)
        cfg = self.cfg
        if getattr(cfg, "class_names", None) is not None:
            self.cfg.class_names = cfg.class_names
            self.cfg.num_classes = cfg.num_classes

        utils.init_distributed_mode(cfg)
        print("git:\n  {}\n".format(utils.get_sha()))
        print(cfg)
        device = torch.device(cfg.device)

        seed = cfg.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        criterion, postprocess = build_criterion_and_postprocessors(cfg)

        self._sync_frozen_params_from_cfg(cfg)

        model = self.model
        model.to(device)

        model_without_ddp = model
        if cfg.distributed:
            if cfg.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.gpu], find_unused_parameters=False
            )
            model_without_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        world_size = utils.get_world_size()
        if world_size > 1:
            lr_scale_mode = getattr(cfg, "lr_scale_mode", "sqrt")
            if lr_scale_mode == 'sqrt':
                lr_scale = math.sqrt(world_size)
            else:
                lr_scale = float(world_size)
            cfg.lr = cfg.lr * lr_scale
            cfg.lr_encoder = cfg.lr_encoder * lr_scale
            if utils.is_main_process():
                print(
                    f'Multi-GPU: scaling LR by {lr_scale_mode}(world_size={world_size}) '
                    f'-> lr={cfg.lr:.2e}, lr_encoder={cfg.lr_encoder:.2e}'
                )

        param_dicts = get_param_dict(cfg, model_without_ddp)
        param_dicts = [p for p in param_dicts if p['params'].requires_grad]
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr, weight_decay=cfg.weight_decay)

        dataset_train = build_dataset(image_set="train", cfg=cfg, resolution=cfg.resolution)
        dataset_val   = build_dataset(image_set="val", cfg=cfg, resolution=cfg.resolution)
        dataset_test  = build_dataset(
            image_set="test" if cfg.dataset_file == "roboflow" else "val",
            cfg=cfg,
            resolution=cfg.resolution,
        )

        debug_data_limit = int(getattr(cfg, "debug_data_limit", 0) or 0)
        if debug_data_limit > 0:
            train_n = min(debug_data_limit, len(dataset_train))
            val_n   = min(debug_data_limit, len(dataset_val))
            test_n  = min(debug_data_limit, len(dataset_test))
            dataset_train = torch.utils.data.Subset(dataset_train, range(train_n))
            dataset_val   = torch.utils.data.Subset(dataset_val,   range(val_n))
            dataset_test  = torch.utils.data.Subset(dataset_test,  range(test_n))
            logger.info(
                "Debug data limit enabled: train=%d, val=%d, test=%d",
                train_n, val_n, test_n,
            )

        total_batch_size_for_lr = cfg.batch_size * utils.get_world_size() * cfg.grad_accum_steps
        num_training_steps_per_epoch_lr = (
            len(dataset_train) + total_batch_size_for_lr - 1
        ) // total_batch_size_for_lr
        total_training_steps_lr = num_training_steps_per_epoch_lr * cfg.epochs
        warmup_steps_lr         = num_training_steps_per_epoch_lr * cfg.warmup_epochs
        lr_gamma      = getattr(cfg, "lr_gamma", 0.1)
        lr_milestones = getattr(cfg, "lr_milestones", [80, 160])
        milestone_steps = [m * num_training_steps_per_epoch_lr for m in sorted(lr_milestones)]
        if cfg.lr_scheduler == 'multistep' and utils.is_main_process():
            print(f'LR scheduler: multistep, milestones={lr_milestones} epochs, gamma={lr_gamma}')

        def lr_lambda(current_step: int):
            # Phase 1: warmup
            if current_step < warmup_steps_lr:
                return float(current_step) / float(max(1, warmup_steps_lr))

            progress = float(current_step - warmup_steps_lr) / float(
                max(1, total_training_steps_lr - warmup_steps_lr)
            )

            if cfg.lr_scheduler == 'cosine':
                return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

            if cfg.lr_scheduler == 'wsd':
                # Warmup-Stable-Decay: stable phase then cosine decay
                stable_ratio = getattr(cfg, "lr_stable_ratio", 0.7)
                if progress < stable_ratio:
                    return 1.0
                decay_progress = (progress - stable_ratio) / max(1e-8, 1.0 - stable_ratio)
                return cfg.lr_min_factor + (1 - cfg.lr_min_factor) * 0.5 * (
                    1 + math.cos(math.pi * decay_progress)
                )

            if cfg.lr_scheduler == 'cosine_restart':
                # Cosine với warm restart mỗi restart_period epochs
                # Giúp thoát local minima khi plateau (epoch 20+)
                restart_period = getattr(cfg, "lr_restart_period", 50)
                restart_decay  = getattr(cfg, "lr_restart_decay", 0.8)
                restart_steps  = int(restart_period * num_training_steps_per_epoch_lr)
                elapsed        = current_step - warmup_steps_lr
                cycle_idx      = elapsed // max(1, restart_steps)
                cycle_progress = (elapsed % max(1, restart_steps)) / max(1, restart_steps)
                # LR peak giảm dần qua các cycle
                peak_factor    = restart_decay ** cycle_idx
                return peak_factor * (
                    cfg.lr_min_factor + (1 - cfg.lr_min_factor) * 0.5 * (
                        1 + math.cos(math.pi * cycle_progress)
                    )
                )

            if cfg.lr_scheduler == 'linear':
                return 1.0 - progress * (1.0 - cfg.lr_min_factor)

            if cfg.lr_scheduler == 'step':
                drop_step = cfg.lr_drop * num_training_steps_per_epoch_lr
                return 1.0 if current_step < drop_step else lr_gamma

            if cfg.lr_scheduler == 'multistep':
                gamma_exp = sum(1 for ms in milestone_steps if current_step >= ms)
                return lr_gamma ** gamma_exp

            return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        if cfg.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val   = DistributedSampler(dataset_val,  shuffle=False)
            sampler_test  = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test  = torch.utils.data.SequentialSampler(dataset_test)

        effective_batch_size = cfg.batch_size * cfg.grad_accum_steps
        if debug_data_limit > 0:
            min_batches = kwargs.get("min_batches", 1)
        else:
            min_batches = kwargs.get("min_batches", 5)

        num_workers = cfg.num_workers
        if num_workers > 0 and multiprocessing.get_start_method(allow_none=True) == 'spawn':
            import __main__
            if not hasattr(__main__, '__file__') or not __main__.__name__ == '__main__':
                warnings.warn(
                    "Setting num_workers to 0 because the script is not wrapped in "
                    "`if __name__ == '__main__':`. This is required for multiprocessing "
                    "with the 'spawn' start method.",
                    RuntimeWarning,
                )
                num_workers = 0

        if len(dataset_train) < effective_batch_size * min_batches:
            logger.info(
                f"Training with uniform sampler because dataset is too small: "
                f"{len(dataset_train)} < {effective_batch_size * min_batches}"
            )
            sampler = torch.utils.data.RandomSampler(
                dataset_train,
                replacement=True,
                num_samples=effective_batch_size * min_batches,
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_size=effective_batch_size,
                collate_fn=utils.collate_fn,
                num_workers=num_workers,
                sampler=sampler,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )
        else:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, effective_batch_size, drop_last=True
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )

        data_loader_val = DataLoader(
            dataset_val, cfg.batch_size, sampler=sampler_val,
            drop_last=False, collate_fn=utils.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        data_loader_test = DataLoader(
            dataset_test, cfg.batch_size, sampler=sampler_test,
            drop_last=False, collate_fn=utils.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        base_ds      = get_coco_api_from_dataset(dataset_val)
        base_ds_test = get_coco_api_from_dataset(dataset_test)

        if cfg.use_ema:
            self.ema_m = ModelEma(model_without_ddp, decay=cfg.ema_decay, tau=cfg.ema_tau)
        else:
            self.ema_m = None

        output_dir = Path(cfg.output_dir)

        if utils.is_main_process():
            print("Get benchmark")
            if cfg.do_benchmark:
                benchmark_model = copy.deepcopy(model_without_ddp)
                bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
                print(json.dumps(bm, indent=2))
                del benchmark_model

        # ----------------------------------------------------------------
        # Resume từ checkpoint
        # ----------------------------------------------------------------
        if cfg.resume:
            checkpoint = torch.load(cfg.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

            if cfg.use_ema:
                if 'ema_model' in checkpoint:
                    self.ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
                else:
                    del self.ema_m
                    self.ema_m = ModelEma(model, decay=cfg.ema_decay, tau=cfg.ema_tau)

            if (
                not cfg.eval
                and 'optimizer' in checkpoint
                and 'lr_scheduler' in checkpoint
                and 'epoch' in checkpoint
            ):
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                cfg.start_epoch = checkpoint['epoch'] + 1
        # ----------------------------------------------------------------

        if cfg.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocess, data_loader_val, base_ds, device, cfg
            )
            if cfg.output_dir:
                if not cfg.segmentation_head:
                    utils.save_on_master(
                        coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
                    )
                else:
                    utils.save_on_master(
                        coco_evaluator.coco_eval["segm"].eval, output_dir / "eval.pth"
                    )
            return

        total_batch_size = effective_batch_size * utils.get_world_size()
        num_training_steps_per_epoch = (
            len(dataset_train) + total_batch_size - 1
        ) // total_batch_size

        schedules = {}
        if cfg.dropout > 0:
            schedules['do'] = drop_scheduler(
                cfg.dropout, cfg.epochs, num_training_steps_per_epoch,
                cfg.cutoff_epoch, cfg.drop_mode, cfg.drop_schedule,
            )
            print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

        if cfg.drop_path > 0:
            schedules['dp'] = drop_scheduler(
                cfg.drop_path, cfg.epochs, num_training_steps_per_epoch,
                cfg.cutoff_epoch, cfg.drop_mode, cfg.drop_schedule,
            )
            print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))

        print("Start training")
        start_time = time.time()
        best_map_holder   = BestMetricHolder(use_ema=cfg.use_ema)
        best_map_5095     = 0
        best_map_50       = 0
        best_map_ema_5095 = 0
        best_map_ema_50   = 0

        for epoch in range(cfg.start_epoch, cfg.epochs):
            epoch_start_time = time.time()

            if cfg.distributed:
                sampler_train.set_epoch(epoch)

            model.train()
            criterion.train()

            train_stats = train_one_epoch(
                model, criterion, lr_scheduler, data_loader_train,
                optimizer, device, epoch,
                effective_batch_size, cfg.clip_max_norm,
                ema_m=self.ema_m,
                schedules=schedules,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=cfg.vit_encoder_num_layers,
                cfg=cfg,
                callbacks=callbacks,
                postprocess=postprocess,
            )

            train_epoch_time     = time.time() - epoch_start_time
            train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))

            if cfg.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if (epoch + 1) % cfg.lr_drop == 0 or (epoch + 1) % cfg.checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

                for checkpoint_path in checkpoint_paths:
                    weights = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "cfg": namespace_to_dict(cfg),
                        "args": cfg,
                    }
                    if cfg.use_ema:
                        weights['ema_model'] = self.ema_m.module.state_dict()

                    if not cfg.dont_save_weights:
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        utils.save_on_master(weights, checkpoint_path)

            with torch.no_grad():
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocess, data_loader_val, base_ds, device, cfg=cfg
                )

            if not cfg.segmentation_head:
                map_regular = test_stats["coco_eval_bbox"][0]
            else:
                map_regular = test_stats["coco_eval_masks"][0]

            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                best_map_5095 = max(best_map_5095, map_regular)
                map50 = (
                    test_stats["coco_eval_bbox"][1]
                    if not cfg.segmentation_head
                    else test_stats["coco_eval_masks"][1]
                )
                best_map_50 = max(best_map_50, map50)
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                if not cfg.dont_save_weights:
                    best_weights = {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "cfg": namespace_to_dict(cfg),
                        "args": cfg,
                    }
                    utils.save_on_master(best_weights, checkpoint_path)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v  for k, v in test_stats.items()},
                'epoch':        epoch,
                'n_parameters': n_parameters,
            }

            if cfg.use_ema:
                ema_test_stats, _ = evaluate(
                    self.ema_m.module, criterion, postprocess,
                    data_loader_val, base_ds, device, cfg=cfg,
                )
                log_stats.update({f'ema_test_{k}': v for k, v in ema_test_stats.items()})
                map_ema = (
                    ema_test_stats["coco_eval_bbox"][0]
                    if not cfg.segmentation_head
                    else ema_test_stats["coco_eval_masks"][0]
                )
                best_map_ema_5095 = max(best_map_ema_5095, map_ema)
                _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
                if _isbest:
                    map_ema_50 = (
                        ema_test_stats["coco_eval_bbox"][1]
                        if not cfg.segmentation_head
                        else ema_test_stats["coco_eval_masks"][1]
                    )
                    best_map_ema_50 = max(best_map_ema_50, map_ema_50)
                    checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                    if not cfg.dont_save_weights:
                        ema_weights = {
                            "model": self.ema_m.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "cfg": namespace_to_dict(cfg),
                            "args": cfg,
                        }
                        utils.save_on_master(ema_weights, checkpoint_path)

            log_stats.update(best_map_holder.summary())

            ep_paras = {'epoch': epoch, 'n_parameters': n_parameters}
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            log_stats['train_epoch_time'] = train_epoch_time_str
            epoch_time     = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if cfg.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            if not cfg.segmentation_head:
                                torch.save(
                                    coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name,
                                )
                            else:
                                torch.save(
                                    coco_evaluator.coco_eval["segm"].eval,
                                    output_dir / "eval" / name,
                                )

            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)

            if self.stop_early:
                print(f"Early stopping requested, stopping at epoch {epoch}")
                break

        best_is_ema = best_map_ema_5095 > best_map_5095

        if utils.is_main_process():
            if best_is_ema:
                shutil.copy2(
                    output_dir / 'checkpoint_best_ema.pth',
                    output_dir / 'checkpoint_best_total.pth',
                )
            else:
                shutil.copy2(
                    output_dir / 'checkpoint_best_regular.pth',
                    output_dir / 'checkpoint_best_total.pth',
                )

            utils.strip_checkpoint(output_dir / 'checkpoint_best_total.pth')

            best_map_5095 = max(best_map_5095, best_map_ema_5095)
            results = ema_test_stats["results_json"] if best_is_ema else test_stats["results_json"]
            class_map = results["class_map"]
            results["class_map"] = {"valid": class_map}
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

            total_time     = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            print('Results saved to {}'.format(output_dir / "results.json"))

        if best_is_ema:
            self.model = self.ema_m.module
        self.model.eval()

        if cfg.run_test:
            best_state_dict = torch.load(
                output_dir / 'checkpoint_best_total.pth',
                map_location='cpu', weights_only=False,
            )['model']
            model.load_state_dict(best_state_dict)
            model.eval()
            test_stats, _ = evaluate(
                model, criterion, postprocess, data_loader_test, base_ds_test, device, cfg=cfg
            )
            print(f"Test results: {test_stats}")
            with open(output_dir / "results.json", "r") as f:
                results = json.load(f)
            test_metrics = test_stats["results_json"]["class_map"]
            results["class_map"]["test"] = test_metrics
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

        for callback in callbacks["on_train_end"]:
            callback()

    def export(
        self,
        output_dir="output",
        infer_dir=None,
        simplify=False,
        backbone_only=False,
        opset_version=17,
        verbose=True,
        force=False,
        shape=None,
        batch_size=1,
        **kwargs,
    ):
        """Export the trained model to ONNX format"""
        print("Exporting model to ONNX format")
        try:
            from rfdetr.deploy.export import export_onnx, make_infer_image, onnx_simplify
        except ImportError:
            print(
                "It seems some dependencies for ONNX export are missing. "
                "Please run `pip install rfdetr[onnxexport]` and try again."
            )
            raise

        device = self.device
        model  = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)

        if shape is None:
            shape = (self.resolution, self.resolution)
        else:
            if shape[0] % 14 != 0 or shape[1] % 14 != 0:
                raise ValueError("Shape must be divisible by 14")

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names   = ['input']
        output_names  = ['features'] if backbone_only else ['dets', 'labels']
        dynamic_axes  = None

        self.model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                print(f"PyTorch inference output shape: {features.shape}")
            elif self.cfg.segmentation_head:
                outputs = model(input_tensors)
                print(
                    f"PyTorch inference output shapes — "
                    f"Boxes: {outputs['pred_boxes'].shape}, "
                    f"Labels: {outputs['pred_logits'].shape}, "
                    f"Masks: {outputs['pred_masks'].shape}"
                )
            else:
                outputs = model(input_tensors)
                print(
                    f"PyTorch inference output shapes — "
                    f"Boxes: {outputs['pred_boxes'].shape}, "
                    f"Labels: {outputs['pred_logits'].shape}"
                )

        model.cpu()
        input_tensors = input_tensors.cpu()

        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version,
        )
        print(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=force,
            )
            print(f"Successfully simplified ONNX model to: {sim_output_file}")

        print("ONNX export completed successfully")
