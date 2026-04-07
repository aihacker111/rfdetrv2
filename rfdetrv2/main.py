# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Model — high-level Python API for RF-DETR (train / export / inference).

CLI entry point is at the bottom under ``if __name__ == '__main__':``.

For training pipelines prefer the structured ``TrainingPipeline`` in
``rfdetrv2.pipeline.train``.  ``Model`` is kept for backward compatibility.
"""

from __future__ import annotations

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
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Callable, DefaultDict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import rfdetrv2.util.misc as utils
from rfdetrv2.args import get_args_parser, populate_args
from rfdetrv2.datasets import build_dataset, get_coco_api_from_dataset
from rfdetrv2.core import evaluate, train_one_epoch
from rfdetrv2.models import PostProcess, build_criterion_and_postprocessors, build_model
from rfdetrv2.util.benchmark import benchmark
from rfdetrv2.util.drop_scheduler import drop_scheduler
from rfdetrv2.util.get_param_dicts import get_param_dict
from rfdetrv2.util.utils import BestMetricHolder, ModelEma, clean_state_dict

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy("file_system")

logger = getLogger(__name__)


# ---------------------------------------------------------------------------
# Model — Python API
# ---------------------------------------------------------------------------

class Model:
    """High-level RF-DETR model wrapper (train / export).

    Prefer ``rfdetrv2.pipeline.TrainingPipeline`` for new code — it is
    better structured, fully typed, and supports all training options.
    This class is kept for backward-compatibility with existing scripts.
    """

    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self.args = args
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)

        if args.pretrain_weights is not None:
            self._load_pretrain_weights(args)

        if args.backbone_lora:
            self._apply_backbone_lora()

        self.model = self.model.to(self.device)
        self.postprocess = PostProcess(num_select=args.num_select)
        self.stop_early = False

    # ------------------------------------------------------------------
    # Pretrained weights
    # ------------------------------------------------------------------

    def _load_pretrain_weights(self, args) -> None:
        print("Loading pretrain weights …")
        try:
            checkpoint = torch.load(args.pretrain_weights, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load pretrain weights from '{args.pretrain_weights}': {e}"
            ) from e

        if "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
            self.args.class_names = checkpoint["args"].class_names
            self.class_names = checkpoint["args"].class_names

        ckpt_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
        if ckpt_num_classes != args.num_classes + 1:
            self.reinitialize_detection_head(ckpt_num_classes)

        if args.pretrain_exclude_keys:
            for key in args.pretrain_exclude_keys:
                checkpoint["model"].pop(key, None)

        if args.pretrain_keys_modify_to_load:
            from rfdetrv2.util.obj365_to_coco_model import get_coco_pretrain_from_obj365
            for key in args.pretrain_keys_modify_to_load:
                try:
                    checkpoint["model"][key] = get_coco_pretrain_from_obj365(
                        self.model.state_dict()[key],
                        checkpoint["model"][key],
                    )
                except Exception:
                    print(f"Failed to convert key '{key}', removing from checkpoint.")
                    checkpoint["model"].pop(key, None)

        num_desired = args.num_queries * args.group_detr
        for name, state in checkpoint["model"].items():
            if name.endswith(("refpoint_embed.weight", "query_feat.weight")):
                checkpoint["model"][name] = state[:num_desired]

        self.model.load_state_dict(checkpoint["model"], strict=False)

    def _apply_backbone_lora(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "backbone_lora=True requires `peft`. "
                "Install it with: pip install peft"
            ) from exc
        print("Applying LoRA to backbone …")
        lora_config = LoraConfig(
            r=16, lora_alpha=16, use_dora=True,
            target_modules=["q_proj", "v_proj", "k_proj", "qkv",
                            "query", "key", "value", "cls_token", "register_tokens"],
        )
        self.model.backbone[0].encoder = get_peft_model(self.model.backbone[0].encoder, lora_config)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reinitialize_detection_head(self, num_classes: int) -> None:
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self) -> None:
        self.stop_early = True
        print("Early stopping requested — will complete current epoch and stop.")

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------

    def train(self, callbacks: DefaultDict[str, List[Callable]] = None, **kwargs):
        if callbacks is None:
            callbacks = defaultdict(list)

        _SUPPORTED = {"on_fit_epoch_end", "on_train_batch_start", "on_train_end"}
        for key in callbacks:
            if key not in _SUPPORTED:
                raise ValueError(f"Unsupported callback '{key}'. Supported: {_SUPPORTED}")

        args = populate_args(**kwargs)
        if getattr(args, "class_names", None) is not None:
            self.args.class_names = args.class_names
            self.args.num_classes = args.num_classes

        utils.init_distributed_mode(args)
        print(f"git:\n  {utils.get_sha()}\n")
        print(args)
        device = torch.device(args.device)

        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        criterion, postprocess = build_criterion_and_postprocessors(args)

        if getattr(args, "freeze_encoder", False):
            backbone = self.model.backbone
            if hasattr(backbone, "__getitem__") and hasattr(backbone[0], "encoder"):
                for p in backbone[0].encoder.parameters():
                    p.requires_grad = False
                if utils.is_main_process():
                    logger.info("DINOv3 encoder frozen.")

        model = self.model.to(device)
        model_without_ddp = model

        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=False
            )
            model_without_ddp = model.module

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {n_params}")

        world_size = utils.get_world_size()
        if world_size > 1:
            mode = getattr(args, "lr_scale_mode", "sqrt")
            scale = math.sqrt(world_size) if mode == "sqrt" else float(world_size)
            args.lr *= scale
            args.lr_encoder *= scale
            if utils.is_main_process():
                print(f"Multi-GPU LR scaled ×{scale:.3f} → lr={args.lr:.2e}, lr_encoder={args.lr_encoder:.2e}")

        param_dicts = [p for p in get_param_dict(args, model_without_ddp) if p["params"].requires_grad]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

        dataset_train = build_dataset("train", args=args, resolution=args.resolution)
        dataset_val   = build_dataset("val",   args=args, resolution=args.resolution)
        dataset_test  = build_dataset(
            "test" if args.dataset_file == "roboflow" else "val",
            args=args, resolution=args.resolution,
        )

        debug_limit = int(getattr(args, "debug_data_limit", 0) or 0)
        if debug_limit > 0:
            dataset_train = torch.utils.data.Subset(dataset_train, range(min(debug_limit, len(dataset_train))))
            dataset_val   = torch.utils.data.Subset(dataset_val,   range(min(debug_limit, len(dataset_val))))
            dataset_test  = torch.utils.data.Subset(dataset_test,  range(min(debug_limit, len(dataset_test))))

        total_bs = args.batch_size * utils.get_world_size() * args.grad_accum_steps
        steps_per_epoch_lr = max(1, (len(dataset_train) + total_bs - 1) // total_bs)
        total_steps_lr  = steps_per_epoch_lr * args.epochs
        warmup_steps_lr = int(steps_per_epoch_lr * args.warmup_epochs)
        lr_gamma        = getattr(args, "lr_gamma", 0.1)
        lr_milestones   = getattr(args, "lr_milestones", [80, 160])
        milestone_steps = [m * steps_per_epoch_lr for m in sorted(lr_milestones)]

        def lr_lambda(step: int) -> float:
            if step < warmup_steps_lr:
                return step / max(1, warmup_steps_lr)
            progress = (step - warmup_steps_lr) / max(1, total_steps_lr - warmup_steps_lr)
            sched    = args.lr_scheduler
            min_f    = getattr(args, "lr_min_factor", 0.05)
            if sched == "cosine":
                return min_f + (1 - min_f) * 0.5 * (1 + math.cos(math.pi * progress))
            if sched == "wsd":
                stable = getattr(args, "lr_stable_ratio", 0.7)
                if progress < stable:
                    return 1.0
                dp = (progress - stable) / max(1e-8, 1.0 - stable)
                return min_f + (1 - min_f) * 0.5 * (1 + math.cos(math.pi * dp))
            if sched == "cosine_restart":
                period  = int(getattr(args, "lr_restart_period", 50) * steps_per_epoch_lr)
                decay   = getattr(args, "lr_restart_decay", 0.8)
                elapsed = step - warmup_steps_lr
                cycle   = elapsed // max(1, period)
                prog    = (elapsed % max(1, period)) / max(1, period)
                return decay ** cycle * (min_f + (1 - min_f) * 0.5 * (1 + math.cos(math.pi * prog)))
            if sched == "linear":
                return 1.0 - progress * (1.0 - min_f)
            if sched == "step":
                return 1.0 if step < args.lr_drop * steps_per_epoch_lr else lr_gamma
            if sched == "multistep":
                return lr_gamma ** sum(1 for ms in milestone_steps if step >= ms)
            return 1.0

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val   = DistributedSampler(dataset_val,  shuffle=False)
            sampler_test  = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val   = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test  = torch.utils.data.SequentialSampler(dataset_test)

        effective_bs = args.batch_size * args.grad_accum_steps
        min_batches  = kwargs.get("min_batches", 5 if not debug_limit else 1)
        num_workers  = self._resolve_num_workers(args.num_workers)

        if len(dataset_train) < effective_bs * min_batches:
            logger.info("Dataset too small — using replacement sampler.")
            loader_train = DataLoader(
                dataset_train, batch_size=effective_bs,
                sampler=torch.utils.data.RandomSampler(
                    dataset_train, replacement=True, num_samples=effective_bs * min_batches
                ),
                collate_fn=utils.collate_fn, num_workers=num_workers,
                pin_memory=True, persistent_workers=num_workers > 0,
            )
        else:
            loader_train = DataLoader(
                dataset_train,
                batch_sampler=torch.utils.data.BatchSampler(sampler_train, effective_bs, drop_last=True),
                collate_fn=utils.collate_fn, num_workers=num_workers,
                pin_memory=True, persistent_workers=num_workers > 0,
            )

        loader_val = DataLoader(
            dataset_val, args.batch_size, sampler=sampler_val,
            drop_last=False, collate_fn=utils.collate_fn, num_workers=num_workers,
            pin_memory=True, persistent_workers=num_workers > 0,
        )
        loader_test = DataLoader(
            dataset_test, args.batch_size, sampler=sampler_test,
            drop_last=False, collate_fn=utils.collate_fn, num_workers=num_workers,
            pin_memory=True, persistent_workers=num_workers > 0,
        )

        base_ds      = get_coco_api_from_dataset(dataset_val)
        base_ds_test = get_coco_api_from_dataset(dataset_test)

        if args.use_ema:
            self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)
        else:
            self.ema_m = None

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if utils.is_main_process() and args.do_benchmark:
            bm = benchmark(copy.deepcopy(model_without_ddp).float(), dataset_val, output_dir)
            print(json.dumps(bm, indent=2))

        if args.resume:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            model_without_ddp.load_state_dict(ckpt["model"], strict=True)
            if args.use_ema:
                if "ema_model" in ckpt:
                    self.ema_m.module.load_state_dict(clean_state_dict(ckpt["ema_model"]))
                else:
                    self.ema_m = ModelEma(model, decay=args.ema_decay, tau=args.ema_tau)
            if not args.eval and all(k in ckpt for k in ("optimizer", "lr_scheduler", "epoch")):
                optimizer.load_state_dict(ckpt["optimizer"])
                lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
                args.start_epoch = ckpt["epoch"] + 1

        if args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocess, loader_val, base_ds, device, args=args
            )
            if args.output_dir:
                key = "segm" if args.segmentation_head else "bbox"
                utils.save_on_master(coco_evaluator.coco_eval[key].eval, output_dir / "eval.pth")
            return

        total_batch      = effective_bs * utils.get_world_size()
        steps_per_epoch  = max(1, (len(dataset_train) + total_batch - 1) // total_batch)

        schedules = {}
        if args.dropout > 0:
            schedules["do"] = drop_scheduler(
                args.dropout, args.epochs, steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule,
            )
        if args.drop_path > 0:
            schedules["dp"] = drop_scheduler(
                args.drop_path, args.epochs, steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule,
            )

        print("Start training")
        start_time    = time.time()
        best_holder   = BestMetricHolder(use_ema=args.use_ema)
        best_5095     = best_ema_5095 = 0.0
        test_stats    = ema_test_stats = {}

        for epoch in range(args.start_epoch, args.epochs):
            epoch_start = time.time()

            if args.distributed:
                sampler_train.set_epoch(epoch)

            model.train()
            criterion.train()

            train_stats = train_one_epoch(
                model=model, criterion=criterion, lr_scheduler=lr_scheduler,
                data_loader=loader_train, optimizer=optimizer, device=device, epoch=epoch,
                batch_size=effective_bs, max_norm=args.clip_max_norm, ema_m=self.ema_m,
                schedules=schedules, num_training_steps_per_epoch=steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers, args=args,
                callbacks=callbacks,
            )

            epoch_time_str = str(datetime.timedelta(seconds=int(time.time() - epoch_start)))

            if args.output_dir:
                paths = [output_dir / "checkpoint.pth"]
                if (epoch + 1) % args.checkpoint_interval == 0:
                    paths.append(output_dir / f"checkpoint{epoch:04d}.pth")
                weights = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch, "args": args,
                }
                if args.use_ema:
                    weights["ema_model"] = self.ema_m.module.state_dict()
                if not args.dont_save_weights:
                    for p in paths:
                        utils.save_on_master(weights, p)

            with torch.no_grad():
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocess, loader_val, base_ds, device, args=args
                )

            map_key  = "coco_eval_masks" if args.segmentation_head else "coco_eval_bbox"
            map_reg  = test_stats[map_key][0]
            if best_holder.update(map_reg, epoch, is_ema=False):
                best_5095 = max(best_5095, map_reg)
                if not args.dont_save_weights:
                    utils.save_on_master(weights, output_dir / "checkpoint_best_regular.pth")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}":  v for k, v in test_stats.items()},
                "epoch": epoch, "n_parameters": n_params,
                "train_epoch_time": epoch_time_str,
                "epoch_time": str(datetime.timedelta(seconds=int(time.time() - epoch_start))),
            }

            if args.use_ema:
                ema_test_stats, _ = evaluate(
                    self.ema_m.module, criterion, postprocess, loader_val, base_ds, device, args=args
                )
                log_stats.update({f"ema_test_{k}": v for k, v in ema_test_stats.items()})
                map_ema = ema_test_stats[map_key][0]
                best_ema_5095 = max(best_ema_5095, map_ema)
                if best_holder.update(map_ema, epoch, is_ema=True) and not args.dont_save_weights:
                    ema_weights = {**weights, "model": self.ema_m.module.state_dict()}
                    utils.save_on_master(ema_weights, output_dir / "checkpoint_best_ema.pth")

            log_stats.update(best_holder.summary())

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                    (output_dir / "eval").mkdir(exist_ok=True)
                    key = "segm" if args.segmentation_head else "bbox"
                    names = ["latest.pth"] + ([f"{epoch:03d}.pth"] if epoch % 50 == 0 else [])
                    for name in names:
                        torch.save(coco_evaluator.coco_eval[key].eval, output_dir / "eval" / name)

            for cb in callbacks["on_fit_epoch_end"]:
                cb(log_stats)

            if self.stop_early:
                print(f"Early stop at epoch {epoch}.")
                break

        best_is_ema = best_ema_5095 > best_5095
        if utils.is_main_process():
            src = "checkpoint_best_ema.pth" if best_is_ema else "checkpoint_best_regular.pth"
            shutil.copy2(output_dir / src, output_dir / "checkpoint_best_total.pth")
            utils.strip_checkpoint(output_dir / "checkpoint_best_total.pth")

            results = (ema_test_stats if best_is_ema else test_stats).get("results_json", {})
            results["class_map"] = {"valid": results.get("class_map", [])}
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

            total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            print(f"Training time: {total_time}")
            print(f"Results: {output_dir / 'results.json'}")

        if best_is_ema:
            self.model = self.ema_m.module
        self.model.eval()

        if args.run_test:
            best_state = torch.load(
                output_dir / "checkpoint_best_total.pth", map_location="cpu", weights_only=False
            )["model"]
            model.load_state_dict(best_state)
            model.eval()
            test_stats, _ = evaluate(
                model, criterion, postprocess, loader_test, base_ds_test, device, args=args
            )
            with open(output_dir / "results.json", "r") as f:
                results = json.load(f)
            results["class_map"]["test"] = test_stats["results_json"]["class_map"]
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

        for cb in callbacks["on_train_end"]:
            cb()

    # ------------------------------------------------------------------
    # Export (ONNX)
    # ------------------------------------------------------------------

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
        """Export the model to ONNX format."""
        try:
            from rfdetrv2.deploy.export import export_onnx, make_infer_image, onnx_simplify
        except ImportError:
            raise ImportError(
                "ONNX export dependencies missing. "
                "Run: pip install rfdetr[onnxexport]"
            )

        device = self.device
        model  = copy.deepcopy(self.model.to("cpu")).to(device)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)

        if shape is None:
            shape = (self.resolution, self.resolution)
        elif shape[0] % 14 != 0 or shape[1] % 14 != 0:
            raise ValueError("Export shape must be divisible by 14.")

        from rfdetrv2.deploy.export import make_infer_image
        input_tensors = make_infer_image(infer_dir, shape, batch_size, device)
        input_names   = ["input"]
        output_names  = ["features"] if backbone_only else ["dets", "labels"]

        self.model.eval()
        with torch.no_grad():
            out = model(input_tensors)
            if isinstance(out, dict):
                print(f"PyTorch output — boxes: {out['pred_boxes'].shape}, logits: {out['pred_logits'].shape}")
            else:
                print(f"PyTorch output shape: {out.shape}")

        model.cpu()
        input_tensors = input_tensors.cpu()

        output_file = export_onnx(
            output_dir=output_dir, model=model,
            input_names=input_names, input_tensors=input_tensors,
            output_names=output_names, dynamic_axes=None,
            backbone_only=backbone_only, verbose=verbose, opset_version=opset_version,
        )
        print(f"ONNX saved: {output_file}")

        if simplify:
            sim = onnx_simplify(onnx_dir=output_file, input_names=input_names,
                                input_tensors=input_tensors, force=force)
            print(f"Simplified ONNX: {sim}")

        self.model = self.model.to(device)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_num_workers(requested: int) -> int:
        if requested > 0 and multiprocessing.get_start_method(allow_none=True) == "spawn":
            import __main__
            if not getattr(__main__, "__file__", None) or __main__.__name__ != "__main__":
                warnings.warn(
                    "num_workers set to 0: script is not wrapped in "
                    "`if __name__ == '__main__':` (required for 'spawn' start method).",
                    RuntimeWarning,
                )
                return 0
        return requested


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse as _ap

    parser = _ap.ArgumentParser(
        "LWDETR training and evaluation script",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cfg = vars(args)

    if args.subcommand == "export_model":
        _SKIP = {
            "num_classes", "grad_accum_steps", "lr", "lr_encoder", "weight_decay",
            "epochs", "lr_drop", "clip_max_norm", "lr_vit_layer_decay",
            "lr_component_decay", "dropout", "drop_path", "drop_mode", "drop_schedule",
            "cutoff_epoch", "pretrained_encoder", "pretrain_weights",
            "pretrain_exclude_keys", "pretrain_keys_modify_to_load",
            "set_cost_class", "set_cost_bbox", "set_cost_giou",
            "cls_loss_coef", "bbox_loss_coef", "giou_loss_coef", "focal_alpha",
            "aux_loss", "sum_group_losses", "use_varifocal_loss",
            "use_position_supervised_loss", "ia_bce_loss",
            "dataset_file", "coco_path", "dataset_dir", "square_resize_div_64",
            "output_dir", "checkpoint_interval", "seed", "resume", "start_epoch",
            "eval", "use_ema", "ema_decay", "ema_tau", "num_workers", "device",
            "world_size", "dist_url", "sync_bn", "fp16_eval",
            "infer_dir", "verbose", "opset_version",
        }
        for key in _SKIP:
            cfg.pop(key, None)
        from rfdetrv2.deploy.export import main as export_main
        cfg["batch_size"] = 1
        export_main(**cfg)
    else:
        from rfdetrv2.main import Model
        Model(**cfg).train()
