# ------------------------------------------------------------------------
# Shared helpers for scripts/train.py, finetune.py, evaluate.py, inference.py
# ------------------------------------------------------------------------
from __future__ import annotations

import argparse
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_CONFIGS = REPO_ROOT / "rfdetrv2" / "configs"

VARIANTS = frozenset({"nano", "small", "base", "large"})

_MODE_SUBDIR = {
    "train": "train_from_scratch",
    "finetune": "finetune",
    "evaluate": "train_from_scratch",
    "inference": "inference",
}


def resolve_yaml(mode: str, variant: str | None, config: str | None) -> Path:
    if config:
        p = Path(config)
        if not p.is_absolute():
            p = REPO_ROOT / p
        p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Config not found: {p}")
        return p
    v = variant or "base"
    if v not in VARIANTS:
        raise ValueError(f"--variant must be one of {sorted(VARIANTS)}, got {v!r}")
    subdir = _MODE_SUBDIR[mode]
    path = PACKAGED_CONFIGS / subdir / f"{v}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Packaged config missing: {path}")
    return path


def empty_callbacks() -> defaultdict[str, list]:
    return defaultdict(list)


_TRAIN_OVERRIDE_KEYS = (
    "dataset_dir",
    "coco_path",
    "output_dir",
    "pretrain_weights",
    "epochs",
    "batch_size",
    "num_workers",
    "resume",
    "lr",
    "lr_encoder",
    "device",
    "eval",
    "run_test",
    "dataset_file",
    "amp",
    "use_ema",
)


def overrides_from_args(args: Namespace, keys: tuple[str, ...] = _TRAIN_OVERRIDE_KEYS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        if k not in vars(args):
            continue
        v = getattr(args, k)
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        out[k] = v
    return out


def add_config_args(parser: argparse.ArgumentParser, *, mode: str) -> None:
    g = parser.add_argument_group("config")
    g.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to run YAML (merged over packaged train_default). If omitted, uses --variant.",
    )
    g.add_argument(
        "--variant",
        type=str,
        choices=sorted(VARIANTS),
        default=None,
        help=f"Packaged YAML under rfdetrv2/configs/{_MODE_SUBDIR[mode]}/{{nano,small,base,large}}.yaml (default: base).",
    )


def add_data_and_run_args(parser: argparse.ArgumentParser) -> None:
    p = parser.add_argument_group("data / run")
    p.add_argument("--dataset-dir", "--dataset_dir", dest="dataset_dir", type=str, default=None)
    p.add_argument("--coco-path", "--coco_path", dest="coco_path", type=str, default=None)
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=None)
    p.add_argument(
        "--dataset-file",
        "--dataset_file",
        dest="dataset_file",
        type=str,
        choices=("coco", "roboflow", "o365"),
        default=None,
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint .pth to resume training state.")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-encoder", "--lr_encoder", dest="lr_encoder", type=float, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu | mps")
    sup = argparse.SUPPRESS
    p.add_argument("--amp", dest="amp", action="store_true", default=sup)
    p.add_argument("--no-amp", dest="amp", action="store_false", default=sup)
    p.add_argument("--run-test", "--run_test", dest="run_test", action="store_true", default=sup)
    p.add_argument("--no-run-test", "--no_run_test", dest="run_test", action="store_false", default=sup)
    p.add_argument("--use-ema", "--use_ema", dest="use_ema", action="store_true", default=sup)
    p.add_argument("--no-use-ema", "--no_use_ema", dest="use_ema", action="store_false", default=sup)


def add_pretrain_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pretrain-weights",
        "--pretrain_weights",
        dest="pretrain_weights",
        type=str,
        default=None,
        help="Detector checkpoint path or HF alias (e.g. rfdetrv2_base). Merged into cfg before build.",
    )
