#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Train from scratch: ``Pipeline.train`` + YAML under rfdetrv2/configs/train_from_scratch/.
# ------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import _common  # noqa: E402
from rfdetrv2.runner.trainer import Pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RF-DETRv2 training (from scratch) via Pipeline + YAML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _common.add_config_args(parser, mode="train")
    _common.add_data_and_run_args(parser)
    _common.add_pretrain_arg(parser)
    args = parser.parse_args()
    cfg_path = _common.resolve_yaml("train", args.variant, args.config)
    overrides = _common.overrides_from_args(args)
    if args.pretrain_weights is not None:
        overrides["pretrain_weights"] = args.pretrain_weights

    pipe = Pipeline(config=str(cfg_path), **overrides)
    pipe.train(_common.empty_callbacks())


if __name__ == "__main__":
    main()
