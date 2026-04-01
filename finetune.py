#!/usr/bin/env python3
"""
Repo-root fine-tune launcher.

Always runs ``scripts/finetune.py`` (argparse + ``model.finetune``). Use this path
if your command is ``python finetune.py`` or ``torchrun ... finetune.py`` so you
never pick up a stale ``DATASET_DIR`` from the environment.

Hyperparameters and dataset paths: pass CLI flags or edit ``scripts/finetune.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SCRIPT = _ROOT / "scripts" / "finetune.py"


if __name__ == "__main__":
    if not _SCRIPT.is_file():
        sys.exit(f"Missing {_SCRIPT}; expected scripts/finetune.py next to this file.")
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    sys.argv[0] = str(_SCRIPT)
    import runpy

    runpy.run_path(str(_SCRIPT), run_name="__main__")
