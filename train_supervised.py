"""
Backward-compatible launcher: forwards to ``scripts/train_supervised.py``.

Run from repository root: ``python train_supervised.py [args]``
(preferred canonical path: ``python scripts/train_supervised.py``).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SCRIPT = _ROOT / "scripts" / "train_supervised.py"


def main() -> None:
    raise SystemExit(subprocess.call([sys.executable, str(_SCRIPT), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
