"""Launch ``scripts/visualize_prototype_rope2d.py`` from repo root (same CLI)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "scripts" / "visualize_prototype_rope2d.py"


def main() -> None:
    raise SystemExit(subprocess.call([sys.executable, str(_SCRIPT), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
