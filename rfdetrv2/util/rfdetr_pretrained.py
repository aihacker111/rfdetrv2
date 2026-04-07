"""
RF-DETR COCO checkpoints → ``<project_root>/rfdetr_pretrained`` (HuggingFace).

Use :func:`resolve_rfdetr_coco_checkpoint` for CLI-style resolution (path or auto-download).
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Internal download utilities (self-contained, no dinov3_pretrained dependency)
# ---------------------------------------------------------------------------

def _download_file(
    url: str, dest: Path, chunk_size: int = 1024 * 1024, *, timeout: int = 120
) -> None:
    """Stream-download *url* into *dest*, writing atomically via a .part file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "RF-DETR/rfdetr_pretrained"})
    try:
        with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            total      = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(
                        f"\r  {dest.name}: {pct}% ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)",
                        end="", file=sys.stderr, flush=True,
                    )
            if total:
                print(file=sys.stderr)
        tmp.replace(dest)
    except (OSError, URLError):
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


@contextmanager
def _exclusive_download_lock(lock_path: Path):
    """Serialize concurrent downloads of the same file (multi-process safe)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        yield
        return
    import fcntl
    with open(lock_path, "a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

logger = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co/myn0908/rfdetrv2/resolve/main"

# Official COCO checkpoints (filename on Hub)
RFDETR_COCO_CHECKPOINT_BY_SIZE: dict[str, str] = {
    "nano": "rfdetrv2_nano.pth",
    "small": "rfdetrv2_small.pth",
    "base": "rfdetrv2_base.pth",
    "large": "rfdetrv2_large.pth",
}


def rfdetr_pretrained_dir(project_root: Path | None = None) -> Path:
    """``<project_root>/rfdetr_pretrained``."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return project_root / "rfdetr_pretrained"


def _hf_url(filename: str) -> str:
    return f"{HF_BASE_URL.rstrip('/')}/{filename}?download=true"


def download_rfdetr_coco_checkpoint(
    project_root: Path,
    model_size: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Download the COCO checkpoint for *model_size* if missing."""
    if model_size not in RFDETR_COCO_CHECKPOINT_BY_SIZE:
        raise ValueError(
            f"Unknown model_size={model_size!r}; expected one of "
            f"{sorted(RFDETR_COCO_CHECKPOINT_BY_SIZE)}"
        )
    fname = RFDETR_COCO_CHECKPOINT_BY_SIZE[model_size]
    out_dir = rfdetr_pretrained_dir(project_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / fname
    if target.exists() and not overwrite:
        logger.debug("Skip existing RF-DETR checkpoint: %s", target)
        return target

    lock_path = out_dir / f".{fname}.download.lock"
    with _exclusive_download_lock(lock_path):
        if target.exists() and not overwrite:
            return target
        url = _hf_url(fname)
        logger.info("Downloading RF-DETR COCO weights → %s  url=%s", target.name, url)
        print(f"Downloading {fname} from HuggingFace ...", file=sys.stderr)
        _download_file(url, target, timeout=600)
        print(f"Saved → {target}", file=sys.stderr)
    return target


def resolve_rfdetr_coco_checkpoint(
    project_root: Path,
    model_size: str,
    *,
    explicit: str | None = None,
) -> str | None:
    """Path to COCO RF-DETR ``.pth`` for ``pretrain_weights=`` (local, or auto-download).

    *explicit*: CLI/env path if the file exists; otherwise Hub file for *model_size*.
    Returns ``None`` when ``RFDETR_SKIP_COCO_CHECKPOINT=1``.
    """
    if os.environ.get("RFDETR_SKIP_COCO_CHECKPOINT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return None

    candidates: list[Path] = []
    if explicit and str(explicit).strip():
        raw = Path(explicit.strip()).expanduser()
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            candidates.append((project_root / raw).resolve())

    for p in candidates:
        if p.is_file():
            logger.info("Using RF-DETR COCO checkpoint: %s", p)
            return str(p)

    if candidates:
        logger.warning(
            "COCO_WEIGHTS path not found (%s); downloading official checkpoint for size=%r.",
            explicit,
            model_size,
        )

    dest = download_rfdetr_coco_checkpoint(project_root, model_size)
    return str(dest.resolve())
