"""
Download RF-DETR COCO-trained checkpoints from Hugging Face into ``<project_root>/rfdetr_pretrained``.

Repo: https://huggingface.co/myn0908/rfdetrv2
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from rfdetrv2.utils.dinov3_pretrained import _download_file, _exclusive_download_lock

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
    explicit: str | None,
) -> str | None:
    """Return path to a COCO-pretrained RF-DETR ``.pth`` for ``Pipeline(pretrain_weights=...)``.

    Parameters
    ----------
    explicit:
        Optional path from env/CLI. If it points to an existing file, that path is used.
        If set but missing, falls back to downloading the official Hub file for *model_size*.
        If ``None`` or empty, downloads (or reuses) under ``rfdetr_pretrained/``.

    Returns
    -------
    str | None
        Absolute path to ``.pth``, or ``None`` if ``RFDETR_SKIP_COCO_CHECKPOINT=1``.
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


def resolve_pretrain_weights_path(
    pretrain_weights: str,
    project_root: Path | None = None,
    *,
    redownload: bool = False,
) -> str:
    """Resolve ``pretrain_weights`` to a local ``.pth`` path for :class:`~rfdetrv2.runner.Pipeline`.

    Resolution order
    ----------------
    1. If *pretrain_weights* is an existing file path → return it resolved.
    2. If it is an ``http(s)`` URL → download into ``rfdetr_pretrained/`` (by filename) and return.
    3. If it is a HuggingFace COCO checkpoint alias (``rfdetrv2_base``, ``base``, ``rfdetrv2_base.pth``, …)
       matching :data:`RFDETR_COCO_CHECKPOINT_BY_SIZE` → ensure file under ``rfdetr_pretrained/`` (download if missing).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    raw = (pretrain_weights or "").strip()
    if not raw:
        raise ValueError("pretrain_weights is empty")

    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve())

    if raw.startswith("http://") or raw.startswith("https://"):
        out_dir = rfdetr_pretrained_dir(project_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = raw.split("/")[-1].split("?")[0] or "weights.pth"
        dest = out_dir / name
        if redownload and dest.is_file():
            dest.unlink()
        if not dest.is_file():
            logger.info("Downloading pretrain_weights URL → %s", dest)
            print(f"Downloading {name} ...", file=sys.stderr)
            _download_file(raw, dest, timeout=600)
            print(f"Saved → {dest}", file=sys.stderr)
        return str(dest.resolve())

    key = raw.lower()
    if key.endswith(".pth"):
        key = key[: -len(".pth")]
    if key.startswith("rfdetrv2_"):
        size_key = key[len("rfdetrv2_") :]
    else:
        size_key = key

    if size_key not in RFDETR_COCO_CHECKPOINT_BY_SIZE:
        raise FileNotFoundError(
            f"pretrain_weights={pretrain_weights!r} is not an existing file, URL, or HF alias. "
            f"Use a path to a .pth file, a https:// URL, or one of: "
            f"{sorted(RFDETR_COCO_CHECKPOINT_BY_SIZE)} / rfdetrv2_<size> "
            f"(files: {dict(RFDETR_COCO_CHECKPOINT_BY_SIZE)}). "
            f"Hub: {HF_BASE_URL}"
        )

    fname = RFDETR_COCO_CHECKPOINT_BY_SIZE[size_key]
    target = rfdetr_pretrained_dir(project_root) / fname
    download_rfdetr_coco_checkpoint(project_root, size_key, overwrite=redownload)
    if not target.is_file():
        raise FileNotFoundError(f"Expected checkpoint at {target} after download.")
    return str(target.resolve())
