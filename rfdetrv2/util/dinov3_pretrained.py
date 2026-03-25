"""
Download official DINOv3 LVD1689M checkpoints into ``<project_root>/dinov3_pretrained``.

Weights are hosted on HuggingFace: https://huggingface.co/myn0908/dinov3
Use is subject to the DINOv3 License Agreement.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co/myn0908/dinov3/resolve/main"

# (filename, path_under_HF_repo)
DINOV3_PRETRAINED_MANIFEST: tuple[tuple[str, str], ...] = (
    (
        "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    ),
    (
        "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    ),
    (
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    ),
)


def dinov3_pretrained_dir(project_root: Path | None = None) -> Path:
    """Return ``<project_root>/dinov3_pretrained`` (default project = parent of ``rfdetrv2``)."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return project_root / "dinov3_pretrained"


def _url_for_relative(rel: str) -> str:
    """Build full HuggingFace download URL, appending ?download=true."""
    rel = rel.lstrip("/")
    return f"{HF_BASE_URL.rstrip('/')}/{rel}?download=true"


def _download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream-download *url* into *dest*, writing atomically via a .part temp file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "RF-DETR/dinov3_pretrained"})
    try:
        with urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
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
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
            if total:
                print(file=sys.stderr)  # newline after progress
        tmp.replace(dest)
    except (OSError, URLError):
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def download_dinov3_pretrained_weights(
    dest_dir: Path | None = None,
    *,
    project_root: Path | None = None,
    overwrite: bool = False,
    manifest: Iterable[tuple[str, str]] | None = None,
) -> Path:
    """Download DINOv3 LVD1689M checkpoints from HuggingFace (skips files that already exist).

    Parameters
    ----------
    dest_dir:
        Target directory. Default: ``dinov3_pretrained_dir(project_root)``.
    overwrite:
        If True, re-download even when the file already exists.
    manifest:
        Optional list of ``(filename, hf_relative_path)`` pairs.
        Defaults to :data:`DINOV3_PRETRAINED_MANIFEST`.

    Returns
    -------
    Path
        The directory containing the downloaded ``.pth`` files.
    """
    out = dest_dir or dinov3_pretrained_dir(project_root)
    out.mkdir(parents=True, exist_ok=True)
    entries = list(manifest) if manifest is not None else list(DINOV3_PRETRAINED_MANIFEST)

    for filename, rel in entries:
        target = out / filename
        if target.exists() and not overwrite:
            logger.debug("Skip existing DINOv3 weight: %s", target)
            continue
        url = _url_for_relative(rel)
        logger.info("Downloading DINOv3 weights → %s  url=%s", target.name, url)
        print(f"Downloading {filename} from HuggingFace ...", file=sys.stderr)
        _download_file(url, target)
        print(f"Saved → {target}", file=sys.stderr)

    return out


def ensure_dinov3_pretrained_weights(
    project_root: Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Ensure the default ``dinov3_pretrained`` folder exists with all three LVD1689M checkpoints."""
    return download_dinov3_pretrained_weights(
        project_root=project_root, overwrite=overwrite
    )


def resolve_pretrained_encoder_path(
    project_root: Path,
    model_size: str,
    *,
    explicit: str | None,
    weights_by_size: dict[str, str],
) -> str:
    """Return path to a DINOv3 ``.pth`` weight file.

    Resolution order
    ----------------
    1. *explicit* path — used as-is if provided.
    2. ``dinov3_pretrained/<filename>`` under project root — if the file exists.
    3. ``<project_root>/<filename>`` — legacy flat placement.
    4. Auto-download from HuggingFace into ``dinov3_pretrained/``.

    Parameters
    ----------
    project_root:
        Root of the RF-DETR project tree.
    model_size:
        One of ``"nano"``, ``"small"``, ``"base"``, ``"large"``.
    explicit:
        User-supplied path (``pretrained_encoder`` argument).
    weights_by_size:
        Mapping from *model_size* to the expected ``.pth`` filename.
    """
    if explicit:
        return explicit

    fname = weights_by_size[model_size]
    p_hub  = dinov3_pretrained_dir(project_root) / fname
    p_root = project_root / fname

    if p_hub.exists():
        return str(p_hub)
    if p_root.exists():
        return str(p_root)

    # Auto-download only the required file
    logger.info(
        "DINOv3 weights for size=%r not found locally — downloading from HuggingFace.",
        model_size,
    )
    download_dinov3_pretrained_weights(
        project_root=project_root,
        manifest=[(fname, fname)],  # download only what we need
    )

    if p_hub.exists():
        return str(p_hub)
    if p_root.exists():
        return str(p_root)

    raise FileNotFoundError(
        f"Could not obtain DINOv3 weights for model_size={model_size!r} "
        f"(expected {p_hub} or {p_root})."
    )