"""
Download official DINOv3 LVD1689M checkpoints into ``<project_root>/dinov3_pretrained``.

Weights are hosted on HuggingFace: https://huggingface.co/myn0908/dinov3
Use is subject to the DINOv3 License Agreement.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable
import shutil
import tempfile
import zipfile
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

HF_BASE_URL = "https://huggingface.co/myn0908/dinov3/resolve/main"

# Official DINOv3 tree (hubconf.py + package) — used when ``<project>/dinov3`` is missing.
DEFAULT_DINOV3_SOURCE_ZIP_URL = (
    "https://github.com/facebookresearch/dinov3/archive/refs/heads/main.zip"
)

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


@contextmanager
def _exclusive_download_lock(lock_path: Path):
    """Serialize downloads of the same file (e.g. ``torchrun`` / multi-process)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # No cross-process flock on Windows; single-writer .part is usually enough.
        yield
        return
    import fcntl

    with open(lock_path, "a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _download_file(
    url: str, dest: Path, chunk_size: int = 1024 * 1024, *, timeout: int = 120
) -> None:
    """Stream-download *url* into *dest*, writing atomically via a .part temp file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "RF-DETR/dinov3_pretrained"})
    try:
        with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            last_pct = -1
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    if pct >= last_pct + 10:
                        logger.info("  %s: %d%% (%.1f/%.1f MB)", dest.name, pct, downloaded / 1e6, total / 1e6)
                        last_pct = pct
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
        lock_path = out / f".{filename}.download.lock"
        with _exclusive_download_lock(lock_path):
            # Another rank may have finished while we waited.
            if target.exists() and not overwrite:
                logger.debug("Skip existing DINOv3 weight (after lock): %s", target)
                continue
            url = _url_for_relative(rel)
            logger.info("Downloading DINOv3 weights: %s (url=%s)", target.name, url)
            _download_file(url, target)
            logger.info("DINOv3 weights saved: %s", target)

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


def ensure_dinov3_hub_source(dest_repo_dir: Path) -> Path:
    """Ensure *dest_repo_dir* contains the DINOv3 torch.hub tree (``hubconf.py`` + ``dinov3/`` package).

    If ``hubconf.py`` is missing, downloads the official GitHub archive (zip), extracts it
    into *dest_repo_dir*, then removes the temp zip. Uses a file lock so ``torchrun`` ranks
    do not race.

    Override zip URL with env ``DINOV3_SOURCE_ZIP_URL``. Set ``DINOV3_SKIP_SOURCE_DOWNLOAD=1``
    to disable auto-fetch (raises if the tree is still missing).
    """
    hubconf = dest_repo_dir / "hubconf.py"
    if hubconf.is_file():
        return dest_repo_dir

    skip = os.environ.get("DINOV3_SKIP_SOURCE_DOWNLOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if skip:
        raise FileNotFoundError(
            f"DINOv3 hub source missing at '{dest_repo_dir}' (no hubconf.py). "
            f"Unset DINOV3_SKIP_SOURCE_DOWNLOAD or copy the `dinov3/` folder from this project."
        )

    lock_path = dest_repo_dir.parent / ".dinov3_hub_source.download.lock"
    with _exclusive_download_lock(lock_path):
        if (dest_repo_dir / "hubconf.py").is_file():
            return dest_repo_dir

        url = os.environ.get("DINOV3_SOURCE_ZIP_URL", "").strip() or DEFAULT_DINOV3_SOURCE_ZIP_URL
        logger.info("Downloading DINOv3 hub source tree: %s (url=%s)", dest_repo_dir, url)

        parent = dest_repo_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        zip_path = parent / ".dinov3_main_source.zip"
        _download_file(url, zip_path, timeout=600)

        try:
            with tempfile.TemporaryDirectory(dir=str(parent)) as tmp:
                tmp_root = Path(tmp)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmp_root)
                # GitHub archive has a single top-level dir, e.g. ``dinov3-main/``.
                subdirs = [p for p in tmp_root.iterdir() if p.is_dir()]
                if len(subdirs) != 1:
                    raise RuntimeError(
                        f"Unexpected zip layout under {tmp_root}: {subdirs!r}"
                    )
                extracted = subdirs[0]
                if dest_repo_dir.exists():
                    shutil.rmtree(dest_repo_dir)
                shutil.move(str(extracted), str(dest_repo_dir))
        finally:
            try:
                zip_path.unlink()
            except OSError:
                pass

        if not (dest_repo_dir / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"After extract, hubconf.py still missing at '{dest_repo_dir}'."
            )
        logger.info("DINOv3 hub source saved: %s", dest_repo_dir)

    return dest_repo_dir


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