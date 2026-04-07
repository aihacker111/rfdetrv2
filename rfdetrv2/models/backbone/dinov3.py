"""
DINOv3 backbone adapter for RF-DETR.

Loading strategy
----------------
Source code  — downloaded automatically by ``torch.hub`` from GitHub and
               cached in ``~/.cache/torch/hub/facebookresearch_dinov3_main/``.

Weights      — downloaded automatically from HuggingFace on first use and
               cached in ``~/.cache/torch/hub/checkpoints/``.

No local ``dinov3/`` directory, no ``DINOV3_REPO_DIR`` env var, and no manual
weight download is required.  Simply instantiate ``DinoV3`` and everything
is fetched on demand.

Override weights
~~~~~~~~~~~~~~~~
Pass ``pretrained_encoder="/path/to/your.pth"`` to use a local checkpoint
instead of the auto-downloaded HuggingFace weights.

Windowed-attention implementation
----------------------------------
DINOv3 is loaded as a black-box from ``torch.hub``, so we cannot modify the
internals of each transformer layer.  Windowed attention is realised at the
*input level*:

    1. The input image (B, C, H, W) is tiled into W_n × W_n non-overlapping
       windows → (B·W_n², C, H/W_n, W/W_n).
    2. The full DINOv3 encoder runs on each window independently.
    3. The resulting per-window feature maps are reassembled into the original
       spatial layout → (B, C, H', W').

Usage
-----
    backbone = DinoV3(size="base")                          # auto-download
    backbone = DinoV3(size="base",
                      pretrained_encoder="/my/weights.pth") # local weights
"""

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hub & weights configuration
# ---------------------------------------------------------------------------

# GitHub repository that torch.hub uses to fetch DINOv3 source code.
# Cached automatically in ~/.cache/torch/hub/facebookresearch_dinov3_main/
DINOV3_HUB_REPO: str = "facebookresearch/dinov3"

# DINOv3: embed_dim per model size
SIZE_TO_WIDTH: dict[str, int] = {
    "nano":  384,   # dinov3_vits16    (21M, MLP FFN)
    "small": 384,   # dinov3_vits16plus (29M, SwiGLU FFN)
    "base":  768,   # dinov3_vitb16
    "large": 1024,  # dinov3_vitl16  — no public weights; supply pretrained_encoder manually
}

# torch.hub entry-point names defined in DINOv3's hubconf.py
SIZE_TO_HUB_NAME: dict[str, str] = {
    "nano":  "dinov3_vits16",
    "small": "dinov3_vits16plus",
    "base":  "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

# HuggingFace LVD1689M pre-trained weights — (filename, download URL).
# "large" is intentionally absent: no public dinov3_vitl16 weights exist.
# Pass pretrained_encoder="/path/to/vitl16.pth" explicitly when using large.
_HF_BASE = "https://huggingface.co/myn0908/dinov3/resolve/main"
SIZE_TO_WEIGHTS: dict[str, tuple[str, str]] = {
    "nano": (
        "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        f"{_HF_BASE}/dinov3_vits16_pretrain_lvd1689m-08c60483.pth?download=true",
    ),
    "small": (
        "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        f"{_HF_BASE}/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth?download=true",
    ),
    "base": (
        "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        f"{_HF_BASE}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?download=true",
    ),
    # "large" excluded — no public LVD1689M ViT-L weights available.
}

# Kept for backwards-compatibility with call-sites that used the old names.
size_to_width    = SIZE_TO_WIDTH
size_to_hub_name = SIZE_TO_HUB_NAME


# ---------------------------------------------------------------------------
# Weight resolution & loading
# ---------------------------------------------------------------------------

def _resolve_weights_path(size: str, pretrained_encoder: Optional[str]) -> str:
    """Return a local path to DINOv3 weights, auto-downloading if needed.

    Priority
    --------
    1. *pretrained_encoder* — if set and file exists, use as-is.
    2. ``~/.cache/torch/hub/checkpoints/<filename>`` — cached HuggingFace
       download; fetched automatically on first call for nano / small / base.

    Note: ``size="large"`` has no public auto-download weights.
    Supply ``pretrained_encoder="/path/to/dinov3_vitl16.pth"`` explicitly.
    """
    if pretrained_encoder is not None:
        p = Path(pretrained_encoder).expanduser().resolve()
        if p.exists():
            logger.info("DINOv3: using local weights: %s", p)
            return str(p)
        raise FileNotFoundError(
            f"pretrained_encoder path not found: '{pretrained_encoder}'"
        )

    # Large has no public auto-download — require explicit path
    if size not in SIZE_TO_WEIGHTS:
        raise ValueError(
            f"No auto-download weights available for DINOv3 size='{size}'. "
            f"Available auto-download sizes: {sorted(SIZE_TO_WEIGHTS)}. "
            f"For '{size}', provide: "
            f"pretrained_encoder='/path/to/dinov3_{SIZE_TO_HUB_NAME.get(size, size)}.pth'"
        )

    fname, url = SIZE_TO_WEIGHTS[size]
    cache_dir  = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / fname

    if not dest.exists():
        logger.info("DINOv3: downloading weights '%s' from HuggingFace …", fname)
        print(f"Downloading DINOv3 weights ({fname}) …", flush=True)
        torch.hub.download_url_to_file(url, str(dest), progress=True)
        print(f"DINOv3 weights cached → {dest}")
    else:
        logger.debug("DINOv3: using cached weights: %s", dest)

    return str(dest)


def _load_dinov3_hub(hub_name: str, weights_path: str) -> nn.Module:
    """Load DINOv3 via ``torch.hub`` (auto-downloads source from GitHub).

    Source code is cached in ``~/.cache/torch/hub/facebookresearch_dinov3_main/``.
    Set ``TORCH_HOME`` to override the cache root.
    """
    logger.info(
        "Loading DINOv3 '%s' from hub='%s'  weights=%s",
        hub_name, DINOV3_HUB_REPO, weights_path,
    )
    return torch.hub.load(
        DINOV3_HUB_REPO,
        hub_name,
        force_reload=False,
        weights=weights_path,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_enable_grad_checkpointing(model: nn.Module) -> bool:
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        return True
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return True
    patched = 0
    for m in model.modules():
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = True
            patched += 1
    return patched > 0


# ---------------------------------------------------------------------------
# Adapter: torch.hub model → RF-DETR feature-map format
# ---------------------------------------------------------------------------

class TorchHubDinov3BackboneAdapter(nn.Module):
    """Normalise DINOv3 ``torch.hub`` outputs to the RF-DETR multi-scale format.

    Calls ``model.get_intermediate_layers`` and converts the resulting token
    sequences into 4-D feature maps ``(B, C, H', W')``.
    """

    def __init__(
        self,
        model: nn.Module,
        out_feature_indexes: List[int],
        patch_size: int,
    ) -> None:
        super().__init__()
        self.model              = model
        self.out_feature_indexes = out_feature_indexes
        self.patch_size         = patch_size

    @staticmethod
    def _extract_hidden_tensor(layer_output) -> torch.Tensor:
        if isinstance(layer_output, torch.Tensor):
            return layer_output
        if isinstance(layer_output, (tuple, list)):
            for item in layer_output:
                if isinstance(item, torch.Tensor):
                    return item
        raise TypeError(
            f"Unsupported intermediate layer output type: {type(layer_output)}"
        )

    @staticmethod
    def _infer_token_grid(
        num_tokens: int, input_hw: Tuple[int, int]
    ) -> Tuple[int, int]:
        h_in, w_in = input_hw
        if h_in > 0 and w_in > 0:
            target_ratio = h_in / w_in
            best: Optional[Tuple[int, int]] = None
            best_err = float("inf")
            for h in range(1, int(math.sqrt(num_tokens)) + 1):
                if num_tokens % h != 0:
                    continue
                w   = num_tokens // h
                err = abs((h / w) - target_ratio)
                if err < best_err:
                    best     = (h, w)
                    best_err = err
            if best is not None:
                return best

        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                f"Cannot infer a rectangular token grid from num_tokens={num_tokens}"
            )
        return side, side

    def _tokens_to_feature_map(
        self, tokens: torch.Tensor, input_hw: Tuple[int, int]
    ) -> torch.Tensor:
        if tokens.dim() == 4:
            return tokens
        if tokens.dim() != 3:
            raise ValueError(
                f"Expected token tensor with 3 or 4 dims, got shape {tuple(tokens.shape)}"
            )

        b, n, c  = tokens.shape
        h_exp    = input_hw[0] // self.patch_size
        w_exp    = input_hw[1] // self.patch_size

        if n == h_exp * w_exp + 1:      # strip CLS token
            tokens = tokens[:, 1:, :]
            n      = tokens.shape[1]

        if n == h_exp * w_exp:
            h, w = h_exp, w_exp
        else:
            h, w = self._infer_token_grid(n, input_hw)

        return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...]]:
        if not hasattr(self.model, "get_intermediate_layers"):
            raise RuntimeError(
                "The loaded DINOv3 backbone does not expose "
                "`get_intermediate_layers`, which is required to extract "
                "multi-scale RF-DETR features."
            )

        layer_outputs = self.model.get_intermediate_layers(
            x,
            n=self.out_feature_indexes,
            reshape=False,
            return_class_token=False,
        )

        input_hw: Tuple[int, int] = (x.shape[2], x.shape[3])
        feature_maps = [
            self._tokens_to_feature_map(
                self._extract_hidden_tensor(lo), input_hw
            )
            for lo in layer_outputs
        ]
        return (tuple(feature_maps),)


# ---------------------------------------------------------------------------
# Main backbone module
# ---------------------------------------------------------------------------

class DinoV3(nn.Module):
    """DINOv3 backbone for RF-DETR.

    Source code and weights are fetched automatically on first use:

    * **Source code** — ``torch.hub`` downloads ``facebookresearch/dinov3``
      from GitHub and caches it in ``~/.cache/torch/hub/``.
    * **Weights** — ``torch.hub.download_url_to_file`` fetches the
      LVD1689M checkpoint from HuggingFace and caches it in
      ``~/.cache/torch/hub/checkpoints/``.

    No local ``dinov3/`` directory or ``DINOV3_REPO_DIR`` env var is needed.

    Parameters
    ----------
    size:
        ``"nano"`` | ``"small"`` | ``"base"`` — weights are downloaded
        automatically from HuggingFace on first use.
        ``"large"`` — **no public weights available**; you must supply
        ``pretrained_encoder="/path/to/dinov3_vitl16.pth"`` explicitly.
    pretrained_encoder:
        Optional path to a local ``.pth`` weights file.  When ``None``
        (default) weights are downloaded automatically (nano/small/base only).
    shape:
        ``(H, W)`` of the input images in pixels.
    out_feature_indexes:
        Layer indices (0-based) from which to extract intermediate features.
    use_windowed_attn:
        Split the image into ``num_windows × num_windows`` tiles and run the
        encoder on each tile independently.
    num_windows:
        Number of window tiles along each spatial axis.
    gradient_checkpointing:
        Enable activation checkpointing to trade compute for memory.
    """

    def __init__(
        self,
        shape:                  Tuple[int, int]  = (640, 640),
        out_feature_indexes:    List[int]         = [2, 4, 5, 9],
        size:                   str               = "base",
        use_registers:          bool              = False,
        use_windowed_attn:      bool              = False,
        gradient_checkpointing: bool              = False,
        load_dinov3_weights:    bool              = True,
        pretrained_encoder:     Optional[str]     = None,
        patch_size:             int               = 16,
        num_windows:            int               = 2,
        positional_encoding_size: int             = 0,
        drop_path_rate:         float             = 0.0,
    ) -> None:
        super().__init__()

        # ── validation ────────────────────────────────────────────────────
        if use_registers:
            raise ValueError(
                "use_registers=True is not supported for the DINOv3 hub backbone."
            )
        if not load_dinov3_weights:
            raise ValueError(
                "load_dinov3_weights=False is not supported — RF-DETR requires "
                "pre-trained DINOv3 weights."
            )
        if size not in SIZE_TO_HUB_NAME:
            raise ValueError(
                f"Unknown DINOv3 size '{size}'. "
                f"Choose from: {sorted(SIZE_TO_HUB_NAME)}."
            )
        if use_windowed_attn and num_windows > 1:
            for axis_name, axis_size in zip(("height", "width"), shape):
                if axis_size % (patch_size * num_windows) != 0:
                    raise ValueError(
                        f"When use_windowed_attn=True the input {axis_name} must be "
                        f"divisible by patch_size × num_windows = "
                        f"{patch_size} × {num_windows} = {patch_size * num_windows}, "
                        f"but got {axis_size}."
                    )
        if drop_path_rate > 0.0:
            logger.warning(
                "drop_path_rate=%.4f is ignored for the DINOv3 hub backbone.",
                drop_path_rate,
            )
        if positional_encoding_size != 0:
            logger.warning(
                "positional_encoding_size is ignored for the DINOv3 hub backbone."
            )

        # ── config ────────────────────────────────────────────────────────
        self.patch_size        = patch_size
        self.shape             = shape
        self.num_windows       = num_windows
        self.use_windowed_attn = use_windowed_attn and num_windows > 1
        self._export           = False

        # ── load weights + model via torch.hub ───────────────────────────
        hub_name     = SIZE_TO_HUB_NAME[size]
        weights_path = _resolve_weights_path(size, pretrained_encoder)
        hub_model    = _load_dinov3_hub(hub_name, weights_path)

        # ── optional gradient checkpointing ──────────────────────────────
        if gradient_checkpointing:
            ok = _try_enable_grad_checkpointing(hub_model)
            if ok:
                logger.info("Gradient checkpointing enabled on DINOv3 hub model.")
            else:
                logger.warning(
                    "Could not enable gradient checkpointing on the DINOv3 hub "
                    "model — no supported API found."
                )

        # ── wrap in adapter ───────────────────────────────────────────────
        self.encoder = TorchHubDinov3BackboneAdapter(
            model               = hub_model,
            out_feature_indexes = out_feature_indexes,
            patch_size          = patch_size,
        )
        self._out_feature_channels = [SIZE_TO_WIDTH[size]] * len(out_feature_indexes)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> None:
        self._export = True

    # ------------------------------------------------------------------
    # Windowing helpers
    # ------------------------------------------------------------------

    def _split_into_windows(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        B, C, H, W = x.shape
        W_n        = self.num_windows
        h_win, w_win = H // W_n, W // W_n
        x_win = (
            x.reshape(B, C, W_n, h_win, W_n, w_win)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B * W_n * W_n, C, h_win, w_win)
        )
        return x_win, B, W_n, h_win, w_win

    @staticmethod
    def _merge_windows(feat: torch.Tensor, B: int, W_n: int) -> torch.Tensor:
        _bw, C, h_f, w_f = feat.shape
        return (
            feat.reshape(B, W_n, W_n, C, h_f, w_f)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(B, C, W_n * h_f, W_n * w_f)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        H, W = x.shape[2], x.shape[3]
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input shape must be divisible by patch_size={self.patch_size}, "
                f"got {tuple(x.shape)}."
            )

        if self.use_windowed_attn:
            x_win, B, W_n, h_win, w_win = self._split_into_windows(x)
            if h_win % self.patch_size != 0 or w_win % self.patch_size != 0:
                raise ValueError(
                    f"Window size ({h_win}, {w_win}) is not divisible by "
                    f"patch_size={self.patch_size}."
                )
            window_features: Tuple[torch.Tensor, ...] = self.encoder(x_win)[0]
            return [self._merge_windows(feat, B, W_n) for feat in window_features]

        return list(self.encoder(x)[0])
