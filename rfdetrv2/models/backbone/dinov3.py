# import math
# from pathlib import Path
# import logging

# import torch
# import torch.nn as nn

# logger = logging.getLogger(__name__)

# size_to_width = {
#     "tiny": 192,
#     "small": 384,
#     "base": 768,
#     "large": 1024,
# }

# size_to_hub_name = {
#     "small": "dinov3_vits16plus",
#     "base": "dinov3_vitb16",
#     "large": "dinov3_vitl16",
# }

# DEFAULT_DINOV3_REPO_DIR = Path(__file__).resolve().parents[3] / "dinov3"


# def parse_torch_hub_source_spec(spec: str | None) -> tuple[str | None, str | None]:
#     """Parse optional `repo_or_dir::weights` source spec. Returns (None, None) if spec is None."""
#     if not spec:
#         return None, None
#     if "::" not in spec:
#         return spec, None
#     repo_or_dir, weights = spec.split("::", maxsplit=1)
#     return repo_or_dir, (weights or None)


# class TorchHubDinov3BackboneAdapter(nn.Module):
#     """Adapter that normalizes DINOv3 torch.hub outputs to RF-DETR format."""

#     def __init__(
#         self,
#         model: nn.Module,
#         out_feature_indexes: list[int],
#         patch_size: int,
#     ) -> None:
#         super().__init__()
#         self.model = model
#         self.out_feature_indexes = out_feature_indexes
#         self.patch_size = patch_size

#     @staticmethod
#     def _extract_hidden_tensor(layer_output):
#         if isinstance(layer_output, torch.Tensor):
#             return layer_output
#         if isinstance(layer_output, (tuple, list)):
#             for item in layer_output:
#                 if isinstance(item, torch.Tensor):
#                     return item
#         raise TypeError(f"Unsupported intermediate layer output type: {type(layer_output)}")

#     @staticmethod
#     def _infer_token_grid(num_tokens: int, input_hw: tuple[int, int]) -> tuple[int, int]:
#         h_in, w_in = input_hw
#         if h_in > 0 and w_in > 0:
#             target_ratio = h_in / w_in
#             best = None
#             best_err = float("inf")
#             for h in range(1, int(math.sqrt(num_tokens)) + 1):
#                 if num_tokens % h != 0:
#                     continue
#                 w = num_tokens // h
#                 err = abs((h / w) - target_ratio)
#                 if err < best_err:
#                     best = (h, w)
#                     best_err = err
#             if best is not None:
#                 return best

#         side = int(math.isqrt(num_tokens))
#         if side * side != num_tokens:
#             raise ValueError(f"Cannot infer token grid from num_tokens={num_tokens}")
#         return side, side

#     def _tokens_to_feature_map(self, tokens: torch.Tensor, input_hw: tuple[int, int]) -> torch.Tensor:
#         if tokens.dim() == 4:
#             return tokens
#         if tokens.dim() != 3:
#             raise ValueError(f"Expected token tensor with 3 or 4 dims, got shape {tuple(tokens.shape)}")

#         b, n, c = tokens.shape
#         h = input_hw[0] // self.patch_size
#         w = input_hw[1] // self.patch_size
#         if n == h * w + 1:
#             tokens = tokens[:, 1:, :]
#             n = tokens.shape[1]
#         if n != h * w:
#             h, w = self._infer_token_grid(n, input_hw)
#         return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

#     def forward(self, x: torch.Tensor):
#         if not hasattr(self.model, "get_intermediate_layers"):
#             raise RuntimeError(
#                 "Loaded DINOv3 backbone does not expose get_intermediate_layers, which is required"
#                 " to extract multi-scale RF-DETR features."
#             )
#         layer_outputs = self.model.get_intermediate_layers(
#             x,
#             n=self.out_feature_indexes,
#             reshape=False,
#             return_class_token=False,
#         )

#         feature_maps = []
#         for layer_output in layer_outputs:
#             hidden = self._extract_hidden_tensor(layer_output)
#             feature_maps.append(self._tokens_to_feature_map(hidden, (x.shape[2], x.shape[3])))

#         return (tuple(feature_maps),)


# class DinoV3(nn.Module):
#     def __init__(
#         self,
#         shape: tuple[int, int] = (640, 640),
#         out_feature_indexes: list[int] = [2, 4, 5, 9],
#         size: str = "base",
#         use_registers: bool = True,
#         use_windowed_attn: bool = True,
#         gradient_checkpointing: bool = False,
#         load_dinov3_weights: bool = True,
#         pretrained_encoder: str | None = None,
#         patch_size: int = 14,
#         num_windows: int = 4,
#         positional_encoding_size: int = 37,
#         drop_path_rate: float = 0.0,
#     ) -> None:
#         super().__init__()
#         if use_registers:
#             raise ValueError("DINOv3 registers mode is not supported in RFDETRV2.")
#         if use_windowed_attn:
#             raise ValueError("Windowed attention is not supported in RFDETRV2.")
#         if gradient_checkpointing:
#             raise ValueError("Gradient checkpointing is not supported for the DINOv3 hub backbone.")
#         if drop_path_rate > 0.0:
#             logger.warning("drop_path_rate is ignored for the DINOv3 hub backbone.")
#         if not load_dinov3_weights:
#             raise ValueError("RFDETRV2 requires loading DINOv3 weights from local .pth files.")

#         hub_name = size_to_hub_name[size]
#         self.patch_size = patch_size
#         self.shape = shape
#         self.num_windows = num_windows

#         repo_or_dir, weights = parse_torch_hub_source_spec(pretrained_encoder)
#         if pretrained_encoder is None:
#             project_root = Path(__file__).resolve().parents[3]
#             weight_candidates = sorted(project_root.glob(f"{hub_name}*.pth"))
#             if weight_candidates:
#                 repo_or_dir = str(DEFAULT_DINOV3_REPO_DIR)
#                 weights = str(weight_candidates[0])
#             else:
#                 raise FileNotFoundError(
#                     f"Could not find local DINOv3 weights for '{hub_name}'. "
#                     f"Either place a file like '{hub_name}*.pth' in the project root, or pass "
#                     f"pretrained_encoder='/path/to/weights.pth' (or 'path/to/dinov3::path/to/weights.pth')."
#                 )
#         else:
#             # pretrained_encoder set: may be "repo::weights" or path to .pth file
#             weights_path = Path(pretrained_encoder).expanduser().resolve()
#             if "::" not in pretrained_encoder and weights_path.suffix == ".pth" and weights_path.exists():
#                 if not DEFAULT_DINOV3_REPO_DIR.exists():
#                     raise FileNotFoundError(
#                         f"Local DINOv3 repo not found at {DEFAULT_DINOV3_REPO_DIR}. "
#                         "Ensure the 'dinov3' folder exists in your project root, or use "
#                         "pretrained_encoder='path/to/dinov3::path/to/weights.pth' to specify both."
#                     )
#                 repo_or_dir = str(DEFAULT_DINOV3_REPO_DIR)
#                 weights = str(weights_path)
#                 logger.info(f"Loading DINOv3 weights from local file: {weights}")
#             elif "::" in pretrained_encoder:
#                 if not Path(repo_or_dir).exists():
#                     raise FileNotFoundError(f"DINOv3 repo not found at {repo_or_dir}")
#                 if not weights:
#                     raise ValueError("pretrained_encoder 'repo::weights' format requires a weights path after ::")
#                 if not Path(weights).exists():
#                     raise FileNotFoundError(f"Weights file not found at {weights}")
#             else:
#                 raise FileNotFoundError(
#                     f"pretrained_encoder must be a path to an existing .pth file, got: {pretrained_encoder}"
#                 )

#         source = "local"
#         logger.info(f"Loading DINOv3 from {repo_or_dir} (model={hub_name})")
#         hub_model = torch.hub.load(repo_or_dir, hub_name, source=source, weights=weights)
#         self.encoder = TorchHubDinov3BackboneAdapter(
#             model=hub_model,
#             out_feature_indexes=out_feature_indexes,
#             patch_size=patch_size,
#         )

#         self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
#         self._export = False

#     def export(self) -> None:
#         self._export = True

#     def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
#         block_size = self.patch_size
#         if x.shape[2] % block_size != 0 or x.shape[3] % block_size != 0:
#             raise ValueError(f"Backbone requires input shape to be divisible by {block_size}, but got {x.shape}")
#         x = self.encoder(x)
#         return list(x[0])










"""
DINOv3 backbone adapter for RF-DETR.

Windowed-attention implementation
----------------------------------
DINOv3 is loaded as a black-box from a local torch.hub repository, so we cannot
modify the internals of each transformer layer.  Instead, windowed attention is
realised at the *input level*:

    1. The input image (B, C, H, W) is tiled into W_n × W_n non-overlapping
       windows → (B·W_n², C, H/W_n, W/W_n).
    2. The full DINOv3 encoder runs on each window independently (each window
       token sequence only attends to sibling tokens — identical semantics to
       windowed self-attention restricted to all layers).
    3. The resulting per-window feature maps are reassembled into the original
       spatial layout → (B, C, H', W').

This is equivalent to running every transformer layer in windowed mode.
The tradeoff vs. the hybrid DINOv2 scheme (some layers windowed, some global)
is that there is no cross-window information exchange inside the backbone;
the MultiScaleProjector's ConvTranspose / C2f layers partially compensate for
this at the neck level.

Usage
-----
    # 2 × 2 windows (4 windows total) on a 640 × 640 image:
    backbone = DinoV3(size="base", use_windowed_attn=True, num_windows=2)
"""

import math
from pathlib import Path
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model size tables
# ---------------------------------------------------------------------------

# DINOv3: vits16 (21M, MLP FFN) vs vits16plus (29M, SwiGLU FFN) — both embed_dim=384
SIZE_TO_WIDTH: dict[str, int] = {
    "tiny":  192,   # reserved (no DINOv3 hub model)
    "nano":  384,   # dinov3_vits16
    "small": 384,   # dinov3_vits16plus
    "base":  768,
    "large": 1024,
}

SIZE_TO_HUB_NAME: dict[str, str] = {
    "nano":  "dinov3_vits16",      # ViT-S 21M, MLP FFN
    "small": "dinov3_vits16plus",  # ViT-S+ 29M, SwiGLU FFN
    "base":  "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

# Kept for backwards-compatibility with call-sites that used the old names.
size_to_width   = SIZE_TO_WIDTH
size_to_hub_name = SIZE_TO_HUB_NAME

DEFAULT_DINOV3_REPO_DIR = Path(__file__).resolve().parents[3] / "dinov3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_torch_hub_source_spec(
    spec: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Parse optional ``repo_or_dir::weights`` source spec.

    Returns ``(None, None)`` when *spec* is ``None``.
    """
    if not spec:
        return None, None
    if "::" not in spec:
        return spec, None
    repo_or_dir, weights = spec.split("::", maxsplit=1)
    return repo_or_dir, (weights or None)


def _try_enable_grad_checkpointing(model: nn.Module) -> bool:
    """Attempt to enable gradient checkpointing on *model*.

    Returns ``True`` when the call succeeded, ``False`` otherwise.
    """
    # timm / DINOv3 convention
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        return True
    # HuggingFace convention
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return True
    # PyTorch torch.utils.checkpoint manual block-level patching
    # (last resort — iterate over ViT blocks)
    patched = 0
    for module in model.modules():
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = True
            patched += 1
    return patched > 0


# ---------------------------------------------------------------------------
# Adapter: torch.hub model → RF-DETR feature-map format
# ---------------------------------------------------------------------------

class TorchHubDinov3BackboneAdapter(nn.Module):
    """Normalise DINOv3 ``torch.hub`` outputs to the RF-DETR multi-scale format.

    The adapter calls ``model.get_intermediate_layers`` and converts the
    resulting token sequences into 4-D feature maps ``(B, C, H', W')``.
    """

    def __init__(
        self,
        model: nn.Module,
        out_feature_indexes: List[int],
        patch_size: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.out_feature_indexes = out_feature_indexes
        self.patch_size = patch_size

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_hidden_tensor(layer_output) -> torch.Tensor:
        """Extract the hidden-state tensor from a (possibly wrapped) layer output."""
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
        """Infer the (h, w) grid size from the number of patch tokens.

        Prefers the factorisation whose aspect ratio is closest to *input_hw*.
        Falls back to the square root when the input shape is not available.
        """
        h_in, w_in = input_hw
        if h_in > 0 and w_in > 0:
            target_ratio = h_in / w_in
            best: Optional[Tuple[int, int]] = None
            best_err = float("inf")
            for h in range(1, int(math.sqrt(num_tokens)) + 1):
                if num_tokens % h != 0:
                    continue
                w = num_tokens // h
                err = abs((h / w) - target_ratio)
                if err < best_err:
                    best = (h, w)
                    best_err = err
            if best is not None:
                return best

        side = int(math.isqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                f"Cannot infer a rectangular token grid from num_tokens={num_tokens}"
            )
        return side, side

    # ------------------------------------------------------------------
    # Token → spatial feature map
    # ------------------------------------------------------------------

    def _tokens_to_feature_map(
        self, tokens: torch.Tensor, input_hw: Tuple[int, int]
    ) -> torch.Tensor:
        """Reshape a ``(B, N, C)`` token tensor into ``(B, C, H', W')``.

        Strips the CLS token when present (detected by ``N == H'*W' + 1``).
        DINOv3 register tokens are already removed by ``get_intermediate_layers``
        when ``return_class_token=False``; if they are present they are handled
        by the grid-inference fallback.
        """
        if tokens.dim() == 4:
            # Already spatial — nothing to do.
            return tokens
        if tokens.dim() != 3:
            raise ValueError(
                f"Expected token tensor with 3 or 4 dims, got shape {tuple(tokens.shape)}"
            )

        b, n, c = tokens.shape
        h_exp = input_hw[0] // self.patch_size
        w_exp = input_hw[1] // self.patch_size

        # Strip CLS token if present
        if n == h_exp * w_exp + 1:
            tokens = tokens[:, 1:, :]
            n = tokens.shape[1]

        # Normal case
        if n == h_exp * w_exp:
            h, w = h_exp, w_exp
        else:
            # Fallback: infer grid (covers models with extra register tokens)
            h, w = self._infer_token_grid(n, input_hw)

        return tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...]]:
        """Run the DINOv3 encoder and return ``((f0, f1, ..., fn),)``."""
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

    Parameters
    ----------
    shape:
        ``(H, W)`` of the input images in pixels.  Both dimensions must be
        divisible by *patch_size* × *num_windows*.
    out_feature_indexes:
        Layer indices (0-based) from which to extract intermediate features.
        These become the multi-scale inputs to the ``MultiScaleProjector``.
    size:
        One of ``"small"``, ``"base"``, ``"large"``.
    use_windowed_attn:
        Split the image into ``num_windows × num_windows`` tiles and run the
        encoder on each tile independently before merging the feature maps.
        Reduces self-attention complexity from O((H·W)²) to
        O((H·W / num_windows²)²) per window.
    num_windows:
        Number of window tiles along each spatial axis (total tiles =
        ``num_windows²``).  Requires ``H`` and ``W`` to be divisible by
        ``patch_size × num_windows``.
    gradient_checkpointing:
        Enable activation checkpointing to trade compute for memory.  Calls
        ``model.set_grad_checkpointing(True)`` if available; warns and
        continues otherwise.
    pretrained_encoder:
        One of:
        - ``None``: auto-discover ``<hub_name>*.pth`` in the project root.
        - ``"/path/to/weights.pth"``: load weights from a specific file using
          the default DINOv3 repo at ``DEFAULT_DINOV3_REPO_DIR``.
        - ``"path/to/dinov3::path/to/weights.pth"``: specify both the repo
          directory and the weights file explicitly.
    """

    def __init__(
        self,
        shape: Tuple[int, int] = (640, 640),
        out_feature_indexes: List[int] = [2, 4, 5, 9],
        size: str = "base",
        use_registers: bool = False,
        use_windowed_attn: bool = False,
        gradient_checkpointing: bool = False,
        load_dinov3_weights: bool = True,
        pretrained_encoder: Optional[str] = None,
        patch_size: int = 16,
        num_windows: int = 2,
        positional_encoding_size: int = 0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Validate arguments
        # ------------------------------------------------------------------
        if use_registers:
            raise ValueError(
                "use_registers=True is not supported for the DINOv3 hub backbone. "
                "DINOv3 uses register tokens internally; they are stripped before "
                "feature maps are returned and cannot be used as separate outputs."
            )
        if not load_dinov3_weights:
            raise ValueError(
                "load_dinov3_weights=False is not supported. "
                "RF-DETR requires pre-trained DINOv3 weights."
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
                "drop_path_rate=%.4f is ignored for the DINOv3 hub backbone "
                "(stochastic depth must be configured in the hub repo itself).",
                drop_path_rate,
            )
        if positional_encoding_size != 0:
            logger.warning(
                "positional_encoding_size is ignored for the DINOv3 hub backbone."
            )

        # ------------------------------------------------------------------
        # Store config
        # ------------------------------------------------------------------
        self.patch_size = patch_size
        self.shape = shape
        self.num_windows = num_windows
        self.use_windowed_attn = use_windowed_attn and num_windows > 1
        self._export = False

        # ------------------------------------------------------------------
        # Resolve repo / weights paths
        # ------------------------------------------------------------------
        hub_name = SIZE_TO_HUB_NAME[size]
        repo_or_dir, weights = self._resolve_weights(
            hub_name, pretrained_encoder
        )

        # ------------------------------------------------------------------
        # Load from torch.hub
        # ------------------------------------------------------------------
        logger.info(
            "Loading DINOv3 '%s' from repo=%s  weights=%s",
            hub_name,
            repo_or_dir,
            weights,
        )
        hub_model = torch.hub.load(
            repo_or_dir, hub_name, source="local", weights=weights
        )

        # ------------------------------------------------------------------
        # Optional: gradient checkpointing
        # ------------------------------------------------------------------
        if gradient_checkpointing:
            ok = _try_enable_grad_checkpointing(hub_model)
            if ok:
                logger.info("Gradient checkpointing enabled on DINOv3 hub model.")
            else:
                logger.warning(
                    "Could not enable gradient checkpointing on the DINOv3 hub "
                    "model — no supported API found.  Training will use more "
                    "memory.  Consider updating the DINOv3 hub repo."
                )

        # ------------------------------------------------------------------
        # Wrap in adapter
        # ------------------------------------------------------------------
        self.encoder = TorchHubDinov3BackboneAdapter(
            model=hub_model,
            out_feature_indexes=out_feature_indexes,
            patch_size=patch_size,
        )

        self._out_feature_channels = [SIZE_TO_WIDTH[size]] * len(out_feature_indexes)

    # ------------------------------------------------------------------
    # Static / class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_weights(
        hub_name: str, pretrained_encoder: Optional[str]
    ) -> Tuple[str, str]:
        """Return ``(repo_or_dir, weights_path)`` for ``torch.hub.load``."""
        if pretrained_encoder is None:
            project_root = Path(__file__).resolve().parents[3]
            candidates = sorted(project_root.glob(f"{hub_name}*.pth"))
            if not candidates:
                raise FileNotFoundError(
                    f"Could not find local DINOv3 weights for '{hub_name}'. "
                    f"Place a file like '{hub_name}*.pth' in the project root, "
                    "or pass pretrained_encoder='/path/to/weights.pth' "
                    "(or 'path/to/dinov3::path/to/weights.pth')."
                )
            return str(DEFAULT_DINOV3_REPO_DIR), str(candidates[0])

        repo_or_dir, weights = parse_torch_hub_source_spec(pretrained_encoder)

        if "::" in pretrained_encoder:
            # Explicit "repo::weights" form
            if not Path(repo_or_dir).exists():
                raise FileNotFoundError(
                    f"DINOv3 repo not found at '{repo_or_dir}'"
                )
            if not weights:
                raise ValueError(
                    "The 'repo::weights' format requires a weights path after '::'."
                )
            if not Path(weights).exists():
                raise FileNotFoundError(
                    f"DINOv3 weights file not found at '{weights}'"
                )
            return repo_or_dir, weights

        # Plain path to a .pth file
        weights_path = Path(pretrained_encoder).expanduser().resolve()
        if weights_path.suffix != ".pth" or not weights_path.exists():
            raise FileNotFoundError(
                f"pretrained_encoder must be a path to an existing .pth file, "
                f"got: '{pretrained_encoder}'"
            )
        if not DEFAULT_DINOV3_REPO_DIR.exists():
            raise FileNotFoundError(
                f"Local DINOv3 repo not found at '{DEFAULT_DINOV3_REPO_DIR}'. "
                "Ensure the 'dinov3' folder exists in your project root, or "
                "use 'path/to/dinov3::path/to/weights.pth' to specify both."
            )
        logger.info("Loading DINOv3 weights from local file: %s", weights_path)
        return str(DEFAULT_DINOV3_REPO_DIR), str(weights_path)

    # ------------------------------------------------------------------
    # Windowing helpers
    # ------------------------------------------------------------------

    def _split_into_windows(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """Tile ``(B, C, H, W)`` into ``(B·W_n², C, H/W_n, W/W_n)`` windows.

        Returns
        -------
        x_win : torch.Tensor  shape ``(B·W_n², C, h_win, w_win)``
        B, W_n, h_win, w_win : ints for use in :meth:`_merge_windows`.
        """
        B, C, H, W = x.shape
        W_n = self.num_windows
        h_win, w_win = H // W_n, W // W_n

        # (B, C, H, W)
        # → (B, C, W_n, h_win, W_n, w_win)   [split spatial dims]
        # → (B, W_n, W_n, C, h_win, w_win)   [bring window indices forward]
        # → (B·W_n², C, h_win, w_win)         [flatten batch × windows]
        x_win = (
            x.reshape(B, C, W_n, h_win, W_n, w_win)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B * W_n * W_n, C, h_win, w_win)
        )
        return x_win, B, W_n, h_win, w_win

    @staticmethod
    def _merge_windows(
        feat: torch.Tensor, B: int, W_n: int
    ) -> torch.Tensor:
        """Reassemble per-window feature maps into a single spatial feature map.

        Parameters
        ----------
        feat : ``(B·W_n², C, h', w')``
        B    : original batch size
        W_n  : number of windows per axis

        Returns
        -------
        ``(B, C, W_n·h', W_n·w')``
        """
        _bw, C, h_f, w_f = feat.shape
        # (B·W_n², C, h', w')
        # → (B, W_n, W_n, C, h', w')
        # → (B, C, W_n, h', W_n, w')
        # → (B, C, W_n·h', W_n·w')
        return (
            feat.reshape(B, W_n, W_n, C, h_f, w_f)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(B, C, W_n * h_f, W_n * w_f)
        )

    # ------------------------------------------------------------------
    # Export mode
    # ------------------------------------------------------------------

    def export(self) -> None:
        """Switch the module to TorchScript / ONNX export mode."""
        self._export = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run the DINOv3 encoder and return a list of spatial feature maps.

        When ``use_windowed_attn=True`` the input is split into
        ``num_windows × num_windows`` tiles, each tile is processed by the
        encoder independently, and the resulting feature maps are merged back
        into full-resolution grids.

        Parameters
        ----------
        x : ``(B, 3, H, W)`` — pixel values, normalised.

        Returns
        -------
        List of ``(B, C, H_i, W_i)`` tensors, one per entry in
        ``out_feature_indexes``.
        """
        H, W = x.shape[2], x.shape[3]

        # Sanity check: spatial dims divisible by patch size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Backbone requires input shape to be divisible by "
                f"patch_size={self.patch_size}, but got {tuple(x.shape)}."
            )

        # ---------------------------------------------------------------
        # Path A — windowed attention
        # ---------------------------------------------------------------
        if self.use_windowed_attn:
            x_win, B, W_n, h_win, w_win = self._split_into_windows(x)

            # Sanity check: windows must also be patch-aligned
            if h_win % self.patch_size != 0 or w_win % self.patch_size != 0:
                raise ValueError(
                    f"Window size ({h_win}, {w_win}) is not divisible by "
                    f"patch_size={self.patch_size}.  Adjust num_windows or "
                    f"the input resolution."
                )

            # Run encoder on all windows (shape: B·W_n², C, h_win, w_win)
            window_features: Tuple[torch.Tensor, ...] = self.encoder(x_win)[0]

            # Merge each scale back to full resolution
            return [
                self._merge_windows(feat, B, W_n)
                for feat in window_features
            ]

        # ---------------------------------------------------------------
        # Path B — standard (full-image) forward
        # ---------------------------------------------------------------
        return list(self.encoder(x)[0])