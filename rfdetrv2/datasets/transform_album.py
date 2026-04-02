# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR / Conditional DETR / DETR
# Augmentations ported to albumentations for faster throughput.
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
Uses albumentations (OpenCV-backed) for all spatial operations.
"""
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import PIL
from numbers import Number

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import torch
import torchvision.transforms.functional as F

from rfdetrv2.util.box_ops import box_xyxy_to_cxcywh
from rfdetrv2.util.misc import interpolate


# ─── Shared BboxParams used by every spatial transform ───────────────────────
_BBOX_PARAMS = A.BboxParams(
    format="pascal_voc",
    label_fields=["class_labels", "orig_indices"],
    min_area=1.0,
    min_visibility=0.0,
    check_each_transform=True,
)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _to_numpy(img: Any) -> np.ndarray:
    if isinstance(img, PIL.Image.Image):
        return np.asarray(img)
    if isinstance(img, torch.Tensor):
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return np.asarray(img)


def _image_wh(img: Any) -> Tuple[int, int]:
    if isinstance(img, PIL.Image.Image):
        return img.size
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    return (w, h)


def _apply_albu(
    compose: A.Compose,
    img: Any,
    target: Optional[Dict[str, Any]],
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    was_pil = isinstance(img, PIL.Image.Image)
    img_np = _to_numpy(img)

    if target is None:
        result = compose(
            image=img_np, bboxes=[], class_labels=[], orig_indices=[], masks=[]
        )
        out = PIL.Image.fromarray(result["image"]) if was_pil else result["image"]
        return out, None

    target = target.copy()
    h, w = img_np.shape[:2]

    has_boxes = "boxes" in target and len(target["boxes"]) > 0
    if has_boxes:
        boxes_np = target["boxes"].numpy().copy()
        boxes_np[:, 0::2] = boxes_np[:, 0::2].clip(0, w)
        boxes_np[:, 1::2] = boxes_np[:, 1::2].clip(0, h)
        valid = (boxes_np[:, 2] > boxes_np[:, 0]) & (boxes_np[:, 3] > boxes_np[:, 1])
        boxes_np = boxes_np[valid]
        orig_indices_np = np.where(valid)[0].tolist()
        bboxes = boxes_np.tolist()
        class_labels = target["labels"][valid].tolist()
    else:
        bboxes, class_labels, orig_indices_np = [], [], []

    has_masks = "masks" in target and len(target["masks"]) > 0
    mask_list = (
        [target["masks"][i].numpy() for i in range(len(target["masks"]))]
        if has_masks
        else []
    )

    result = compose(
        image=img_np,
        bboxes=bboxes,
        class_labels=class_labels,
        orig_indices=orig_indices_np,
        masks=mask_list,
    )

    img_out = result["image"]
    new_h, new_w = img_out.shape[:2]

    surviving_local = list(result["orig_indices"])
    new_bboxes = result["bboxes"]
    new_class_labels = result["class_labels"]

    if len(new_bboxes) > 0:
        target["boxes"] = torch.tensor(new_bboxes, dtype=torch.float32)
        target["labels"] = torch.tensor(new_class_labels, dtype=torch.int64)
        b = target["boxes"]
        target["area"] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        if "iscrowd" in target and has_boxes:
            orig_tensor_indices = torch.tensor(
                [orig_indices_np[i] for i in surviving_local], dtype=torch.long
            )
            target["iscrowd"] = target["iscrowd"][orig_tensor_indices]
    else:
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["labels"] = torch.zeros((0,), dtype=torch.int64)
        target["area"] = torch.zeros((0,), dtype=torch.float32)
        if "iscrowd" in target:
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

    if has_masks:
        all_masks_out = result.get("masks", [])
        kept = [all_masks_out[i] for i in surviving_local] if all_masks_out else []
        if kept:
            target["masks"] = torch.from_numpy(np.stack(kept)).bool()
        else:
            target["masks"] = torch.zeros((0, new_h, new_w), dtype=torch.bool)

    target["size"] = torch.tensor([new_h, new_w])

    if was_pil:
        img_out = PIL.Image.fromarray(img_out)

    return img_out, target


# ─── Size computation ─────────────────────────────────────────────────────────

def _get_size_with_aspect_ratio(
    image_size: Tuple[int, int], size: int, max_size: Optional[int] = None
) -> Tuple[int, int]:
    w, h = image_size
    if max_size is not None:
        min_orig = float(min(w, h))
        max_orig = float(max(w, h))
        if max_orig / min_orig * size > max_size:
            size = int(round(max_size * min_orig / max_orig))
    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)
    if w < h:
        return (int(size * h / w), size)
    return (size, int(size * w / h))


# ─── Transform classes ────────────────────────────────────────────────────────

class RandomCrop:
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self._compose = A.Compose(
            [A.RandomCrop(height=size[0], width=size[1])],
            bbox_params=_BBOX_PARAMS,
        )

    def __call__(self, img: Any, target: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _apply_albu(self._compose, img, target)


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int) -> None:
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: Any, target: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        iw, ih = _image_wh(img)
        cw = random.randint(self.min_size, min(iw, self.max_size))
        ch = random.randint(self.min_size, min(ih, self.max_size))
        compose = A.Compose(
            [A.RandomCrop(height=ch, width=cw)],
            bbox_params=_BBOX_PARAMS,
        )
        return _apply_albu(compose, img, target)


class CenterCrop:
    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        self._compose = A.Compose(
            [A.CenterCrop(height=size[0], width=size[1])],
            bbox_params=_BBOX_PARAMS,
        )

    def __call__(self, img: Any, target: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _apply_albu(self._compose, img, target)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self._compose = A.Compose(
            [A.HorizontalFlip(p=p)],
            bbox_params=_BBOX_PARAMS,
        )

    def __call__(self, img: Any, target: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _apply_albu(self._compose, img, target)


class RandomResize:
    def __init__(self, sizes: List[int], max_size: Optional[int] = None) -> None:
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(
        self, img: Any, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        size = random.choice(self.sizes)
        new_h, new_w = _get_size_with_aspect_ratio(_image_wh(img), size, self.max_size)
        compose = A.Compose(
            [A.Resize(height=new_h, width=new_w, interpolation=cv2.INTER_LINEAR)],
            bbox_params=_BBOX_PARAMS,
        )
        return _apply_albu(compose, img, target)


class SquareResize:
    def __init__(self, sizes: List[int]) -> None:
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes

    def __call__(
        self, img: Any, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        size = random.choice(self.sizes)
        compose = A.Compose(
            [A.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR)],
            bbox_params=_BBOX_PARAMS,
        )
        return _apply_albu(compose, img, target)


class RandomPad:
    def __init__(self, max_pad: int) -> None:
        self.max_pad = max_pad

    def __call__(self, img: Any, target: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        iw, ih = _image_wh(img)
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        compose = A.Compose(
            [
                A.PadIfNeeded(
                    min_height=ih + pad_y,
                    min_width=iw + pad_x,
                    position="bottom_right",
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(127, 127, 127),
                    p=1.0,
                )
            ],
            bbox_params=_BBOX_PARAMS,
        )
        return _apply_albu(compose, img, target)


class PILtoNdArray:
    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return np.asarray(img), target


class NdArraytoPIL:
    def __call__(
        self, img: np.ndarray, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        return PIL.Image.fromarray(img.astype(np.uint8)), target


class Pad:
    """
    Pad image to a specified size or multiple of size_divisor.

    pad_mode:
        -1  →  explicit offsets
         0  →  right / bottom  (default)
         1  →  center
         2  →  left / top
    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, int], List[int]]] = None,
        size_divisor: int = 32,
        pad_mode: int = 0,
        offsets: Optional[List[int]] = None,
        fill_value: Tuple[float, float, float] = (127.5, 127.5, 127.5),
    ) -> None:
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"size must be int or Sequence, got {type(size)}")
        if isinstance(size, int):
            size = [size, size]
        assert pad_mode in (-1, 0, 1, 2), "pad_mode must be one of [-1, 0, 1, 2]"
        if pad_mode == -1:
            assert offsets is not None, "offsets required when pad_mode=-1"

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = tuple(int(v) for v in fill_value)
        self.offsets = offsets

    def _apply_numpy(
        self, im: np.ndarray, target: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert im_h <= h and im_w <= w
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            return im.astype(np.float32), target

        if self.pad_mode == -1:
            off_x, off_y = self.offsets
        elif self.pad_mode == 0:
            off_x, off_y = 0, 0
        elif self.pad_mode == 1:
            off_x, off_y = (w - im_w) // 2, (h - im_h) // 2
        else:
            off_x, off_y = w - im_w, h - im_h

        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[off_y:off_y + im_h, off_x:off_x + im_w] = im.astype(np.float32)

        target = target.copy()
        target["size"] = torch.tensor([h, w])

        if self.pad_mode == 0:
            return canvas, target

        if "boxes" in target and len(target["boxes"]) > 0:
            offsets_arr = np.array([off_x, off_y, off_x, off_y], dtype=np.float32)
            target["boxes"] = torch.from_numpy(target["boxes"].numpy() + offsets_arr)

        return canvas, target

    def __call__(
        self, im: np.ndarray, target: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._apply_numpy(im, target)


class RandomExpand:
    """
    Randomly expand the canvas by up to `ratio`×, placing the image at a
    random position on the new canvas.
    """

    def __init__(
        self,
        ratio: float = 4.0,
        prob: float = 0.5,
        fill_value: Union[float, List[float], Tuple] = (127.5, 127.5, 127.5),
    ) -> None:
        assert ratio > 1.01, "expand ratio must be > 1.01"
        self.ratio = ratio
        self.prob = prob
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        self.fill_value = tuple(fill_value)

    def __call__(
        self, img: np.ndarray, target: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if np.random.uniform() < self.prob:
            return img, target

        height, width = img.shape[:2]
        ratio = np.random.uniform(1.0, self.ratio)
        new_h = int(height * ratio)
        new_w = int(width * ratio)
        if new_h <= height or new_w <= width:
            return img, target

        off_y = np.random.randint(0, new_h - height)
        off_x = np.random.randint(0, new_w - width)

        pad_op = Pad(
            size=[new_h, new_w],
            pad_mode=-1,
            offsets=[off_x, off_y],
            fill_value=self.fill_value,
        )
        return pad_op(img, target)


class RandomSelect:
    """Apply transforms1 with probability p, otherwise transforms2."""

    def __init__(self, transforms1: Any, transforms2: Any, p: float = 0.5) -> None:
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img: Any, target: Any) -> Tuple[Any, Any]:
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    """Convert PIL Image or HWC numpy array to a float32 CHW tensor in [0, 1]."""

    def __call__(
        self,
        img: Union[PIL.Image.Image, np.ndarray],
        target: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return F.to_tensor(img), target


class RandomErasing:
    import torchvision.transforms as _T

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import torchvision.transforms as _T
        self.eraser = _T.RandomErasing(*args, **kwargs)

    def __call__(
        self, img: torch.Tensor, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.eraser(img), target


class Normalize:
    """Normalize a CHW tensor and convert boxes from xyxy to normalised cxcywh."""

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(
        self,
        image: torch.Tensor,
        target: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


# ─── MinerU2.5-style Document Augmentation (§3.3.4) ──────────────────────────
# Four categories: Spatial, Background, Color, Degradation.
# Spatial is skipped for layout-analysis samples to preserve bbox coordinates.

class DocumentAugmentation:
    """
    Simulates real-world document degradation across four categories from
    MinerU2.5 §3.3.4:
      - Spatial      : scaling, grid distortion, rotation
      - Background   : texture, weather, shadow, scanlines, watermark
      - Color        : brightness/contrast, illumination, RGB shift
      - Degradation  : PSF blur, vibration blur, gaussian blur, erosion/dilation

    Args:
        apply_spatial: set False for layout-analysis samples to avoid
                       corrupting bounding-box coordinates (MinerU2.5 §3.3.4).
        p:            overall probability of applying augmentation per sample.
    """

    def __init__(self, apply_spatial: bool = True, p: float = 0.5) -> None:
        self.apply_spatial = apply_spatial
        self.p = p

        spatial_ops = (
            [
                A.OneOf([
                    A.ShiftScaleRotate(
                        shift_limit=0.05, scale_limit=0.15,
                        rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1.0,
                    ),
                    A.GridDistortion(
                        num_steps=5, distort_limit=0.2,
                        border_mode=cv2.BORDER_CONSTANT, p=1.0,
                    ),
                    A.Rotate(
                        limit=5, border_mode=cv2.BORDER_CONSTANT,
                        crop_border=True, p=1.0,
                    ),
                ], p=0.4),
            ]
            if apply_spatial else []
        )

        background_ops = [
            A.OneOf([
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1), num_shadows_lower=1,
                    num_shadows_upper=2, shadow_dimension=4, p=1.0,
                ),
                A.RandomFog(
                    fog_coef_lower=0.05, fog_coef_upper=0.2,
                    alpha_coef=0.1, p=1.0,
                ),
                A.RandomRain(
                    slant_lower=-5, slant_upper=5, drop_length=8,
                    drop_width=1, drop_color=(180, 180, 180),
                    blur_value=1, brightness_coefficient=0.9,
                    rain_type=None, p=1.0,
                ),
                # Scanlines: simulates poor scanner output
                A.GridDropout(
                    ratio=0.03, unit_size_min=1, unit_size_max=2,
                    holes_number_x=1, random_offset=False, p=1.0,
                ),
            ], p=0.35),
        ]

        color_ops = [
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.25, contrast_limit=0.25, p=1.0,
                ),
                A.RandomGamma(gamma_limit=(70, 130), p=1.0),   # illumination
                A.RGBShift(
                    r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0,
                ),
            ], p=0.4),
        ]

        degradation_ops = [
            A.OneOf([
                # PSF blur — simulates lens defocus
                A.Defocus(radius=(1, 3), alias_blur=0.1, p=1.0),
                # Vibration / motion blur
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                # Gaussian blur
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=1.0),
                # Morphological erosion / dilation (text thickness variation)
                A.Morphological(scale=(1, 3), operation="erosion", p=0.5),
                A.Morphological(scale=(1, 3), operation="dilation", p=0.5),
            ], p=0.35),
        ]

        all_ops = spatial_ops + background_ops + color_ops + degradation_ops

        self._compose = A.Compose(
            [A.SomeOf(all_ops, n=min(2, len(all_ops)), replace=False, p=self.p)],
            bbox_params=_BBOX_PARAMS,
        )

    def __call__(
        self, img: Any, target: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Optional[Dict[str, Any]]]:
        return _apply_albu(self._compose, img, target)


# ─── Compose ──────────────────────────────────────────────────────────────────

class Compose:
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, image: Any, target: Any) -> Tuple[Any, Any]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + "("]
        for t in self.transforms:
            lines.append(f"    {t}")
        lines.append(")")
        return "\n".join(lines)