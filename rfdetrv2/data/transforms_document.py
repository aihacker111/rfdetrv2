# ------------------------------------------------------------------------
# Document parsing augmentations (Table 2 style) — PIL + NumPy + SciPy only.
# Same calling convention as rfdetrv2.data.transforms: __call__(img, target).
# ------------------------------------------------------------------------
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageFilter
import torch

from scipy import ndimage

from rfdetrv2.data.transforms import resize as _resize_fn
from rfdetrv2.data.transforms import _align_hw_to_multiple


def _pil_to_float01(img: PIL.Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _float01_to_pil(arr: np.ndarray) -> PIL.Image.Image:
    x = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return PIL.Image.fromarray(x, mode="RGB")


def _boxes_to_numpy(target: Dict[str, Any]) -> np.ndarray:
    b = target.get("boxes")
    if b is None or b.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return b.detach().cpu().numpy().astype(np.float32)


def _set_boxes(target: Dict[str, Any], boxes_np: np.ndarray, w: int, h: int) -> None:
    if boxes_np.size == 0:
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        if "labels" in target:
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
        return
    boxes_np = boxes_np.reshape(-1, 4)
    boxes_np[:, 0::2] = np.clip(boxes_np[:, 0::2], 0, w)
    boxes_np[:, 1::2] = np.clip(boxes_np[:, 1::2], 0, h)
    keep = (boxes_np[:, 2] > boxes_np[:, 0]) & (boxes_np[:, 3] > boxes_np[:, 1])
    boxes_np = boxes_np[keep]
    target["boxes"] = torch.from_numpy(boxes_np.astype(np.float32))
    if "labels" in target:
        if target["labels"].numel() == len(keep):
            target["labels"] = target["labels"][torch.as_tensor(keep, dtype=torch.bool)]
        elif target["boxes"].numel() == 0:
            target["labels"] = torch.zeros((0,), dtype=torch.int64)


def _rotate_xyxy_boxes(boxes: np.ndarray, w: int, h: int, angle_deg: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    cx, cy = w / 2.0, h / 2.0
    out = []
    for x1, y1, x2, y2 in boxes:
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        r = []
        for x, y in corners:
            dx, dy = x - cx, y - cy
            xr = c * dx - s * dy + cx
            yr = s * dx + c * dy + cy
            r.append([xr, yr])
        r = np.array(r, dtype=np.float32)
        out.append(
            [
                float(r[:, 0].min()),
                float(r[:, 1].min()),
                float(r[:, 0].max()),
                float(r[:, 1].max()),
            ]
        )
    return np.array(out, dtype=np.float32)


def _warp_xyxy_with_displacement(
    boxes: np.ndarray, dx: np.ndarray, dy: np.ndarray, w: int, h: int
) -> np.ndarray:
    if boxes.size == 0:
        return boxes

    def sample_disp(x: float, y: float) -> Tuple[float, float]:
        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))
        return float(dx[yi, xi]), float(dy[yi, xi])

    out = []
    for x1, y1, x2, y2 in boxes:
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        moved = []
        for x, y in corners:
            u, v = sample_disp(x, y)
            moved.append([x + u, y + v])
        moved = np.array(moved, dtype=np.float32)
        out.append(
            [
                float(moved[:, 0].min()),
                float(moved[:, 1].min()),
                float(moved[:, 0].max()),
                float(moved[:, 1].max()),
            ]
        )
    return np.array(out, dtype=np.float32)


def _grid_distortion_maps(h: int, w: int, steps: int = 4, strength: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    sh, sw = steps + 1, steps + 1
    rx = (np.random.rand(sh, sw) * 2 - 1).astype(np.float32) * strength
    ry = (np.random.rand(sh, sw) * 2 - 1).astype(np.float32) * strength
    zx = (w - 1) / max(sw - 1, 1)
    zy = (h - 1) / max(sh - 1, 1)
    dx = ndimage.zoom(rx, (h / sh, w / sw), order=1)[:h, :w]
    dy = ndimage.zoom(ry, (h / sh, w / sw), order=1)[:h, :w]
    if dx.shape != (h, w):
        dx = np.array(PIL.Image.fromarray(dx).resize((w, h), PIL.Image.BILINEAR))
        dy = np.array(PIL.Image.fromarray(dy).resize((w, h), PIL.Image.BILINEAR))
    return dx.astype(np.float32), dy.astype(np.float32)


def _apply_inverse_grid_distortion(rgb: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Sample input at (x + dx, y + dy) for each output pixel (inverse warp)."""
    h, w = rgb.shape[:2]
    ys, xs = np.indices((h, w), dtype=np.float32)
    xmap = np.clip(xs + dx, 0, w - 1)
    ymap = np.clip(ys + dy, 0, h - 1)
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = ndimage.map_coordinates(
            rgb[:, :, c].astype(np.float32),
            [ymap, xmap],
            order=1,
            mode="reflect",
        )
    return out


def _disk_kernel(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    s = 2 * r + 1
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    mask = yy * yy + xx * xx <= r * r
    k = mask.astype(np.float32)
    k /= k.sum() + 1e-8
    return k


def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), dtype=np.float32)
    t = math.radians(angle_deg)
    r = length // 2
    for u in range(-r, r + 1):
        x = int(round(r + u * math.cos(t)))
        y = int(round(r + u * math.sin(t)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    s = float(k.sum())
    if s < 1e-6:
        k[r, :] = 1.0
        s = float(k.sum())
    k /= s
    return k


def _conv_rgb(rgb: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = ndimage.convolve(
            rgb[:, :, c], kernel, mode="nearest"
        )
    return np.clip(out, 0, 1)


def _finalize_patch_and_target(
    img: PIL.Image.Image,
    target: Dict[str, Any],
    patch_size: int,
) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
    w, h = img.size
    ha, wa = _align_hw_to_multiple((h, w), patch_size)
    target = target.copy()
    if ha == h and wa == w:
        target["size"] = torch.tensor([h, w])
        if "area" in target and target.get("boxes") is not None and target["boxes"].numel() > 0:
            b = target["boxes"]
            target["area"] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return img, target

    arr = _pil_to_float01(img)
    boxes = _boxes_to_numpy(target)
    if ha <= h and wa <= w:
        y0 = (h - ha) // 2
        x0 = (w - wa) // 2
        arr = arr[y0 : y0 + ha, x0 : x0 + wa]
        if boxes.size:
            boxes = boxes.copy()
            boxes[:, [0, 2]] -= x0
            boxes[:, [1, 3]] -= y0
    else:
        pad_h, pad_w = ha - h, wa - w
        arr = np.pad(
            arr,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=1.0,
        )
    img = _float01_to_pil(arr)
    w2, h2 = img.size
    _set_boxes(target, boxes, w2, h2)
    target["size"] = torch.tensor([h2, w2])
    if "area" in target and target.get("boxes") is not None and target["boxes"].numel() > 0:
        b = target["boxes"]
        target["area"] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return img, target


# ----- Table 2: spatial (bbox-aware) -----


class RandomDocumentExtraScale(object):
    """Mild rescale around current size (Table: scaling)."""

    def __init__(self, scale_limit: Tuple[float, float] = (0.94, 1.06), p: float = 1.0) -> None:
        self.lo, self.hi = scale_limit
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        w, h = img.size
        f = random.uniform(self.lo, self.hi)
        nw, nh = max(16, int(round(w * f))), max(16, int(round(h * f)))
        return _resize_fn(img, target, (nw, nh), max_size=None, size_divisor=None)


class RandomDocumentGridDistortion(object):
    """Grid distortion; updates boxes via displacement sampling at corners."""

    def __init__(self, steps: int = 4, strength: float = 3.5, p: float = 1.0) -> None:
        self.steps = steps
        self.strength = strength
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        w, h = img.size
        rgb = _pil_to_float01(img)
        dx, dy = _grid_distortion_maps(h, w, self.steps, self.strength)
        rgb = _apply_inverse_grid_distortion(rgb, dx, dy)
        boxes = _warp_xyxy_with_displacement(_boxes_to_numpy(target), dx, dy, w, h)
        target = target.copy()
        _set_boxes(target, boxes, w, h)
        return _float01_to_pil(rgb), target


class RandomDocumentRotation(object):
    """Small in-plane rotation; expand=False, white fill."""

    def __init__(self, degrees: float = 8.0, p: float = 1.0) -> None:
        self.deg = float(degrees)
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        angle = random.uniform(-self.deg, self.deg)
        w, h = img.size
        boxes = _rotate_xyxy_boxes(_boxes_to_numpy(target), w, h, angle)
        img = img.convert("RGB").rotate(
            angle, resample=PIL.Image.BILINEAR, expand=False, fillcolor=(255, 255, 255)
        )
        target = target.copy()
        _set_boxes(target, boxes, w, h)
        return img, target


# ----- Table 2: background (image-only) -----


class RandomDocumentTexture(object):
    """High-frequency texture / grain."""

    def __init__(self, sigma: float = 0.02, p: float = 1.0) -> None:
        self.sigma = sigma
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        x = _pil_to_float01(img)
        noise = np.random.normal(0, self.sigma, x.shape).astype(np.float32)
        return _float01_to_pil(x + noise), target


class RandomDocumentWeatherFog(object):
    """Low-frequency attenuation (weather / haze)."""

    def __init__(self, strength: Tuple[float, float] = (0.06, 0.18), p: float = 1.0) -> None:
        self.strength = strength
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        x = _pil_to_float01(img)
        h, w = x.shape[:2]
        g = ndimage.gaussian_filter(np.random.rand(h, w).astype(np.float32), sigma=max(h, w) / 12)
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        a = random.uniform(*self.strength)
        x = x * (1.0 - a * g[..., None])
        return _float01_to_pil(x), target


class RandomDocumentBackgroundTint(object):
    """Uneven background / illumination on paper."""

    def __init__(self, delta: float = 0.08, p: float = 1.0) -> None:
        self.delta = delta
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        x = _pil_to_float01(img)
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * random.uniform(0.25, 0.75), h * random.uniform(0.25, 0.75)
        r = max(w, h) * random.uniform(0.35, 0.65)
        field = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        tint = self.delta * (random.random() - 0.5) * 2 * field[..., None]
        return _float01_to_pil(np.clip(x + tint, 0, 1)), target


class RandomDocumentWatermark(object):
    """Light diagonal strokes."""

    def __init__(self, alpha: float = 0.12, p: float = 1.0) -> None:
        self.alpha = alpha
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        x = _pil_to_float01(img)
        h, w = x.shape[:2]
        lines = PIL.Image.new("L", (w, h), 0)
        draw = PIL.ImageDraw.Draw(lines)
        for _ in range(random.randint(2, 6)):
            x0, y0 = random.randint(0, w - 1), random.randint(0, h - 1)
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            draw.line([(x0, y0), (x1, y1)], fill=255, width=random.randint(1, 2))
        m = ndimage.gaussian_filter(np.asarray(lines, dtype=np.float32) / 255.0, sigma=0.8)
        a = self.alpha * random.random()
        x = np.clip(x * (1.0 - a * m[..., None]) + a * 0.85 * m[..., None], 0, 1)
        return _float01_to_pil(x), target


class RandomDocumentScanlines(object):
    def __init__(self, strength: float = 0.1, p: float = 1.0) -> None:
        self.strength = strength
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        x = _pil_to_float01(img)
        h, w = x.shape[:2]
        step = max(2, h // 200)
        m = np.ones((h, w, 1), dtype=np.float32)
        for y in range(0, h, step):
            m[y : y + 1, :, :] *= 1.0 - self.strength * random.random()
        return _float01_to_pil(x * m), target


class RandomDocumentShadow(object):
    """Soft dark region (scanner shadow)."""

    def __init__(self, max_darken: float = 0.35, p: float = 1.0) -> None:
        self.max_darken = max_darken
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        x = _pil_to_float01(img)
        h, w = x.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        pil_m = PIL.Image.fromarray((mask * 255).astype(np.uint8))
        dr = PIL.ImageDraw.Draw(pil_m)
        x0, y0 = random.randint(0, w // 3), random.randint(0, h // 3)
        x1, y1 = w - random.randint(0, w // 4), h - random.randint(0, h // 4)
        dr.ellipse([x0, y0, x1, y1], fill=255)
        m = ndimage.gaussian_filter(np.asarray(pil_m, dtype=np.float32) / 255.0, sigma=max(h, w) / 25)
        d = self.max_darken * random.random()
        x = x * (1.0 - d * m[..., None])
        return _float01_to_pil(x), target


# ----- Table 2: color -----


class RandomDocumentBrightnessContrast(object):
    def __init__(
        self,
        brightness: Tuple[float, float] = (0.85, 1.15),
        contrast: Tuple[float, float] = (0.85, 1.2),
        p: float = 1.0,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        img = img.convert("RGB")
        b = random.uniform(*self.brightness)
        c = random.uniform(*self.contrast)
        if b != 1.0:
            img = PIL.ImageEnhance.Brightness(img).enhance(b)
        if c != 1.0:
            img = PIL.ImageEnhance.Contrast(img).enhance(c)
        return img, target


class RandomDocumentGamma(object):
    """Global illumination (gamma)."""

    def __init__(self, gamma: Tuple[float, float] = (0.85, 1.2), p: float = 1.0) -> None:
        self.gamma = gamma
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        g = random.uniform(*self.gamma)
        x = _pil_to_float01(img)
        x = np.clip(x ** g, 0, 1)
        return _float01_to_pil(x), target


class RandomDocumentRGBShift(object):
    def __init__(self, shift: int = 18, p: float = 1.0) -> None:
        self.shift = shift
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        x = np.asarray(img.convert("RGB"), dtype=np.int16)
        x[:, :, 0] += random.randint(-self.shift, self.shift)
        x[:, :, 1] += random.randint(-self.shift, self.shift)
        x[:, :, 2] += random.randint(-self.shift, self.shift)
        x = np.clip(x, 0, 255).astype(np.uint8)
        return PIL.Image.fromarray(x, mode="RGB"), target


# ----- Table 2: degradation -----


class RandomDocumentGaussianBlur(object):
    def __init__(self, radius: Tuple[float, float] = (0.4, 1.2), p: float = 1.0) -> None:
        self.radius = radius
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        r = random.uniform(*self.radius)
        return img.filter(PIL.ImageFilter.GaussianBlur(radius=r)), target


class RandomDocumentPSFBlur(object):
    """Disk-shaped PSF (out-of-focus)."""

    def __init__(self, radius: Tuple[int, int] = (1, 3), p: float = 1.0) -> None:
        self.radius = radius
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        r = random.randint(self.radius[0], self.radius[1])
        k = _disk_kernel(r)
        x = _pil_to_float01(img)
        return _float01_to_pil(_conv_rgb(x, k)), target


class RandomDocumentMotionBlur(object):
    """Vibration / motion blur."""

    def __init__(self, length: Tuple[int, int] = (5, 11), p: float = 1.0) -> None:
        self.length = length
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        L = random.randint(self.length[0] // 2 * 2 + 1, self.length[1] // 2 * 2 + 1)
        ang = random.uniform(0, 180)
        k = _motion_kernel(L, ang)
        x = _pil_to_float01(img)
        return _float01_to_pil(_conv_rgb(x, k)), target


class RandomDocumentMorphology(object):
    """Random grey-scale erosion or dilation (document stroke thickness)."""

    def __init__(self, size: int = 2, p: float = 1.0) -> None:
        self.size = size
        self.p = p

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        x = _pil_to_float01(img)
        gray = x.mean(axis=2)
        s = max(2, int(self.size))
        foot = np.ones((s, s), dtype=bool)
        if random.random() < 0.5:
            gray = ndimage.grey_erosion(gray, footprint=foot)
        else:
            gray = ndimage.grey_dilation(gray, footprint=foot)
        # blend toward morph result on luminance
        g = gray[..., None]
        y = x * 0.65 + g * 0.35
        return _float01_to_pil(np.clip(y, 0, 1)), target


class DocumentTable2Augment(object):
    """Runs random picks from Table 2 groups, then aligns H/W to *patch_size*."""

    def __init__(
        self,
        p: float = 0.5,
        patch_size: int = 16,
        spatial_p: float = 0.4,
        background_p: float = 0.45,
        color_p: float = 0.55,
        degradation_p: float = 0.45,
    ) -> None:
        self.p = float(p)
        self.patch_size = int(patch_size)
        self.spatial_p = float(spatial_p)
        self.background_p = float(background_p)
        self.color_p = float(color_p)
        self.degradation_p = float(degradation_p)
        self._spatial: List[Any] = [
            RandomDocumentExtraScale(),
            RandomDocumentGridDistortion(),
            RandomDocumentRotation(),
        ]
        self._background: List[Any] = [
            RandomDocumentTexture(),
            RandomDocumentWeatherFog(),
            RandomDocumentBackgroundTint(),
            RandomDocumentWatermark(),
            RandomDocumentScanlines(),
            RandomDocumentShadow(),
        ]
        self._color: List[Any] = [
            RandomDocumentBrightnessContrast(),
            RandomDocumentGamma(),
            RandomDocumentRGBShift(),
        ]
        self._degradation: List[Any] = [
            RandomDocumentPSFBlur(),
            RandomDocumentMotionBlur(),
            RandomDocumentGaussianBlur(),
            RandomDocumentMorphology(),
        ]

    def __call__(
        self, img: PIL.Image.Image, target: Dict[str, Any]
    ) -> Tuple[PIL.Image.Image, Dict[str, Any]]:
        if random.random() > self.p:
            return img, target
        target = target.copy()
        if random.random() < self.spatial_p:
            img, target = random.choice(self._spatial)(img, target)
        if random.random() < self.background_p:
            img, target = random.choice(self._background)(img, target)
        if random.random() < self.color_p:
            img, target = random.choice(self._color)(img, target)
        if random.random() < self.degradation_p:
            img, target = random.choice(self._degradation)(img, target)
        if "masks" in target and target["masks"] is not None and target["masks"].numel() > 0:
            w, h = img.size
            target["masks"] = torch.zeros((0, h, w), dtype=torch.bool)
        return _finalize_patch_and_target(img, target, self.patch_size)
