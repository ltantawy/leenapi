"""SAM-style "segment everything" via FastSAM, colorized per instance.

FastSAM (a CNN, YOLOv8-seg based) produces prompt-free instance masks for
everything in a frame in a single forward pass — the practical way to get the
Segment-Anything "segment everything" look on a Pi 5 CPU. We assign each mask a
distinct color and composite them into a translucent overlay that the caller
blends over the live frame.

Output of ``segment()`` is an *overlay* — a color image plus a per-pixel alpha
map — rather than a finished frame, so a single (slow) segmentation pass can be
reused across many (fast) live frames by the decoupled broadcaster in main.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# How many distinct instance colors to pre-generate before cycling.
_PALETTE_SIZE = 256


def _build_palette(n: int) -> np.ndarray:
    """`n` visually distinct, saturated BGR colors via golden-ratio hue spacing."""
    palette = np.zeros((n, 3), dtype=np.uint8)
    golden = 0.618033988749895
    hue = 0.1
    for i in range(n):
        hue = (hue + golden) % 1.0
        hsv = np.uint8([[[int(hue * 179), 200, 255]]])
        palette[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return palette


def _resolve_model(model: str) -> str:
    """Resolve a model name to a usable path, preferring a fast local NCNN export.

    Order: an explicit existing path → ``models/<name>_ncnn_model`` (fastest on
    the Pi CPU) → ``models/<name>.pt`` → the bare ``<name>.pt`` (ultralytics
    auto-downloads it on first use).
    """
    p = Path(model)
    if p.exists():
        return str(p)

    stem = p.stem if p.suffix == ".pt" else p.name  # "FastSAM-s" from either form
    ncnn_dir = _MODELS_DIR / f"{stem}_ncnn_model"
    if ncnn_dir.is_dir():
        return str(ncnn_dir)
    local_pt = _MODELS_DIR / f"{stem}.pt"
    if local_pt.is_file():
        return str(local_pt)
    return f"{stem}.pt"


class Segmenter:
    """FastSAM segment-everything wrapper producing colorized overlays."""

    def __init__(
        self,
        model: str = "FastSAM-s",
        imgsz: int = 448,
        conf: float = 0.4,
        iou: float = 0.9,
        alpha: float = 0.5,
        retina_masks: bool = False,
        bg_color: tuple[int, int, int] = (40, 40, 40),
        bg_alpha: float = 0.85,
    ):
        try:
            from ultralytics import FastSAM
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "ultralytics is required for FastSAM. Install deps with `uv sync`."
            ) from exc

        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.retina_masks = bool(retina_masks)
        self.palette = _build_palette(_PALETTE_SIZE)
        # Color for pixels no instance mask covers, so the whole frame is blocked.
        self.bg_color = np.array(bg_color, dtype=np.uint8)
        # Background is drawn more opaque than instances so it reads as a solid
        # filled block rather than see-through video.
        self.bg_alpha = float(np.clip(bg_alpha, 0.0, 1.0))

        # The torch (.pt) fallback path defaults to a single CPU thread here;
        # use all cores. Harmless for the NCNN path, which threads on its own.
        try:  # pragma: no cover - env-dependent
            import torch

            torch.set_num_threads(os.cpu_count() or 1)
        except Exception:
            pass

        self._model = FastSAM(_resolve_model(model))

    def segment(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run FastSAM and return ``(color_bgr, alpha_map)`` at the frame's size.

        Every pixel is colored: instance masks get distinct palette colors and
        all remaining (background) pixels get ``self.bg_color``. Background is
        blended at ``self.bg_alpha`` (near-solid) and instances at
        ``self.alpha``, so the frame is fully color-blocked with the background
        reading as a solid fill rather than see-through video.

        Colorization is done via a small integer label map at mask resolution
        (0 = background, k = the k-th painted instance), resized to the frame
        size in a single pass and mapped to colors/opacity through lookup
        tables — far cheaper than resizing and indexing each mask at full
        resolution.
        """
        h, w = frame_bgr.shape[:2]

        results = self._model(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            retina_masks=self.retina_masks,
            device="cpu",
            verbose=False,
        )

        masks = None if not results else results[0].masks
        if masks is None:
            # No instances: the whole frame is one solid background block.
            labels = np.zeros((h, w), dtype=np.int32)
            n = 0
        else:
            masks = masks.data  # (N, mh, mw) tensor, values in [0, 1]
            masks = np.asarray(masks.cpu().numpy() if hasattr(masks, "cpu") else masks)
            n, mh, mw = masks.shape
            # Build a label map at mask resolution. Paint largest masks first
            # so smaller objects land on top (higher label) and stay visible.
            order = np.argsort(masks.reshape(n, -1).sum(axis=1))[::-1]
            labels = np.zeros((mh, mw), dtype=np.int32)
            for draw_idx, i in enumerate(order):
                labels[masks[i] > 0.5] = draw_idx + 1
            # One nearest-neighbor resize of the label map to the frame size.
            if (mh, mw) != (h, w):
                labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

        # Map labels -> color and opacity in single vectorized passes. Row 0 is
        # the background; row k is the (k-1)-th palette color (cycled).
        color_lut = np.empty((n + 1, 3), dtype=np.uint8)
        color_lut[0] = self.bg_color
        color_lut[1:] = self.palette[np.arange(n) % _PALETTE_SIZE]

        alpha_lut = np.full(n + 1, self.alpha, dtype=np.float32)
        alpha_lut[0] = self.bg_alpha

        color = color_lut[labels]
        alpha = alpha_lut[labels]
        return color, alpha

    def blend(
        self, frame_bgr: np.ndarray, color_bgr: np.ndarray, alpha_map: np.ndarray
    ) -> np.ndarray:
        """Composite an overlay from :meth:`segment` onto a (current) frame."""
        if color_bgr.shape[:2] != frame_bgr.shape[:2]:
            h, w = frame_bgr.shape[:2]
            color_bgr = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
            alpha_map = cv2.resize(alpha_map, (w, h), interpolation=cv2.INTER_NEAREST)
        a = alpha_map[:, :, None]
        out = frame_bgr.astype(np.float32) * (1.0 - a) + color_bgr.astype(np.float32) * a
        return out.astype(np.uint8)

    def close(self) -> None:
        pass

    def __enter__(self) -> "Segmenter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
