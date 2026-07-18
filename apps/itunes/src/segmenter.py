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
        imgsz: int = 512,
        conf: float = 0.4,
        iou: float = 0.9,
        alpha: float = 0.5,
        retina_masks: bool = False,
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

        self._model = FastSAM(_resolve_model(model))

    def segment(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run FastSAM and return ``(color_bgr, alpha_map)`` at the frame's size.

        ``color_bgr`` is a (H, W, 3) uint8 image of per-instance colors and
        ``alpha_map`` is a (H, W) float32 opacity map (0 where nothing is
        segmented, ``self.alpha`` under a mask). Blend with :meth:`blend`.
        """
        h, w = frame_bgr.shape[:2]
        color = np.zeros((h, w, 3), dtype=np.uint8)
        alpha = np.zeros((h, w), dtype=np.float32)

        results = self._model(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            retina_masks=self.retina_masks,
            device="cpu",
            verbose=False,
        )
        if not results or results[0].masks is None:
            return color, alpha

        masks = results[0].masks.data  # (N, mh, mw) tensor, values in [0, 1]
        masks = np.asarray(masks.cpu().numpy() if hasattr(masks, "cpu") else masks)

        # Paint largest masks first so smaller objects land on top and stay visible.
        order = np.argsort([float(m.sum()) for m in masks])[::-1]
        for draw_idx, i in enumerate(order):
            m = masks[i]
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            region = m > 0.5
            if not region.any():
                continue
            color[region] = self.palette[draw_idx % _PALETTE_SIZE]
            alpha[region] = self.alpha

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
