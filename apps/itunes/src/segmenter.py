"""Semantic segmentation using MediaPipe's Image Segmenter with DeepLabV3.

DeepLabV3 outputs a per-pixel category mask over ~21 Pascal-VOC classes
(background, person, car, cat, chair, ...). We colorize the category mask and
alpha-blend it over the source frame.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Pascal-VOC 21-class labels, in DeepLabV3 index order.
LABELS = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tv",
)

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "deeplab_v3.tflite"

# Native input resolution of the DeepLabV3 model. Segmenting at this size and
# upscaling the mask keeps CPU latency reasonable on the Pi.
_MODEL_INPUT = 257


def _build_palette(n: int) -> np.ndarray:
    """Deterministic, visually distinct BGR colors; index 0 (background) is black."""
    palette = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        hue = int(179 * (i / max(n - 1, 1)))
        hsv = np.uint8([[[hue, 200, 255]]])
        palette[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return palette


class Segmenter:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL, alpha: float = 0.5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run scripts/download_model.sh first."
            )
        self.alpha = float(alpha)
        self.palette = _build_palette(len(LABELS))

        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def category_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a uint8 category-index mask at the frame's resolution."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._segmenter.segment(mp_image)
        mask = result.category_mask.numpy_view()  # uint8, model resolution
        h, w = frame_bgr.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    def overlay(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Blend a colorized segmentation mask over the frame."""
        mask = self.category_mask(frame_bgr)
        color_mask = self.palette[mask]  # (H, W, 3)
        blended = cv2.addWeighted(frame_bgr, 1.0 - self.alpha, color_mask, self.alpha, 0)

        # Leave background pixels (class 0) as the original image.
        bg = mask == 0
        blended[bg] = frame_bgr[bg]
        return blended

    def present_labels(self, mask: np.ndarray) -> list[str]:
        return [LABELS[i] for i in np.unique(mask) if i < len(LABELS)]

    def close(self) -> None:
        self._segmenter.close()

    def __enter__(self) -> "Segmenter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
