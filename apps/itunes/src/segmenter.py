"""SAM-style "segment everything" via FastSAM, colorized per instance.

FastSAM (a CNN, YOLOv8-seg based) produces prompt-free instance masks for
everything in a frame in a single forward pass — the practical way to get the
Segment-Anything "segment everything" look on a Pi 5 CPU. We assign each mask a
distinct color and produce a fully color-blocked image.

Two display modes:

* **blocks** (default) — every pixel is a solid color (instance colors plus a
  solid background fill), no live video showing through. This is the clean
  "iTunes-ad" silhouette look and makes the background fill unambiguous.
* **overlay** — the colors are returned as a translucent layer the caller
  blends over the live frame.

``segment()`` returns an *overlay* — a color image plus a per-pixel alpha map —
rather than a finished frame, so one (slow) segmentation pass can be reused
across many (fast) live frames by the decoupled broadcaster in main.py. In
blocks mode the alpha map is all ones, so the blend collapses to just the color
image.

**Temporal color stability.** FastSAM re-runs from scratch each pass, so the
number of masks and their relative sizes jitter frame to frame. Coloring by
per-frame rank therefore makes every region flash a new color constantly. To
stop that, a lightweight tracker matches this frame's masks to recent ones by
centroid + area and carries each track's color forward; a track survives a few
missed frames (a grace period) so a momentary FastSAM dropout does not recolor
a region. Color thus follows a region's identity, not its per-frame rank.

**Temporal presence stability (anti-flicker).** Stable color identity is not
enough: FastSAM rebuilds its masks from scratch each pass, so a region it
momentarily misses flips to the background block and back (a purple<->white
pulse) and mask boundaries shimmer. A pixel-level persistence layer smooths the
*displayed color over time* — it EMA-blends toward the new color where a region
is present (so real motion still tracks) and *holds* a region's last color for a
few passes when it briefly drops out before fading to background. A single
``stability`` knob in [0, 1] scales this (0 = off/crisp, higher = more static);
it runs at mask resolution so it costs about a millisecond per pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# How many distinct instance colors to pre-generate before cycling.
_PALETTE_SIZE = 256

# --- Temporal tracker tuning (all distances/areas are frame-normalized) ---
# Max centroid distance (as a fraction of the frame) for a mask to match an
# existing track. Generous enough to survive mask-boundary jitter and modest
# motion between the ~1-10 FPS segmentation passes.
_MATCH_MAX_DIST = 0.18
_MATCH_MAX_DIST2 = _MATCH_MAX_DIST * _MATCH_MAX_DIST
# A match also requires the areas to be within this ratio, so a small mask does
# not steal a large track's color just because their centroids coincide.
_AREA_RATIO_LO, _AREA_RATIO_HI = 0.35, 2.8
# How many consecutive passes a track may go unmatched before it is dropped.
# Keeps a color pinned to a region through brief FastSAM dropouts.
_MAX_MISSED = 8

# --- Spatial color memory ---
# A track only survives a few missed passes; when a region disappears for
# longer and then comes back it would otherwise open a new track and flash a
# fresh color. A coarse grid remembers which color last occupied each cell of
# the frame for much longer, so a region that vanishes and reappears at roughly
# the same place gets its old color back. This is what stops the "colors jump
# as segments disappear then reappear" flicker beyond the track grace period.
_GRID = 24  # cells per axis over the normalized frame
# Passes a grid cell keeps its remembered color after nothing touches it. Long
# so colors persist across extended dropouts; expires eventually so a genuinely
# changed scene can recolor rather than being pinned forever.
_GRID_MAX_AGE = 150

# --- Pixel-level temporal persistence (anti-flicker) ---
# The tracker/grid stabilize a region's *color identity*, but the mask a region
# occupies is rebuilt from scratch every pass from whatever FastSAM returns, so
# a region FastSAM momentarily misses flips to the background block and back
# (purple<->white pulsing) and mask boundaries shimmer pass to pass. Persistence
# smooths the *displayed color per pixel* over time: it EMA-blends toward the new
# color where a region is present and *holds* a region's last color for a few
# passes when it briefly drops out, only then fading to background. This runs at
# mask resolution (~320x320) so it costs ~1 ms/pass. A single ``stability`` knob
# in [0, 1] scales it; these are the endpoints it interpolates between.
# Blend fraction toward the new color where an instance is present. 1.0 = snap
# (no smoothing, old behavior); lower = softer, more static edges.
_PERSIST_FG_BETA_MIN = 0.30  # at stability = 1 (most static)
# Passes to hold a dropped region's color before it starts fading to background.
_PERSIST_HOLD_MAX = 6  # at stability = 1
# Blend fraction toward background once a held region finally fades out.
_PERSIST_BG_BETA_MIN = 0.40  # at stability = 1


def _build_palette(n: int) -> np.ndarray:
    """`n` distinct **shades of purple** as BGR colors.

    All colors live in the blue-violet..magenta hue band so the whole scene
    reads as purples on the white background, but saturation and value are
    spread (via two decorrelated golden-ratio walks) so neighboring regions stay
    distinguishable — light lavender through deep violet. Hue/sat/value are
    kept away from the extremes so no shade washes out to near-white (which would
    vanish against the white background) or collapses to near-black.
    """
    palette = np.zeros((n, 3), dtype=np.uint8)
    golden = 0.618033988749895
    h = 0.0
    v = 0.35
    for i in range(n):
        h = (h + golden) % 1.0
        v = (v + golden * 1.37) % 1.0  # decorrelated from the hue walk
        hue = int(125 + h * 30)            # 125..155: blue-violet -> purple -> magenta
        val = int(150 + v * 95)            # 150..245: mid -> bright (never near-white)
        sat = int(255 - v * 110)           # 255..145: brighter shades a bit less saturated
        hsv = np.uint8([[[hue, sat, val]]])
        palette[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return palette


def _deletterbox_to(labels: np.ndarray, h: int, w: int) -> np.ndarray:
    """Map a label map at the model's (letterboxed) mask size back to the frame.

    FastSAM masks come back aspect-fit into a padded canvas at the model input
    size — e.g. a 1280x720 frame becomes a 320x320 mask with the real content
    in the middle 320x180 rows and gray padding above and below (the NCNN
    backend always returns a full square). Naively resizing the whole padded
    map to the frame squishes the content into a middle band and turns the
    padding into empty bars. Instead crop the padded region, then resize just
    the label map (one cheap nearest-neighbor pass) to the full frame.
    """
    mh, mw = labels.shape
    if (mh, mw) == (h, w):
        return labels
    r = min(mh / h, mw / w)  # scale used to fit the frame into the mask canvas
    uh, uw = round(h * r), round(w * r)  # unpadded content size within the mask
    top, left = (mh - uh) // 2, (mw - uw) // 2
    crop = labels[top : top + uh, left : left + uw]
    crop = np.ascontiguousarray(crop)
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)


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


def _centroids_areas(masks_bool: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame-normalized (cy, cx, area) for each boolean mask in ``(N, mh, mw)``.

    Vectorized: per-row and per-column pixel counts are dotted with the row/col
    indices to get first moments, divided by the area. Areas are returned as a
    fraction of the mask canvas so the tracker is resolution independent.
    """
    n, mh, mw = masks_bool.shape
    rows = np.arange(mh, dtype=np.float64)
    cols = np.arange(mw, dtype=np.float64)
    row_counts = masks_bool.sum(axis=2, dtype=np.float64)  # (N, mh)
    col_counts = masks_bool.sum(axis=1, dtype=np.float64)  # (N, mw)
    areas = row_counts.sum(axis=1)  # (N,)
    safe = np.where(areas > 0, areas, 1.0)
    cy = (row_counts @ rows) / safe / mh
    cx = (col_counts @ cols) / safe / mw
    area_norm = areas / float(mh * mw)
    return cy, cx, area_norm


class Segmenter:
    """FastSAM segment-everything wrapper producing colorized, color-stable output."""

    def __init__(
        self,
        model: str = "FastSAM-s",
        imgsz: int = 320,
        conf: float = 0.4,
        iou: float = 0.9,
        alpha: float = 0.5,
        retina_masks: bool = False,
        bg_color: tuple[int, int, int] = (255, 255, 255),
        bg_alpha: float = 0.85,
        overlay: bool = False,
        stability: float = 0.5,
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
        # filled block rather than see-through video (overlay mode only).
        self.bg_alpha = float(np.clip(bg_alpha, 0.0, 1.0))
        # blocks mode (default): render solid color, no live video underneath.
        self.overlay = bool(overlay)

        # Pixel-level temporal persistence (anti-flicker). ``stability`` in
        # [0, 1] interpolates the endpoints above: 0 disables it entirely (each
        # pass shown as-is, crisp but flickery); higher holds regions through
        # FastSAM dropouts and smooths boundaries, trading a little
        # responsiveness (a brief fading trail behind fast motion) for a much
        # more static picture. ``_sm_color`` is the accumulated color and
        # ``_sm_age`` the passes-since-present per pixel, both at mask resolution
        # and rebuilt if that resolution changes.
        self.stability = float(np.clip(stability, 0.0, 1.0))
        s = self.stability
        self._sm_fg_beta = 1.0 - (1.0 - _PERSIST_FG_BETA_MIN) * s
        self._sm_hold = int(round(_PERSIST_HOLD_MAX * s))
        self._sm_bg_beta = 1.0 - (1.0 - _PERSIST_BG_BETA_MIN) * s
        self._sm_color: np.ndarray | None = None  # (mh, mw, 3) float32
        self._sm_age: np.ndarray | None = None  # (mh, mw) int32

        # Temporal color tracker state. Each track: (cy, cx, area, color_id,
        # missed). color_id indexes the palette (cycled) and is what keeps a
        # region's color stable across passes.
        self._tracks: list[list[float]] = []
        self._next_color = 0

        # Spatial color memory: which color last occupied each frame cell, and
        # how many passes since it was touched (for expiry). -1 == empty. This
        # outlives individual tracks so a region that disappears and reappears
        # recovers its old color instead of flashing a new one.
        self._grid_color = np.full((_GRID, _GRID), -1, dtype=np.int64)
        self._grid_age = np.zeros((_GRID, _GRID), dtype=np.int64)

        # The torch (.pt) fallback path defaults to a single CPU thread here;
        # use all cores. Harmless for the NCNN path, which threads on its own.
        try:  # pragma: no cover - env-dependent
            import torch

            torch.set_num_threads(os.cpu_count() or 1)
        except Exception:
            pass

        self._model = FastSAM(_resolve_model(model))

    def _assign_colors(
        self, cy: np.ndarray, cx: np.ndarray, area: np.ndarray
    ) -> np.ndarray:
        """Return a stable palette color_id per current instance.

        Greedy nearest-centroid matching of this frame's instances to recent
        tracks (gated by centroid distance and area ratio); matched instances
        inherit the track's color, unmatched instances open a new track with a
        fresh color, and unmatched tracks age out after a grace period.
        """
        tracks = self._tracks
        n = len(cy)
        assigned = np.full(n, -1, dtype=np.int64)

        # Collect all admissible (distance, instance, track) pairs, then assign
        # greedily from the closest — a track and an instance are each used once.
        pairs: list[tuple[float, int, int]] = []
        for i in range(n):
            for j, t in enumerate(tracks):
                d2 = (cy[i] - t[0]) ** 2 + (cx[i] - t[1]) ** 2
                if d2 > _MATCH_MAX_DIST2:
                    continue
                ta = t[2]
                ratio = area[i] / ta if ta > 0 else float("inf")
                if ratio < _AREA_RATIO_LO or ratio > _AREA_RATIO_HI:
                    continue
                pairs.append((d2, i, j))
        pairs.sort(key=lambda p: p[0])

        used_tracks: set[int] = set()
        for _d2, i, j in pairs:
            if assigned[i] != -1 or j in used_tracks:
                continue
            assigned[i] = int(tracks[j][3])
            used_tracks.add(j)
            tracks[j][0], tracks[j][1], tracks[j][2] = float(cy[i]), float(cx[i]), float(area[i])
            tracks[j][4] = 0.0  # reset missed counter

        # Age the pre-existing tracks: matched ones survive (missed reset above),
        # unmatched ones increment and drop past the grace period. Done before
        # new tracks are added so a just-created track is never aged this pass.
        survivors = []
        for j, t in enumerate(tracks):
            if j in used_tracks:
                survivors.append(t)
            else:
                t[4] += 1.0
                if t[4] <= _MAX_MISSED:
                    survivors.append(t)

        # Open new tracks for unmatched instances. Prefer a color remembered for
        # this region's grid cell (so a reappearing region recovers its color);
        # only mint a fresh color when the cell has none. This is the key fix for
        # colors jumping as segments disappear and come back.
        for i in range(n):
            if assigned[i] == -1:
                gy, gx = self._grid_cell(cy[i], cx[i])
                remembered = int(self._grid_color[gy, gx])
                if remembered >= 0:
                    cid = remembered
                else:
                    cid = self._next_color % _PALETTE_SIZE
                    self._next_color += 1
                assigned[i] = cid
                survivors.append([float(cy[i]), float(cx[i]), float(area[i]), float(cid), 0.0])

        self._tracks = survivors
        self._update_grid(cy, cx, assigned)
        return assigned

    @staticmethod
    def _grid_cell(cy: float, cx: float) -> tuple[int, int]:
        """Clamp a normalized centroid to a ``(_GRID, _GRID)`` cell index."""
        gy = min(_GRID - 1, max(0, int(cy * _GRID)))
        gx = min(_GRID - 1, max(0, int(cx * _GRID)))
        return gy, gx

    def _update_grid(self, cy: np.ndarray, cx: np.ndarray, assigned: np.ndarray) -> None:
        """Stamp each instance's color into its grid cell and expire stale cells.

        Every cell ages one pass; a cell touched this pass resets to 0 and takes
        the color of the instance there, so the memory tracks the current scene.
        Cells untouched past the grace age are cleared so a changed scene can be
        recolored instead of being pinned to old colors forever.
        """
        self._grid_age += 1
        for i in range(len(cy)):
            gy, gx = self._grid_cell(cy[i], cx[i])
            self._grid_color[gy, gx] = int(assigned[i])
            self._grid_age[gy, gx] = 0
        expired = self._grid_age > _GRID_MAX_AGE
        self._grid_color[expired] = -1

    def _persist(self, color_small: np.ndarray, fg: np.ndarray) -> np.ndarray:
        """Temporally smooth the per-pixel color to kill drop-in/out flicker.

        ``color_small`` is this pass's color image and ``fg`` a boolean mask of
        which pixels an instance covers, both at mask resolution. Where a region
        is present the accumulated color EMA-blends toward the new color (fast,
        so real motion still tracks); where a region just dropped out the old
        color is *held* for ``_sm_hold`` passes before fading to background, so a
        momentary FastSAM miss no longer flashes the block to white. Returns the
        smoothed color at mask resolution. With ``stability == 0`` this is a
        no-op passthrough (crisp but flickery, the original behavior).
        """
        if self.stability <= 0.0:
            return color_small

        c = color_small.astype(np.float32)
        if (
            self._sm_color is None
            or self._sm_color.shape != c.shape
            or self._sm_age is None
        ):
            # First pass (or resolution changed): seed the accumulator. Pixels
            # already background start past the hold so they are not held.
            self._sm_color = c.copy()
            self._sm_age = np.where(fg, 0, self._sm_hold + 1).astype(np.int32)
            return color_small

        sm = self._sm_color
        age = self._sm_age

        # Present pixels: blend toward the new color, reset their idle age.
        sm[fg] += self._sm_fg_beta * (c[fg] - sm[fg])
        age[fg] = 0

        # Absent pixels: age them. Within the hold window keep the old color
        # (the region is treated as a brief dropout); past it, fade to bg.
        bg = ~fg
        age[bg] += 1
        fade = bg & (age > self._sm_hold)
        if fade.any():
            bg_col = self.bg_color.astype(np.float32)
            sm[fade] += self._sm_bg_beta * (bg_col - sm[fade])

        self._sm_color = sm
        self._sm_age = age
        return np.clip(sm, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _deletterbox_color(img: np.ndarray, h: int, w: int) -> np.ndarray:
        """De-letterbox a mask-resolution color image back to the frame size.

        Mirrors :func:`_deletterbox_to` but for a 3-channel color image resized
        with INTER_LINEAR (the persisted colors are already soft, so a linear
        resize keeps stable, smooth block edges).
        """
        mh, mw = img.shape[:2]
        if (mh, mw) == (h, w):
            return img
        r = min(mh / h, mw / w)
        uh, uw = round(h * r), round(w * r)
        top, left = (mh - uh) // 2, (mw - uw) // 2
        crop = np.ascontiguousarray(img[top : top + uh, left : left + uw])
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def segment(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run FastSAM and return ``(color_bgr, alpha_map)`` at the frame's size.

        Every pixel is colored: instance masks get temporally-stable palette
        colors and all remaining (background) pixels get ``self.bg_color``. In
        blocks mode the alpha map is all ones (solid color, no video); in
        overlay mode background is blended at ``self.bg_alpha`` and instances at
        ``self.alpha``.

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
            # No instances this pass. Keep the label map at the persistence
            # accumulator's resolution (so held regions can still fade out) if we
            # have one; otherwise the frame is one solid background block.
            if self.stability > 0.0 and self._sm_color is not None:
                mh, mw = self._sm_color.shape[:2]
            else:
                mh, mw = h, w
            labels = np.zeros((mh, mw), dtype=np.int32)
            color_ids = np.empty(0, dtype=np.int64)
            n = 0
        else:
            masks = masks.data  # (N, mh, mw) tensor, values in [0, 1]
            masks = np.asarray(masks.cpu().numpy() if hasattr(masks, "cpu") else masks)
            masks_bool = masks > 0.5
            n, mh, mw = masks_bool.shape

            # Stable per-instance colors from the temporal tracker.
            cy, cx, area = _centroids_areas(masks_bool)
            color_ids = self._assign_colors(cy, cx, area)

            # Build a label map at mask resolution. Paint largest masks first so
            # smaller objects land on top (higher label) and stay visible. The
            # label is only a draw slot; its *color* comes from color_ids, so
            # occlusion order no longer perturbs which color a region shows.
            order = np.argsort(area)[::-1]
            labels = np.zeros((mh, mw), dtype=np.int32)
            slot_color = np.empty(n, dtype=np.int64)  # color_id per draw slot
            for draw_idx, i in enumerate(order):
                labels[masks_bool[i]] = draw_idx + 1
                slot_color[draw_idx] = color_ids[i]
            color_ids = slot_color

        # Map labels -> color and opacity in single vectorized passes. Row 0 is
        # the background; row k is the tracked color of the k-th painted slot.
        color_lut = np.empty((n + 1, 3), dtype=np.uint8)
        color_lut[0] = self.bg_color
        if n:
            color_lut[1:] = self.palette[color_ids % _PALETTE_SIZE]

        if self.overlay:
            alpha_lut = np.full(n + 1, self.alpha, dtype=np.float32)
            alpha_lut[0] = self.bg_alpha
        else:
            # blocks mode: solid everywhere, no live video underneath.
            alpha_lut = np.ones(n + 1, dtype=np.float32)

        if self.stability > 0.0:
            # Anti-flicker: color at mask resolution, temporally smooth (hold
            # dropouts, ease boundaries), then de-letterbox the smoothed color to
            # the frame. Alpha follows the crisp current labels (all ones in
            # blocks mode, so the held colors show at full strength).
            color_small = color_lut[labels]
            fg = labels > 0
            color_small = self._persist(color_small, fg)
            color = self._deletterbox_color(color_small, h, w)
            alpha = alpha_lut[_deletterbox_to(labels, h, w)]
        else:
            # Original crisp path: de-letterbox labels, then map to color/alpha.
            labels = _deletterbox_to(labels, h, w)
            color = color_lut[labels]
            alpha = alpha_lut[labels]
        return color, alpha

    def blend(
        self, frame_bgr: np.ndarray, color_bgr: np.ndarray, alpha_map: np.ndarray
    ) -> np.ndarray:
        """Composite an overlay from :meth:`segment` onto a (current) frame.

        In blocks mode the alpha map is all ones, so this returns just the color
        image (video does not show through).
        """
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
