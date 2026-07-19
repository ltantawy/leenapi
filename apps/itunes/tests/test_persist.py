"""Unit tests for the pixel-level temporal persistence (anti-flicker) layer.

These target the root cause of the remaining flicker: FastSAM rebuilds masks
each pass, so a region it momentarily misses used to flash to the background
block and back. ``Segmenter._persist`` holds a region's color through such
dropouts and eases boundaries. The tests exercise it directly, without loading
the FastSAM model, by building a bare instance and wiring the same knobs
``__init__`` derives from ``stability``.
"""

import numpy as np

from src.segmenter import (
    Segmenter,
    _PERSIST_BG_BETA_MIN,
    _PERSIST_FG_BETA_MIN,
    _PERSIST_HOLD_MAX,
)


def _make(stability: float, bg_color=(255, 255, 255)) -> Segmenter:
    """A Segmenter with only the persistence state set (no model load)."""
    seg = Segmenter.__new__(Segmenter)
    s = float(np.clip(stability, 0.0, 1.0))
    seg.stability = s
    seg._sm_fg_beta = 1.0 - (1.0 - _PERSIST_FG_BETA_MIN) * s
    seg._sm_hold = int(round(_PERSIST_HOLD_MAX * s))
    seg._sm_bg_beta = 1.0 - (1.0 - _PERSIST_BG_BETA_MIN) * s
    seg._sm_color = None
    seg._sm_age = None
    seg.bg_color = np.array(bg_color, dtype=np.uint8)
    return seg


PURPLE = (200, 40, 160)


def _tile(color, shape=(4, 4)):
    img = np.empty((*shape, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_stability_zero_is_passthrough():
    seg = _make(0.0)
    color = _tile(PURPLE)
    fg = np.ones((4, 4), dtype=bool)
    out = seg._persist(color, fg)
    assert np.array_equal(out, color)
    # Disabled means no accumulator state is created.
    assert seg._sm_color is None


def test_first_pass_seeds_and_returns_input():
    seg = _make(0.5)
    color = _tile(PURPLE)
    fg = np.ones((4, 4), dtype=bool)
    out = seg._persist(color, fg)
    assert np.array_equal(out, color)
    assert seg._sm_color is not None and seg._sm_color.shape == (4, 4, 3)


def test_dropout_is_held_not_flashed_to_background():
    """A region present then briefly missing must keep its color (no flash)."""
    seg = _make(0.8)  # hold window comfortably > 1 pass
    assert seg._sm_hold >= 2
    color = _tile(PURPLE)
    fg_on = np.ones((4, 4), dtype=bool)
    fg_off = np.zeros((4, 4), dtype=bool)

    seg._persist(color, fg_on)              # seed present
    seg._persist(color, fg_on)              # settle toward purple
    held = seg._persist(_tile((255, 255, 255)), fg_off)  # FastSAM misses it

    # Held color stays purple-ish, NOT the white background block.
    assert held.mean(axis=(0, 1))[0] > 120  # B channel of purple, far from 255-white pull
    assert not np.array_equal(held, _tile((255, 255, 255)))


def test_fades_to_background_after_hold_window():
    """Once absent longer than the hold window, the color eases to background."""
    seg = _make(0.8)
    color = _tile(PURPLE)
    fg_on = np.ones((4, 4), dtype=bool)
    fg_off = np.zeros((4, 4), dtype=bool)

    for _ in range(4):
        seg._persist(color, fg_on)  # firmly purple

    out = None
    for _ in range(seg._sm_hold + 30):  # stay absent well past the hold
        out = seg._persist(_tile((255, 255, 255)), fg_off)

    # After a long absence it has decayed to (near) the white background.
    assert np.allclose(out, 255, atol=5)


def test_foreground_tracks_a_new_color_responsively():
    """A sustained color change is followed, not frozen (still responsive)."""
    seg = _make(0.5)
    fg = np.ones((4, 4), dtype=bool)
    seg._persist(_tile((0, 0, 0)), fg)  # seed black
    out = None
    for _ in range(6):
        out = seg._persist(_tile((255, 255, 255)), fg)  # switch to white, sustained
    assert np.allclose(out, 255, atol=8)  # converged to the new color


def test_resolution_change_reseeds():
    seg = _make(0.5)
    fg4 = np.ones((4, 4), dtype=bool)
    seg._persist(_tile(PURPLE, (4, 4)), fg4)
    fg8 = np.ones((8, 8), dtype=bool)
    out = seg._persist(_tile(PURPLE, (8, 8)), fg8)
    assert out.shape == (8, 8, 3)
    assert seg._sm_color.shape == (8, 8, 3)


def test_deletterbox_color_unpads_to_frame():
    # A 320x320 padded canvas for a 16:9 frame -> content in the middle band.
    small = np.zeros((320, 320, 3), dtype=np.uint8)
    small[70:250, :] = PURPLE  # ~180 content rows centered
    out = Segmenter._deletterbox_color(small, 720, 1280)
    assert out.shape == (720, 1280, 3)
    # The unpadded content should fill (nearly) the whole frame, not a band.
    assert out.mean(axis=(0, 1))[0] > 150  # dominated by purple, not black bars
