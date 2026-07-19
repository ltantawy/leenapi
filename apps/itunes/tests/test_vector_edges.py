"""Unit tests for the vector (smooth, rounded) block-edge renderer.

These target the pixelation the renderer exists to remove: FastSAM masks arrive
at 320x320 and used to be upscaled to the frame with nearest-neighbour, turning
every boundary into a ~4 px staircase. ``Segmenter._vector_render`` traces the
masks as polygons, rounds them in point space and fills them at frame
resolution with subpixel anti-aliased edges.

They exercise the renderer directly, without loading the FastSAM model, by
building a bare instance and wiring the same knobs ``__init__`` derives.
"""

import numpy as np

from src.segmenter import (
    _PALETTE_SIZE,
    _VEC_EPS_MAX,
    _VEC_EPS_MIN,
    _VEC_ITERS_MAX,
    _VEC_CUT_MAX,
    _VEC_CUT_MIN,
    _VEC_ITERS_MIN,
    Segmenter,
    _build_palette,
    _chaikin,
    _letterbox_params,
    _smooth_poly,
)


def _make(edge_smooth: float = 0.5, bg_color=(255, 255, 255), overlay=False) -> Segmenter:
    """A Segmenter with only the render state set (no model load)."""
    seg = Segmenter.__new__(Segmenter)
    e = float(np.clip(edge_smooth, 0.0, 1.0))
    seg.vector = True
    seg.edge_smooth = e
    seg._vec_eps = _VEC_EPS_MIN + (_VEC_EPS_MAX - _VEC_EPS_MIN) * e
    seg._vec_iters = int(round(_VEC_ITERS_MIN + (_VEC_ITERS_MAX - _VEC_ITERS_MIN) * e))
    seg._vec_cut = _VEC_CUT_MIN + (_VEC_CUT_MAX - _VEC_CUT_MIN) * e
    seg.palette = _build_palette(_PALETTE_SIZE)
    seg.bg_color = np.array(bg_color, dtype=np.uint8)
    seg.overlay = overlay
    seg.alpha = 0.5
    seg.bg_alpha = 0.85
    seg.stability = 0.0
    seg._sm_color = None
    seg._sm_age = None
    return seg


def _square_mask(mh=320, mw=320, box=(120, 120, 200, 200)) -> np.ndarray:
    """One boolean mask (1, mh, mw) with a filled axis-aligned square."""
    m = np.zeros((1, mh, mw), dtype=bool)
    y0, x0, y1, x1 = box
    m[0, y0:y1, x0:x1] = True
    return m


def _render(seg, masks_bool, h=720, w=1280):
    order = np.array([0])
    return seg._vector_render(masks_bool, order, np.array([0]), h, w)


# --- point-space smoothing primitives -------------------------------------


def test_chaikin_doubles_points_and_stays_inside_hull():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    out = _chaikin(square, 2)
    assert len(out) == len(square) * 4  # doubles per iteration
    # Corner cutting only ever moves points inward: never outside the original.
    assert out.min() >= -1e-5 and out.max() <= 10 + 1e-5


def test_chaikin_rounds_corners_off_the_original_vertices():
    """After cutting, no point sits on a sharp original corner."""
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    out = _chaikin(square, 2)
    for corner in square:
        assert np.abs(out - corner).sum(axis=1).min() > 0.5


def test_chaikin_zero_iters_is_identity():
    pts = np.array([[0, 0], [4, 0], [4, 4]], dtype=np.float32)
    assert np.array_equal(_chaikin(pts, 0), pts)


def test_smooth_poly_removes_the_pixel_staircase():
    """A real mask's stair-stepped diagonal edge collapses to its true corners.

    This is the core anti-pixelation claim: the hundreds of 1-px zigzag points a
    rasterized diagonal produces are exactly the "pixelation", and simplifying
    them away is what makes the upscaled edge straight instead of stepped.
    """
    import cv2

    yy, xx = np.mgrid[0:320, 0:320]
    m = ((yy > 70) & (yy < 250) & ((yy - 70) > xx * 0.5)).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    pts = contours[0][:, 0, :].astype(np.float32)
    assert len(pts) > 200, "the raw contour should be full of staircase points"

    out = _smooth_poly(pts, eps=1.2, iters=2)
    # 2 Chaikin iterations quadruple whatever survived simplification, so a
    # result this small proves the zigzag was discarded, not merely rounded.
    assert len(out) <= 32


# --- letterbox geometry ----------------------------------------------------


def test_letterbox_params_finds_the_16_9_content_band():
    top, left, uh, uw = _letterbox_params(320, 320, 720, 1280)
    assert (uh, uw) == (180, 320)  # 16:9 fit into a square canvas
    assert left == 0 and top == 70  # padding only above/below


# --- full render -----------------------------------------------------------


def test_render_fills_background_and_instance():
    seg = _make()
    color, fg = _render(seg, _square_mask())
    assert color.shape == (720, 1280, 3) and fg.shape == (720, 1280)
    assert fg.any(), "instance should cover some pixels"
    assert not fg.all(), "background should remain"
    # Untouched pixels are the white background block.
    assert np.array_equal(color[0, 0], seg.bg_color)


def test_render_places_the_square_where_the_mask_put_it():
    """Geometry survives the de-letterbox: the block lands in the right place."""
    seg = _make(edge_smooth=0.0)  # faithful, so we can check position tightly
    # Content band for a 16:9 frame in a 320 square is rows 70..250.
    color, fg = _render(seg, _square_mask(box=(70, 0, 250, 160)))
    # Mask covers the full content height and the left half -> same in frame.
    ys, xs = np.nonzero(fg)
    assert ys.min() < 10 and ys.max() > 710       # spans the frame vertically
    assert xs.min() < 10 and xs.max() < 700       # left half only
    assert not fg[:, 900:].any()                  # right half untouched


def test_edges_are_antialiased_not_hard_stepped():
    """The boundary must contain intermediate colors, not a hard 2-color jump."""
    seg = _make()
    color, fg = _render(seg, _square_mask())
    # Walk a row crossing the block edge; count distinct colors along it.
    row = color[360]
    uniq = np.unique(row.reshape(-1, 3), axis=0)
    # Background + instance would be exactly 2 without anti-aliasing.
    assert len(uniq) > 2, "expected blended edge pixels from LINE_AA"


def test_vector_edges_are_smoother_than_a_nearest_upscale():
    """The whole point: the boundary moves smoothly instead of in whole-pixel steps.

    A nearest-neighbour upscale can only place an edge on a pixel boundary, so a
    diagonal advances in 4-pixel jumps — the visible staircase. The vector fill
    is anti-aliased, so partial coverage encodes where the edge really falls and
    it advances a fraction of a pixel per column.

    Measured on *coverage* rather than a thresholded edge row: thresholding
    quantizes the boundary back to whole pixels and would report a staircase for
    any non-integer slope, hiding exactly the difference under test.
    """
    import cv2

    mh = mw = 320
    m = np.zeros((1, mh, mw), dtype=bool)
    yy, xx = np.mgrid[0:mh, 0:mw]
    # A diagonal edge inside the 16:9 content band (rows 70..250).
    m[0] = (yy > 70) & (yy < 250) & (yy - 70 > (xx * 0.5))

    seg = _make()
    color, _ = _render(seg, m)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # Coverage in [0, 1]: 0 where the white background shows, 1 inside the block,
    # fractional on anti-aliased edge pixels.
    vec_cov = np.clip((255.0 - gray) / (255.0 - gray.min()), 0.0, 1.0).sum(axis=0)

    # The old path: nearest-neighbour upscale of the same mask, no partial
    # coverage anywhere.
    crop = m[0][70:250, :].astype(np.uint8)
    near_cov = cv2.resize(crop, (1280, 720), interpolation=cv2.INTER_NEAREST).sum(axis=0)

    # Roughness = mean |2nd difference| of how far the boundary has advanced.
    # A constant-slope edge scores ~0; whole-pixel stepping scores ~1.
    def roughness(profile):
        span = slice(100, 600)  # columns where both boundaries are interior
        return np.abs(np.diff(profile.astype(np.float64)[span], n=2)).mean()

    assert roughness(vec_cov) < 0.5 * roughness(near_cov)


def test_adjacent_regions_do_not_open_a_background_seam():
    """Neighbouring masks must stay touching, not part into a visible gap.

    Chaikin only moves points inward, so without the compensating grow every
    shape shrinks and the frame fills with hairline background seams — breaking
    the "every pixel inside a colored block" fill. Two masks sharing a border
    must still meet after smoothing.
    """
    seg = _make(edge_smooth=1.0)  # strongest smoothing = worst-case shrink
    m = np.zeros((2, 320, 320), dtype=bool)
    m[0, 90:160, 60:260] = True   # upper half
    m[1, 160:230, 60:260] = True  # lower half, sharing the row-160 border
    order = np.array([0, 1])
    _, fg = seg._vector_render(m, order, np.array([0, 1]), 720, 1280)

    # The shared border maps to frame row (160-70)*4 = 360. Sample across it,
    # away from the outer edges of the pair.
    assert fg[355:366, 400:900].all(), "a seam opened between adjacent blocks"


def test_specks_are_dropped():
    """Single-pixel noise is discarded rather than rendered as a tiny block."""
    seg = _make()
    m = np.zeros((1, 320, 320), dtype=bool)
    m[0, 150, 150] = True  # one pixel, far below the min-area threshold
    _, fg = _render(seg, m)
    assert not fg.any()


def test_holes_are_cut_out_not_filled():
    """A donut mask keeps its hole (fillPoly even-odd across all contours)."""
    seg = _make(edge_smooth=0.0)
    m = np.zeros((1, 320, 320), dtype=bool)
    m[0, 90:230, 60:260] = True
    m[0, 130:190, 120:200] = False  # punch a hole
    _, fg = _render(seg, m)
    # Mask x maps to frame x by *4 (left pad 0): ring spans 240..1040, hole
    # 480..800. Mask y maps by *4 after the 70-row top pad: ring 80..640.
    assert not fg[360, 640], "hole center should not be covered"
    assert fg[360, 300], "ring should be covered left of the hole"
    assert fg[360, 900], "ring should be covered right of the hole"


# --- alpha / mode plumbing -------------------------------------------------


def test_blocks_mode_alpha_is_solid():
    seg = _make(overlay=False)
    color, fg = _render(seg, _square_mask())
    _, alpha = seg._finish_vector(color, fg)
    assert np.array_equal(alpha, np.ones_like(alpha))


def test_overlay_mode_alpha_splits_instance_and_background():
    seg = _make(overlay=True)
    color, fg = _render(seg, _square_mask())
    _, alpha = seg._finish_vector(color, fg)
    assert np.isclose(alpha[fg].max(), seg.alpha)
    assert np.isclose(alpha[~fg].max(), seg.bg_alpha)


def test_persistence_runs_at_frame_resolution_in_vector_mode():
    seg = _make()
    seg.stability = 0.5
    seg._sm_fg_beta, seg._sm_hold, seg._sm_bg_beta = 0.65, 3, 0.7
    color, fg = _render(seg, _square_mask())
    out, _ = seg._finish_vector(color, fg)
    assert out.shape == (720, 1280, 3)
    assert seg._sm_color.shape == (720, 1280, 3)


def test_edge_smooth_zero_still_renders():
    """The low end of the knob is faithful-but-antialiased, not broken."""
    seg = _make(edge_smooth=0.0)
    assert seg._vec_iters == _VEC_ITERS_MIN
    _, fg = _render(seg, _square_mask())
    assert fg.any()
