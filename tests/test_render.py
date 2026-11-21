from __future__ import annotations

import io
import math

import numpy as np
import pytest
from PIL import Image
from synthetic_poses import figure, result

from argus_puppet.models import RenderOptions
from argus_puppet.render import (
    FACE_COLOR,
    HAND_DOT_COLOR,
    HAND_EDGE_COLORS,
    LIMB_DIM,
    _ellipse_polygon,
    _fill_convex_polygon,
    canvas_transform,
    render_array,
    render_png,
    render_pose,
)
from argus_puppet.skeleton import FACE, FEET, LEFT_HAND, OPENPOSE_COLORS, OPENPOSE_LIMBS, RIGHT_HAND, to_openpose18

BODY_ONLY = RenderOptions(hands=False, face=False, stroke_scale=1.0)  # the annotator's pixel sizes
SIZE = (400, 800)


def big() -> np.ndarray:
    """The standing figure at 4x, in a 400 x 800 frame."""
    return figure(scale=4)


def dim(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(c * LIMB_DIM) for c in color)  # type: ignore[return-value]


def px(img: np.ndarray, x: float, y: float) -> tuple[int, int, int]:
    return tuple(int(v) for v in img[math.floor(y), math.floor(x)])  # type: ignore[return-value]


def colours(img: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(int(v) for v in c) for c in np.unique(img.reshape(-1, 3), axis=0)}


# ── the colour convention ────────────────────────────────────────────────────────────────────────────────────────


def test_hand_edge_colours_follow_the_hsv_wheel() -> None:
    assert HAND_EDGE_COLORS[0] == (255, 0, 0)
    assert HAND_EDGE_COLORS[5] == (128, 255, 0)  # 127.5 rounds half to even
    assert HAND_EDGE_COLORS[10] == (0, 255, 255)
    assert len(HAND_EDGE_COLORS) == 20 and len(set(HAND_EDGE_COLORS)) == 20


# Limbs whose midpoint no other shape of the standing figure covers (the head limbs crowd each other).
CLEAR_LIMBS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


@pytest.mark.parametrize("i", CLEAR_LIMBS)
def test_limb_midpoint_is_its_colour_dimmed(i: int) -> None:
    kp = big()
    img = render_array(kp, BODY_ONLY, source_size=SIZE)
    body = to_openpose18(kp)
    a, b = OPENPOSE_LIMBS[i]
    mid = (body[a, :2] + body[b, :2]) / 2
    assert px(img, *mid) == dim(OPENPOSE_COLORS[i])


def test_every_joint_centre_is_its_colour_at_full_intensity() -> None:
    kp = big()
    img = render_array(kp, BODY_ONLY, source_size=SIZE)
    body = to_openpose18(kp)
    for j in range(18):
        if j in (14, 15, 16, 17):  # eyes and ears sit close together; checked below at a larger scale
            continue
        assert px(img, *body[j, :2]) == OPENPOSE_COLORS[j], j


def test_head_joints_at_a_larger_scale() -> None:
    kp = figure(scale=12)
    img = render_array(kp, BODY_ONLY, source_size=(1200, 2400))
    body = to_openpose18(kp)
    for j in range(18):
        assert px(img, *body[j, :2]) == OPENPOSE_COLORS[j], j


def test_canvas_holds_only_convention_colours_with_hard_edges() -> None:
    img = render_array(big(), source_size=SIZE)
    allowed = (
        {(0, 0, 0), HAND_DOT_COLOR, FACE_COLOR}
        | set(OPENPOSE_COLORS)
        | {dim(c) for c in OPENPOSE_COLORS}
        | set(HAND_EDGE_COLORS)
    )
    assert colours(img) <= allowed  # no anti-aliased in-between shades


def test_limb_ellipse_has_the_stick_width() -> None:
    """Across the middle of the forearm (a vertical-ish limb), the coloured run is 2 x 4 + 1 px wide, as OpenCV's
    polygon fill (outline included) draws a semi-minor axis of 4."""
    kp = big()
    img = render_array(kp, BODY_ONLY, source_size=SIZE)
    body = to_openpose18(kp)
    a, b = OPENPOSE_LIMBS[3]  # right elbow -> right wrist
    mid = (body[a, :2] + body[b, :2]) / 2
    row = img[int(mid[1])]
    run = np.flatnonzero((row == dim(OPENPOSE_COLORS[3])).all(axis=1))
    assert len(run) == 9


# Integer facts measured from the reference annotator itself: controlnet_aux dwpose/util.py draw_bodypose /
# draw_handpose / draw_facepose at commit b1aac318b15c4e65a21bc516dc236ad4bf593d46, OpenCV 4.14, on a 256 x 256
# (limbs) or 64 x 64 (hand, face) canvas. Limbs are limb 3 (right elbow -> right wrist) alone, so its dimmed colour
# counts the limb pixels the two radius-4 joint discs leave uncovered.
REF_LIMBS = {
    # name: (elbow, wrist, limb pixel count, limb bbox (x0, y0, x1, y1) inclusive)
    "horizontal 80 px": ((88.0, 128.0), (168.0, 128.0), 567, (92, 124, 164, 132)),
    # Float angle 44.9 degrees, drawn at int(44.9) = 44; the exact ellipse's box would be (31, 31, 167, 166).
    "angle 44.9": (
        (28 + 200 * math.cos(math.radians(44.9)), 28 + 200 * math.sin(math.radians(44.9))),
        (28.0, 28.0),
        1415,
        (29, 31, 168, 166),
    ),
    # -44.9 degrees truncates toward zero, to -44 (not -45).
    "angle -44.9": (
        (28 + 200 * math.cos(math.radians(44.9)), 228 - 200 * math.sin(math.radians(44.9))),
        (28.0, 228.0),
        1419,
        (27, 89, 168, 224),
    ),
    # Midpoint (100.9, 100.6) truncates to (100, 100): the 9 px run is columns 96..104, not centred on 100.9.
    "centre truncated": ((100.9, 60.6), (100.9, 140.6), 567, (96, 64, 104, 136)),
}


def _one_limb(elbow: tuple[float, float], wrist: tuple[float, float], scale: float = 1.0) -> np.ndarray:
    kp = np.zeros((133, 3))
    kp[8], kp[10] = (*elbow, 1.0), (*wrist, 1.0)  # COCO right elbow / wrist: OpenPose joints 3 and 4
    side = round(256 * scale)
    img = render_array(kp, BODY_ONLY, source_size=(256, 256), canvas_size=(side, side), stroke_scale=scale)
    return (img == dim(OPENPOSE_COLORS[3])).all(axis=-1)


@pytest.mark.parametrize("name", list(REF_LIMBS))
def test_limbs_rasterise_as_the_reference(name: str) -> None:
    # Regression: limbs were exact float ellipses, ~0.5 px thinner per side than the annotator's and up to 1 px / 1
    # degree off its integer-truncated centre, half-length and angle (horizontal limb: 463 px against 567).
    elbow, wrist, area, box = REF_LIMBS[name]
    mask = _one_limb(elbow, wrist)
    ys, xs = np.nonzero(mask)
    assert int(mask.sum()) == area
    assert (xs.min(), ys.min(), xs.max(), ys.max()) == box


def test_the_limb_polygon_is_opencvs() -> None:
    # cv2.fillConvexPoly(cv2.ellipse2Poly((64, 32), (40, 4), 0, 0, 360, 1)) covers 597 px (the exact 40 x 4 ellipse
    # covers 481): the polygon's outline is filled too, so the shape is about half a pixel fatter on every side.
    canvas = np.zeros((64, 128, 3), dtype=np.uint8)
    _fill_convex_polygon(canvas, _ellipse_polygon(64, 32, 40, 4, 0), (255, 255, 255))
    assert int(canvas.any(axis=-1).sum()) == 597


def test_a_scaled_limb_is_the_grown_ellipse_on_the_annotators_grid() -> None:
    # At stroke scale 2 the annotator would draw on a grid half the size: the 80 px limb there is 9 rows thick, so
    # 18 rows here, centred on the grid pixel's centre (2 * 128 + 0.5).
    mask = _one_limb((88.0, 128.0), (168.0, 128.0), scale=2.0)
    rows = np.flatnonzero(mask[:, 256])
    assert (rows.min(), rows.max()) == (248, 265)


def test_black_background_and_canvas_size() -> None:
    img = render_pose(big(), source_size=SIZE)
    assert img.mode == "RGB" and img.size == SIZE
    arr = np.asarray(img)
    assert tuple(arr[0, 0]) == (0, 0, 0) and tuple(arr[-1, -1]) == (0, 0, 0)


# ── toggles and scores ───────────────────────────────────────────────────────────────────────────────────────────


def test_each_toggle_removes_exactly_its_part() -> None:
    kp = big()
    full = render_array(kp, source_size=SIZE)
    for option, part in (("hands", (LEFT_HAND, RIGHT_HAND)), ("face", (FACE,))):
        off = render_array(kp, RenderOptions(**{option: False}), source_size=SIZE)
        unscored = kp.copy()
        for s in part:
            unscored[s, 2] = 0.0
        assert np.array_equal(off, render_array(unscored, source_size=SIZE)), option
        assert not np.array_equal(off, full), option
    no_body = render_array(kp, RenderOptions(body=False), source_size=SIZE)
    unscored = kp.copy()
    unscored[:17, 2] = 0.0
    assert np.array_equal(no_body, render_array(unscored, source_size=SIZE))
    assert not (colours(no_body) & {dim(c) for c in OPENPOSE_COLORS[:17]} - {(0, 0, 0)})


def test_hands_and_face_draw_their_colours() -> None:
    img = render_array(big(), RenderOptions(body=False), source_size=SIZE)
    cs = colours(img)
    assert HAND_DOT_COLOR in cs and FACE_COLOR in cs
    assert set(HAND_EDGE_COLORS) & cs
    hands_only = render_array(big(), RenderOptions(body=False, face=False), source_size=SIZE)
    assert FACE_COLOR not in colours(hands_only)


def test_feet_are_never_drawn() -> None:
    kp = big()
    moved = kp.copy()
    moved[FEET, :2] += 37.0
    assert np.array_equal(render_array(kp, source_size=SIZE), render_array(moved, source_size=SIZE))


def test_a_low_score_joint_suppresses_its_limbs_and_dot() -> None:
    kp = big()
    kp[8, 2] = 0.1  # right elbow (OpenPose 3): limbs 2 (shoulder-elbow) and 3 (elbow-wrist) go
    img = render_array(kp, BODY_ONLY, source_size=SIZE)
    body = to_openpose18(kp)
    for i in (2, 3):
        a, b = OPENPOSE_LIMBS[i]
        assert px(img, *((body[a, :2] + body[b, :2]) / 2)) == (0, 0, 0)
    assert px(img, *body[3, :2]) == (0, 0, 0)
    assert px(img, *body[4, :2]) == OPENPOSE_COLORS[4]  # the wrist still gets its dot
    assert dim(OPENPOSE_COLORS[2]) not in colours(img) and dim(OPENPOSE_COLORS[3]) not in colours(img)


def test_a_low_score_shoulder_drops_the_synthesised_neck() -> None:
    kp = big()
    kp[5, 2] = 0.0
    img = render_array(kp, BODY_ONLY, source_size=SIZE)
    assert OPENPOSE_COLORS[1] not in colours(img)  # neck dot gone (its score is the lower shoulder's)


def test_visibility_follows_the_annotator() -> None:
    # Regression: body joints were drawn at a score of exactly min_score; the annotator needs strictly more for body
    # joints (and both shoulders above it for the neck), while hand and face points at the threshold are drawn.
    kp = big()
    kp[5, 2] = 0.3  # left shoulder exactly at the threshold: no shoulder dot, no neck, no limbs touching either
    img = render_array(kp, BODY_ONLY, source_size=SIZE)
    for c in (OPENPOSE_COLORS[1], OPENPOSE_COLORS[5], dim(OPENPOSE_COLORS[0]), dim(OPENPOSE_COLORS[12])):
        assert c not in colours(img)
    assert OPENPOSE_COLORS[2] in colours(img)  # the other shoulder still drawn
    at = big()
    at[:, 2] = 0.3
    assert not render_array(at, BODY_ONLY, source_size=SIZE).any()  # body: strictly above
    hands_face = render_array(at, RenderOptions(body=False), source_size=SIZE)
    assert HAND_DOT_COLOR in colours(hands_face) and FACE_COLOR in colours(hands_face)  # hands and face: at or above
    assert not render_array(at, RenderOptions(min_score=0.31), source_size=SIZE).any()


HANDS_ONLY = RenderOptions(body=False, face=False, stroke_scale=1.0)


def _hand_root_at(x: float, *, side: int = 64, scale: float = 1.0) -> np.ndarray:
    """A right hand of two points: the root at (x, 20 * scale) and point 1 at (30, 20) * scale (edge 0, red)."""
    kp = np.zeros((133, 3))
    kp[RIGHT_HAND.start] = (x, 20 * scale, 0.9)
    kp[RIGHT_HAND.start + 1] = (30 * scale, 20 * scale, 0.9)
    return render_array(kp, HANDS_ONLY, source_size=(side, side), stroke_scale=scale)


@pytest.mark.parametrize(("x", "drawn"), [(0.4, False), (0.9, False), (1.0, True), (-3.0, False)])
def test_a_hand_point_under_one_pixel_from_the_left_edge_is_dropped_with_its_edges(x: float, drawn: bool) -> None:
    # Regression: the annotator's eps test (int(x * W) > 0.01, i.e. truncated pixel x and y both >= 1) was missing,
    # so a hand point hugging the left or top edge was drawn. Reference (64 x 64): x = 0.4 and 0.9 leave 49 blue px
    # (point 1's dot) and no red; x = 1.0 draws the root dot and the red edge too.
    img = _hand_root_at(x)
    blue = (img == HAND_DOT_COLOR).all(axis=-1)
    red = (img == HAND_EDGE_COLORS[0]).all(axis=-1)
    assert bool(blue[:, :6].any()) is drawn
    assert bool(red.any()) is drawn
    assert blue[:, 20:].sum() == 49  # point 1's dot, drawn either way


@pytest.mark.parametrize(("y", "drawn"), [(0.4, False), (0.9, False), (1.2, True)])
def test_a_face_point_under_one_pixel_from_the_top_edge_is_dropped(y: float, drawn: bool) -> None:
    kp = np.zeros((133, 3))
    kp[FACE.start] = (30, y, 0.9)
    img = render_array(kp, RenderOptions(body=False, hands=False, stroke_scale=1.0), source_size=(64, 64))
    assert int((img == FACE_COLOR).all(axis=-1).sum()) == (23 if drawn else 0)  # reference: 23 px at y = 1.2


def test_body_joints_have_no_near_edge_test() -> None:
    kp = np.zeros((133, 3))
    kp[10] = (0.4, 20, 0.9)  # right wrist, OpenPose joint 4
    img = render_array(kp, BODY_ONLY, source_size=(64, 64))
    assert int((img == OPENPOSE_COLORS[4]).all(axis=-1).sum()) == 29  # reference: the clipped radius-4 disc


def test_the_near_edge_band_is_one_annotator_grid_pixel() -> None:
    # Under stroke scale k the annotator's grid is the canvas divided by k, so the band is k px here.
    for x, drawn in ((1.5, False), (1.99, False), (2.0, True)):
        img = _hand_root_at(x, side=128, scale=2.0)
        assert bool((img == HAND_EDGE_COLORS[0]).all(axis=-1).any()) is drawn, x


# ── people ───────────────────────────────────────────────────────────────────────────────────────────────────────


def _overlap_people() -> tuple[np.ndarray, np.ndarray]:
    """A's neck -> nose limb (12, blue) crosses B's neck -> right-shoulder limb (0, red) at (50, 50), where neither
    has a joint dot; A's first face point sits on B's left-hand root at (20, 20)."""
    a, b = np.zeros((133, 3)), np.zeros((133, 3))
    a[0], a[5], a[6] = (50, 20, 0.9), (40, 80, 0.9), (60, 80, 0.9)  # A: neck (50, 80), nose (50, 20)
    b[5], b[6] = (110, 50, 0.9), (10, 50, 0.9)  # B: neck (60, 50), right shoulder (10, 50)
    a[FACE.start] = (20, 20, 0.9)
    b[LEFT_HAND.start] = (20, 20, 0.9)
    return a, b


def test_two_people_overlap_in_the_annotator_order() -> None:
    # Regression: people were drawn one after another, so overlap colours depended on the order they were listed.
    a, b = _overlap_people()
    for people in ([a, b], [b, a]):
        img = render_array(people, RenderOptions(stroke_scale=1.0), source_size=(100, 100))
        assert px(img, 50, 50) == dim(OPENPOSE_COLORS[12])  # limb 12 is drawn after limb 0, whoever owns it
        assert px(img, 20, 20) == FACE_COLOR  # every face after every hand
    first, second = (render_array(p, source_size=(100, 100)) for p in ([a, b], [b, a]))
    assert np.array_equal(first, second)  # no same-index overlap here, so the person order cannot matter


def test_right_hands_are_drawn_after_every_left_hand() -> None:
    a, b = np.zeros((133, 3)), np.zeros((133, 3))
    a[RIGHT_HAND.start : RIGHT_HAND.start + 2] = [(10, 50, 0.9), (90, 50, 0.9)]  # edge 0 of A's right hand: red
    b[LEFT_HAND.start + 0] = (50, 10, 0.9)
    b[LEFT_HAND.start + 5] = (50, 90, 0.9)  # edge 4 of B's left hand (root -> 5), crossing at (50, 50)
    for people in ([a, b], [b, a]):
        img = render_array(people, RenderOptions(body=False, face=False, stroke_scale=1.0), source_size=(100, 100))
        assert px(img, 50, 50) == HAND_EDGE_COLORS[0]  # A's right hand, drawn after B's left hand


# ── robustness ───────────────────────────────────────────────────────────────────────────────────────────────────


def test_off_canvas_nan_and_huge_keypoints_do_not_crash() -> None:
    kp = big()
    kp[9, :2] = (-500.0, -500.0)
    kp[10, :2] = (np.nan, 30.0)
    kp[13, :2] = (1e15, -1e15)
    kp[14, :2] = (np.inf, 5.0)
    kp[LEFT_HAND, 0] = np.nan
    kp[RIGHT_HAND, :2] = 1e9
    kp[FACE, 2] = np.nan
    img = render_array(kp, source_size=SIZE)
    assert img.shape == (800, 400, 3) and img.any()
    everything_off = big()
    everything_off[:, :2] += 10_000
    assert not render_array(everything_off, source_size=SIZE).any()


def test_a_partly_off_canvas_limb_is_clipped() -> None:
    kp = big()
    kp[:, 0] -= 150  # the figure's right side leaves the canvas
    img = render_array(kp, BODY_ONLY, source_size=SIZE)
    assert img.shape == (800, 400, 3) and img[:, :5].any()


def test_empty_result_is_a_black_canvas() -> None:
    r = result([], width=64, height=48)
    img = render_array(r)
    assert img.shape == (48, 64, 3) and not img.any()


# ── sizing ───────────────────────────────────────────────────────────────────────────────────────────────────────


def joint_centroid(img: np.ndarray, color: tuple[int, int, int]) -> tuple[float, float]:
    ys, xs = np.nonzero((img == color).all(axis=-1))
    return float(xs.mean()), float(ys.mean())


def test_scaling_to_a_larger_canvas_places_joints_where_expected() -> None:
    kp = figure(scale=2)  # 200 x 400 frame
    img = render_array(kp, BODY_ONLY, source_size=(200, 400), canvas_size=(400, 800))
    body = to_openpose18(kp)
    for j in (0, 4, 7, 10, 13):
        x, y = joint_centroid(img, OPENPOSE_COLORS[j])
        assert abs(x - body[j, 0] * 2) <= 1.0 and abs(y - body[j, 1] * 2) <= 1.0, j


def test_contain_letterboxes_without_distortion() -> None:
    kp = figure(scale=2)  # 200 x 400 frame onto a square canvas: scale 1, 100 px bars each side
    img = render_array(kp, BODY_ONLY, source_size=(200, 400), canvas_size=(400, 400))
    body = to_openpose18(kp)
    for j in (0, 4, 7, 10, 13):
        x, y = joint_centroid(img, OPENPOSE_COLORS[j])
        assert abs(x - (body[j, 0] + 100)) <= 1.0 and abs(y - body[j, 1]) <= 1.0, j
    assert not img[:, :100].any() and not img[:, 300:].any()


def test_options_width_alone_keeps_the_aspect() -> None:
    r = result([figure()], width=100, height=200)
    assert render_pose(r, RenderOptions(width=50)).size == (50, 100)
    assert render_pose(r, RenderOptions(height=50)).size == (25, 50)
    assert render_pose(r, RenderOptions(width=30, height=30)).size == (30, 30)
    assert render_pose(r, RenderOptions(width=30, height=30), canvas_size=(7, 9)).size == (7, 9)


def test_transforms() -> None:
    t = canvas_transform((100, 200), (300, 300), "contain")
    assert (t.sx, t.sy, t.ox, t.oy) == (1.5, 1.5, 75.0, 0.0)
    t = canvas_transform((100, 200), (300, 300), "cover")
    assert (t.sx, t.sy, t.ox, t.oy) == (3.0, 3.0, 0.0, -150.0)
    t = canvas_transform((100, 200), (300, 300), "stretch")
    assert (t.sx, t.sy, t.ox, t.oy) == (3.0, 1.5, 0.0, 0.0)
    pts = np.array([[10.0, 10.0], [30.0, 50.0]])
    t = canvas_transform((100, 200), (100, 100), "skeleton", points=pts, margin=0.1)
    mapped = t.apply(pts)
    assert mapped[:, 1].min() == pytest.approx(10.0) and mapped[:, 1].max() == pytest.approx(90.0)
    assert mapped[:, 0].mean() == pytest.approx(50.0)
    with pytest.raises(ValueError):
        canvas_transform((0, 1), (1, 1))


def test_skeleton_fit_fills_the_canvas_with_a_margin() -> None:
    kp = figure()  # tiny in its frame
    img = render_array(kp, BODY_ONLY, source_size=(1000, 1000), canvas_size=(256, 256), fit="skeleton", margin=0.1)
    ys, xs = np.nonzero(img.any(axis=-1))
    assert ys.min() >= 25 - 5 and ys.max() <= 230 + 5  # joints at the margin, dots overhang by their radius
    assert ys.max() - ys.min() > 190


def test_strokes_scale_with_the_canvas_by_default() -> None:
    # Regression: the default drew the annotator's 512 px sizes on any canvas, half the reference thickness at 1024.
    kp = big()
    assert RenderOptions().stroke_scale == "auto"
    for side, k in ((1024, 2.0), (512, 1.0), (256, 0.5)):
        default = render_array(kp, RenderOptions(hands=False, face=False), source_size=SIZE, canvas_size=(side, side))
        fixed = render_array(kp, BODY_ONLY, source_size=SIZE, canvas_size=(side, side), stroke_scale=k)
        assert np.array_equal(default, fixed), side


def test_stroke_scale_thickens_strokes() -> None:
    kp = big()
    thin = render_array(kp, BODY_ONLY, source_size=SIZE)
    thick = render_array(kp, BODY_ONLY, source_size=SIZE, stroke_scale=2.0)
    assert np.count_nonzero(thick.any(axis=-1)) > 1.5 * np.count_nonzero(thin.any(axis=-1))
    auto = render_array(kp, BODY_ONLY, source_size=SIZE, canvas_size=(1024, 2048), stroke_scale="auto")
    assert auto.shape == (2048, 1024, 3)
    with pytest.raises(ValueError):
        render_array(kp, source_size=SIZE, stroke_scale=0)


def test_all_people_by_default_and_primary_only_on_request() -> None:
    a, b = figure(scale=2), figure(scale=2, dx=200)
    r = result([a, b], width=400, height=400)
    both = render_array(r, BODY_ONLY)
    only = render_array(r, BODY_ONLY, primary_only=True)
    assert both[:, 200:].any() and both[:, :200].any()
    assert only[:, :200].any() and not only[:, 200:].any()
    assert np.array_equal(only, render_array(a, BODY_ONLY, source_size=(400, 400)))
    second_primary = result([a, b], width=400, height=400, primary=1)
    assert not render_array(second_primary, BODY_ONLY, primary_only=True)[:, :200].any()


def test_people_as_arrays_and_stacks() -> None:
    a, b = figure(scale=2), figure(scale=2, dx=200)
    stack = render_array(np.stack([a, b]), source_size=(400, 400))
    listed = render_array([a, b], source_size=(400, 400))
    assert np.array_equal(stack, listed)
    with pytest.raises(ValueError):
        render_array(a)  # bare arrays need a source size
    with pytest.raises(ValueError):
        render_array(np.zeros((17, 3)), source_size=(10, 10))


def test_limbs_of_every_person_are_dimmed_once() -> None:
    a, b = figure(scale=2), figure(scale=2, dx=200)
    img = render_array([a, b], BODY_ONLY, source_size=(400, 400))
    for kp in (a, b):
        body = to_openpose18(kp)
        x, y = (body[3, :2] + body[4, :2]) / 2
        assert px(img, x, y) == dim(OPENPOSE_COLORS[3])


# ── output ───────────────────────────────────────────────────────────────────────────────────────────────────────


def test_png_is_deterministic_and_decodes_to_the_array() -> None:
    r = result([figure(scale=2), figure(scale=2, dx=200)], width=400, height=400)
    one, two = render_png(r), render_png(r)
    assert one == two and one[:8] == b"\x89PNG\r\n\x1a\n"
    decoded = np.asarray(Image.open(io.BytesIO(one)).convert("RGB"))
    assert np.array_equal(decoded, render_array(r))


def test_options_carry_fit_stroke_margin_and_primary_and_keywords_override_them() -> None:
    a, b = figure(), figure(dx=40, arms_up=True)
    r = result([a, b], width=200, height=200, primary=1)
    opts = RenderOptions(width=120, height=300, fit="skeleton", margin=0.2, stroke_scale=2.0, primary_only=True)
    from_options = render_array(r, opts)
    by_keywords = render_array(
        r, RenderOptions(width=120, height=300), fit="skeleton", margin=0.2, stroke_scale=2.0, primary_only=True
    )
    assert np.array_equal(from_options, by_keywords)
    assert not np.array_equal(from_options, render_array(r, opts, fit="contain"))  # a keyword wins over the option
    assert np.array_equal(
        render_array(r, opts, primary_only=False), render_array(r, opts.model_copy(update={"primary_only": False}))
    )
    ys, xs = np.nonzero(from_options.any(axis=-1))
    assert ys.min() >= 0.2 * 300 - 9 and ys.max() <= 0.8 * 300 + 9  # inside the 20 % margin, give a stroke
    auto = RenderOptions(width=1024, height=1024, stroke_scale="auto", hands=False, face=False)
    assert np.count_nonzero(render_array(a, auto, source_size=(100, 200)).any(-1)) > np.count_nonzero(
        render_array(a, auto, source_size=(100, 200), stroke_scale=1.0).any(-1)
    )
