"""OpenPose-convention skeleton control images for pose ControlNets.

SD/SDXL pose ControlNets were trained on skeletons drawn in one fixed colour convention (the one the OpenPose and
DWPose annotators produce). This module draws that convention from scratch with numpy and Pillow, treating it as an
interoperability spec:

* a black RGB canvas;
* **body**: the wholebody keypoints become OpenPose-18 (:func:`skeleton.to_openpose18`, the neck synthesised from
  the shoulders). Each of the 17 :data:`skeleton.OPENPOSE_LIMBS` whose two joints are both visible is a filled
  ellipse centred on the limb's midpoint, semi-major half the limb's length, semi-minor :data:`STICK_WIDTH`, in
  ``OPENPOSE_COLORS[i]``, rasterised as the annotator does: centre, half-length and angle truncated to integers on
  its grid, then OpenCV's 1-degree polygon and convex fill (reproduced exactly at a stroke scale of 1; at any other
  scale, the exact ellipse that fill covers, grown half a grid pixel on both axes, scaled). Once every limb of
  every person is drawn the whole limb layer is multiplied by 0.6
  (:data:`LIMB_DIM`); then each visible joint ``j`` is a filled disc of radius :data:`JOINT_RADIUS` in
  ``OPENPOSE_COLORS[j]`` at full intensity;
* **hands**: per hand, the 20 :data:`skeleton.HAND_EDGES`; edge ``i`` is a :data:`HAND_LINE_WIDTH` px line in
  ``hsv(i / 20, 1, 1)``, drawn when both ends are visible; then each visible hand point is a radius
  :data:`HAND_DOT_RADIUS` disc in pure blue;
* **face**: each visible face landmark (all 68) is a white disc of radius :data:`FACE_DOT_RADIUS`;
* feet are never drawn.

**Visibility** follows the DWPose annotator: a body joint is visible when its score is strictly above ``min_score``
(so the synthesised neck, scored as the lower shoulder, only when both shoulders are), a hand or face point when its
score is at or above it and its coordinates, truncated to integers on the annotator's grid (canvas pixels divided by
the stroke scale), are both at least 1: a hand or face point less than one grid pixel from the left or top edge, or
off the canvas there, is dropped with every hand edge touching it (body joints have no such test). A point with a
non-finite coordinate, or one more than 1e6 px from the origin, is never drawn (treated as unplaced; it keeps the
pixel maths in range).

**Draw order** across people is the annotator's: limb 0 of every person, then limb 1 of every person, and so on; the
0.6 dimming; joint 0 of every person, joint 1, ...; then every left hand, then every right hand (each hand's edges,
then its dots); then every face. So where two people overlap, the higher limb or joint index wins whichever person
is listed first. Later shapes overwrite earlier ones; nothing is blended. Shapes are aliased (hard edges), as the
reference annotators draw them, so every pixel is either black or one of the convention's colours.

**Sizing.** Keypoints live in their source image's pixel frame. The canvas is the source size unless a size is
asked for. ``fit`` decides how the source frame maps onto a canvas of another shape:

* ``"contain"`` (default): one uniform scale so the whole source frame fits, centred (letterboxed). The skeleton is
  never distorted.
* ``"cover"``: one uniform scale so the source frame fills the canvas, centred; parts may fall outside.
* ``"stretch"``: independent x and y scales (distorts the skeleton unless the aspect ratios match).
* ``"skeleton"``: ignore the source frame and fit the drawn points' bounding box into the canvas with a margin,
  uniformly scaled and centred. Used to re-frame library poses for a batch.

Every one of these (``fit``, ``margin``, ``stroke_scale``, ``primary_only``) is a :class:`RenderOptions` field; the
keyword arguments of the render functions override the options when given (``None`` means "take the option").

A point maps as ``x' = x * scale + offset``. **Line widths and radii scale with the canvas** by default
(``stroke_scale="auto"``, ``min(width, height) / 512``): the annotators draw 4 px stick half-widths, radius-4 joints,
2 px hand lines and radius-3 face dots at a 512 px working resolution and resize the result to the output size, so
their strokes grow and shrink with it, below 1 on a canvas under 512 px too. A number fixes the factor
(``stroke_scale=1`` draws the annotator's pixel sizes at any canvas size); hand lines never go below 1 px.
"""

from __future__ import annotations

import colorsys
import io
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from argus_puppet.models import Fit, PersonPose, PoseLibraryEntry, PoseResult, RenderOptions
from argus_puppet.skeleton import (
    FACE,
    HAND_EDGES,
    LEFT_HAND,
    NUM_KEYPOINTS,
    OPENPOSE_COLORS,
    OPENPOSE_LIMBS,
    RIGHT_HAND,
    to_openpose18,
)

__all__ = [
    "FACE_COLOR",
    "FACE_DOT_RADIUS",
    "HAND_DOT_COLOR",
    "HAND_DOT_RADIUS",
    "HAND_EDGE_COLORS",
    "HAND_LINE_WIDTH",
    "JOINT_RADIUS",
    "LIMB_DIM",
    "REFERENCE_RESOLUTION",
    "STICK_WIDTH",
    "Fit",
    "PoseSource",
    "Transform",
    "canvas_size",
    "canvas_transform",
    "render_array",
    "render_png",
    "render_pose",
]

STICK_WIDTH = 4  # limb ellipse semi-minor axis, px
JOINT_RADIUS = 4
LIMB_DIM = 0.6
HAND_LINE_WIDTH = 2
HAND_DOT_RADIUS = 4
HAND_DOT_COLOR = (0, 0, 255)
FACE_DOT_RADIUS = 3
FACE_COLOR = (255, 255, 255)
# stroke_scale="auto" scales strokes by min(canvas side) / this.
REFERENCE_RESOLUTION = 512


def _hand_edge_color(i: int) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(i / len(HAND_EDGES), 1.0, 1.0)
    return (round(r * 255), round(g * 255), round(b * 255))


HAND_EDGE_COLORS: tuple[tuple[int, int, int], ...] = tuple(_hand_edge_color(i) for i in range(len(HAND_EDGES)))

PoseSource = PoseResult | PoseLibraryEntry | PersonPose | np.ndarray | Sequence[PersonPose] | Sequence[np.ndarray]
_DEFAULT_MARGIN = 0.05


# ── geometry ─────────────────────────────────────────────────────────────────────────────────────────────────────


class Transform:
    """Source pixel coordinates to canvas pixel coordinates: ``x' = x * sx + ox``, ``y' = y * sy + oy``."""

    __slots__ = ("ox", "oy", "sx", "sy")

    def __init__(self, sx: float, sy: float, ox: float, oy: float) -> None:
        self.sx, self.sy, self.ox, self.oy = sx, sy, ox, oy

    def apply(self, kp: np.ndarray) -> np.ndarray:
        """Map ``(..., >=2)`` points (extra columns, e.g. scores, ride along)."""
        out = np.array(kp, dtype=np.float64, copy=True)
        out[..., 0] = out[..., 0] * self.sx + self.ox
        out[..., 1] = out[..., 1] * self.sy + self.oy
        return out

    def __repr__(self) -> str:
        return f"Transform(sx={self.sx!r}, sy={self.sy!r}, ox={self.ox!r}, oy={self.oy!r})"


def canvas_transform(
    source_size: tuple[int, int],
    canvas_size: tuple[int, int],
    fit: Fit = "contain",
    *,
    points: np.ndarray | None = None,
    margin: float = _DEFAULT_MARGIN,
) -> Transform:
    """The mapping from a ``(width, height)`` source frame onto a ``(width, height)`` canvas under *fit*.

    ``fit="skeleton"`` fits the bounding box of *points* (``(N, >=2)`` canvas-agnostic source coordinates, already
    filtered to the ones that will be drawn) into the canvas less *margin* (a fraction of each side kept clear on
    every edge); with no points it falls back to ``"contain"``.
    """
    sw, sh = (float(v) for v in source_size)
    cw, ch = (float(v) for v in canvas_size)
    if sw <= 0 or sh <= 0 or cw <= 0 or ch <= 0:
        raise ValueError(f"sizes must be positive, got source {source_size} and canvas {canvas_size}")
    if fit == "stretch":
        return Transform(cw / sw, ch / sh, 0.0, 0.0)
    if fit == "skeleton":
        if not 0.0 <= margin < 0.5:
            raise ValueError(f"margin must be in [0, 0.5), got {margin}")
        pts = np.zeros((0, 2)) if points is None else np.asarray(points, dtype=np.float64)
        pts = pts.reshape(-1, pts.shape[-1])[:, :2] if pts.size else np.zeros((0, 2))
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts):
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            bw, bh = float(hi[0] - lo[0]), float(hi[1] - lo[1])
            aw, ah = cw * (1 - 2 * margin), ch * (1 - 2 * margin)
            if bw <= 0 and bh <= 0:
                s = 1.0
            elif bw <= 0:
                s = ah / bh
            elif bh <= 0:
                s = aw / bw
            else:
                s = min(aw / bw, ah / bh)
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            return Transform(s, s, cw / 2 - cx * s, ch / 2 - cy * s)
        fit = "contain"
    if fit == "contain":
        s = min(cw / sw, ch / sh)
    elif fit == "cover":
        s = max(cw / sw, ch / sh)
    else:
        raise ValueError(f"unknown fit {fit!r}")
    return Transform(s, s, (cw - sw * s) / 2, (ch - sh * s) / 2)


def canvas_size(
    source_size: tuple[int, int], options: RenderOptions | None = None, size: tuple[int, int] | None = None
) -> tuple[int, int]:
    """The ``(width, height)`` a render draws on: *size* if given, else ``options.width`` x ``options.height`` (one
    alone keeps the source's aspect), else *source_size*."""
    options = options or RenderOptions()
    if size is not None:
        w, h = int(size[0]), int(size[1])
    elif options.width is not None and options.height is not None:
        w, h = options.width, options.height
    elif options.width is not None:  # one side given: keep the source's aspect
        w = options.width
        h = max(1, round(options.width * source_size[1] / source_size[0]))
    elif options.height is not None:
        h = options.height
        w = max(1, round(options.height * source_size[0] / source_size[1]))
    else:
        w, h = int(source_size[0]), int(source_size[1])
    if w < 1 or h < 1:
        raise ValueError(f"canvas size must be positive, got {(w, h)}")
    return w, h


_canvas_size = canvas_size


# ── input normalisation ──────────────────────────────────────────────────────────────────────────────────────────


def _people_arrays(
    pose: PoseSource, source_size: tuple[int, int] | None, primary_only: bool
) -> tuple[list[np.ndarray], tuple[int, int]]:
    """Each person's ``(133, 3)`` keypoints, and the source frame size."""
    if isinstance(pose, PoseResult):
        people = [pose.primary_person] if primary_only else list(pose.people)
        arrays = [p.keypoints_array() for p in people if p is not None]
        return arrays, source_size or (pose.width, pose.height)
    if isinstance(pose, PoseLibraryEntry):
        return [pose.keypoints_array()], source_size or (pose.width, pose.height)
    if isinstance(pose, PersonPose):
        items: list[PersonPose | np.ndarray] = [pose]
    elif isinstance(pose, np.ndarray):
        arr = np.asarray(pose, dtype=np.float64)
        items = [arr] if arr.ndim == 2 else list(arr)
    else:
        items = list(pose)
    if primary_only:
        flagged = [p for p in items if isinstance(p, PersonPose) and p.primary]
        items = flagged[:1] if flagged else items[:1]
    arrays = [p.keypoints_array() if isinstance(p, PersonPose) else np.asarray(p, dtype=np.float64) for p in items]
    for a in arrays:
        if a.shape != (NUM_KEYPOINTS, 3):
            raise ValueError(f"expected keypoints shaped ({NUM_KEYPOINTS}, 3), got {a.shape}")
    if source_size is None:
        raise ValueError("source_size is required when rendering bare people or arrays")
    return arrays, source_size


# Coordinates further than this from the origin are treated as unplaced (keeps pixel maths in range).
_MAX_COORD = 1e6


def _confident(points: np.ndarray, min_score: float, *, strict: bool = False) -> np.ndarray:
    """Mask of points that are drawn: finite, sane coordinates and a score at or above *min_score* (strictly above
    with *strict*, the annotator's rule for body joints)."""
    xy = np.nan_to_num(points[:, :2], nan=np.inf)
    placed = (np.abs(xy) < _MAX_COORD).all(axis=1)
    score = np.nan_to_num(points[:, 2], nan=-1.0)
    return placed & (score > min_score if strict else score >= min_score)


def _drawn_points(arrays: list[np.ndarray], options: RenderOptions) -> np.ndarray:
    """Every point the options would draw, in source coordinates (for ``fit="skeleton"``)."""
    out: list[np.ndarray] = []
    for kp in arrays:
        parts: list[tuple[np.ndarray, bool]] = []
        if options.body:
            parts.append((to_openpose18(kp), True))
        if options.hands:
            parts += [(kp[LEFT_HAND], False), (kp[RIGHT_HAND], False)]
        if options.face:
            parts.append((kp[FACE], False))
        for p, strict in parts:
            out.append(p[_confident(p, options.min_score, strict=strict)])
    return np.concatenate(out) if out else np.zeros((0, 3))


# ── rasterising ──────────────────────────────────────────────────────────────────────────────────────────────────


def _box(cx: float, cy: float, rx: float, ry: float, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Integer pixel bounds ``(x0, x1, y0, y1)`` (exclusive ends) of a box around a centre, clipped to the canvas."""
    h, w = shape
    x0, x1 = max(0, math.floor(cx - rx)), min(w, math.ceil(cx + rx) + 1)
    y0, y1 = max(0, math.floor(cy - ry)), min(h, math.ceil(cy + ry) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, x1, y0, y1


# OpenCV's SinTable: the sine of every whole degree 0..450, written to 7 decimals and stored as float32.
_SIN_TABLE = np.round(np.sin(np.radians(np.arange(451))), 7).astype(np.float32).astype(np.float64)
_WHOLE_DEGREES = np.arange(361)
_XY_SHIFT = 16  # OpenCV's fixed-point scanline precision
_XY_HALF = 1 << (_XY_SHIFT - 1)


def _ellipse_polygon(cx: int, cy: int, a: int, b: int, angle: int) -> np.ndarray:
    """The ``(n, 2)`` integer vertices ``cv2.ellipse2Poly((cx, cy), (a, b), angle, 0, 360, 1)`` returns: a point per
    whole degree from the sine table, rounded half to even, consecutive repeats dropped."""
    angle %= 360
    cos_t, sin_t = _SIN_TABLE[450 - angle], _SIN_TABLE[angle]
    u = a * _SIN_TABLE[450 - _WHOLE_DEGREES]
    v = b * _SIN_TABLE[_WHOLE_DEGREES]
    pts = np.stack([np.rint(cx + u * cos_t - v * sin_t), np.rint(cy + u * sin_t + v * cos_t)], axis=1).astype(np.int64)
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = (pts[1:] != pts[:-1]).any(axis=1)
    pts = pts[keep]
    return np.array([[cx, cy], [cx, cy]], dtype=np.int64) if len(pts) == 1 else pts


def _line_pixels(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pixels of OpenCV's 8-connected 1 px lines from each ``p0[i]`` to ``p1[i]`` (integer ``(E, 2)``), as its
    ``LineIterator`` walks them: left to right, one pixel per step along the major axis, a minor step after pixel
    ``i`` when the Bresenham error is negative (closed form ``(2 m i + M - 1) // 2M``). Lines are not clipped."""
    swap = p1[:, 0] < p0[:, 0]
    start = np.where(swap[:, None], p1, p0)
    end = np.where(swap[:, None], p0, p1)
    dx = end[:, 0] - start[:, 0]
    dy = end[:, 1] - start[:, 1]
    y_major = np.abs(dy) > dx
    major = np.where(y_major, np.abs(dy), dx)
    minor = np.where(y_major, dx, np.abs(dy))
    count = major + 1
    edge = np.repeat(np.arange(len(count)), count)
    step = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    big, small = major[edge], minor[edge]
    side = np.where(big > 0, (2 * small * step + big - 1) // np.maximum(2 * big, 1), 0)  # a 0-length edge: 1 px
    ys_major = y_major[edge]
    sign = np.where(dy[edge] < 0, -1, 1)
    xs = start[edge, 0] + np.where(ys_major, side, step)
    ys = start[edge, 1] + np.where(ys_major, step, side) * sign
    return xs, ys


def _trunc_div(a: int, b: int) -> int:
    """C integer division (truncates toward zero); *b* > 0."""
    return a // b if a >= 0 else -(-a // b)


def _fill_convex_polygon(canvas: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    """``cv2.fillConvexPoly(canvas, pts, color)`` (8-connected, no sub-pixel shift), reproduced: the outline as 1 px
    lines, then OpenCV's scanline fill (two edge walkers from the top vertex, 16-bit fixed-point x stepping, spans
    rounded half up, the bottom row left to the outline). The one known difference: OpenCV clips outline lines that
    leave the canvas before walking them, which can move a pixel on the canvas border."""
    h, w = canvas.shape[:2]
    n = len(pts)
    # Outline: edge i runs from vertex i - 1 to vertex i. Only edges whose box touches the canvas are walked.
    p0, p1 = np.roll(pts, 1, axis=0), pts
    lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
    near = (hi[:, 0] >= 0) & (lo[:, 0] < w) & (hi[:, 1] >= 0) & (lo[:, 1] < h)
    if near.any():
        xs, ys = _line_pixels(p0[near], p1[near])
        on = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        canvas[ys[on], xs[on]] = color
    vx, vy = pts[:, 0].tolist(), pts[:, 1].tolist()
    ymin, ymax = min(vy), max(vy)
    if n < 3 or max(vx) < 0 or ymax < 0 or min(vx) >= w or ymin >= h:
        return
    ymax = min(ymax, h - 1)
    # The scanline loop, stepped from event to event (an edge walker reaching its edge's end) rather than row by
    # row: between events x advances by a constant dx per row. Each run of rows is recorded, then all are filled.
    idx = [vy.index(ymin)] * 2
    di = [1, n - 1]
    x = [0, 0]
    dx = [0, 0]
    ye = [ymin, ymin]
    y, edges = ymin, n
    runs: list[tuple[int, int, int, int, int, int]] = []  # first row, end row, x0, dx0, x1, dx1 at the first row
    while True:
        for i in range(2):
            if y < ye[i]:
                continue
            i0 = idx[i]
            i1 = (i0 + di[i]) % n
            while True:
                more, edges = edges > 0, edges - 1
                if not more:
                    break
                ty = vy[i1]
                if ty > y:
                    xs0, xe0 = vx[i0] << _XY_SHIFT, vx[i1] << _XY_SHIFT
                    ye[i], x[i], idx[i] = ty, xs0, i1
                    dx[i] = _trunc_div((xe0 - xs0) * 2 + (ty - y), 2 * (ty - y))
                    break
                i0, i1 = i1, (i1 + di[i]) % n
        if edges < 0:
            break
        nxt = min(ye[0], ye[1], ymax + 1)
        r0 = max(y, 0)  # rows above the canvas only advance the walkers
        if nxt > r0:
            runs.append((r0, nxt, x[0] + (r0 - y) * dx[0], dx[0], x[1] + (r0 - y) * dx[1], dx[1]))
        x[0] += (nxt - y) * dx[0]
        x[1] += (nxt - y) * dx[1]
        y = nxt
        if y > ymax:
            break
    if not runs:
        return
    r = np.array(runs, dtype=np.int64)
    count = r[:, 1] - r[:, 0]
    run = np.repeat(np.arange(len(r)), count)
    step = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
    xa = r[run, 2] + step * r[run, 3]
    xb = r[run, 4] + step * r[run, 5]
    x_lo = (np.minimum(xa, xb) + _XY_HALF) >> _XY_SHIFT  # spans rounded half up, as OpenCV's
    x_hi = (np.maximum(xa, xb) + _XY_HALF) >> _XY_SHIFT
    c0, c1 = max(int(x_lo.min()), 0), min(int(x_hi.max()), w - 1)
    if c1 >= c0:  # the runs cover consecutive rows from r[0, 0]
        cols = np.arange(c0, c1 + 1)
        mask = (cols >= x_lo[:, None]) & (cols <= x_hi[:, None])
        canvas[int(r[0, 0]) : int(r[-1, 1]), c0 : c1 + 1][mask] = color


def _fill_ellipse(
    canvas: np.ndarray,
    centre: tuple[float, float],
    semi_major: float,
    semi_minor: float,
    angle: float,
    color: tuple[int, int, int],
) -> None:
    """Fill a rotated ellipse: pixels ``p`` with ``(u/a)^2 + (v/b)^2 <= 1``, where ``(u, v)`` is ``p - centre`` in
    the ellipse's axes."""
    a, b = max(semi_major, 0.5), max(semi_minor, 0.5)
    cx, cy = centre
    r = max(a, b)
    box = _box(cx, cy, r, r, canvas.shape[:2])
    if box is None:
        return
    x0, x1, y0, y1 = box
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dx, dy = xs - cx, ys - cy
    c, s = math.cos(angle), math.sin(angle)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    mask = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    canvas[y0:y1, x0:x1][mask] = color


def _fill_disc(canvas: np.ndarray, centre: tuple[float, float], radius: float, color: tuple[int, int, int]) -> None:
    """Fill a disc of *radius* around the centre truncated toward zero to an integer pixel, as the annotators'
    ``int()`` does (so a centre at -0.5, just off the canvas, is pixel 0)."""
    cx, cy = int(centre[0]), int(centre[1])
    box = _box(cx, cy, radius, radius, canvas.shape[:2])
    if box is None:
        return
    x0, x1, y0, y1 = box
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    canvas[y0:y1, x0:x1][mask] = color


def _draw_limb(canvas: np.ndarray, pa: np.ndarray, pb: np.ndarray, k: float, color: tuple[int, int, int]) -> None:
    """One limb, as the annotator draws it on its own grid (canvas coordinates divided by *k*): the midpoint, the
    half-length and the angle (from joint b to joint a, whole degrees) truncated toward zero, then
    ``cv2.fillConvexPoly(cv2.ellipse2Poly(...))``. At ``k == 1`` that polygon fill is reproduced exactly. At any
    other scale the annotator's drawing would be resampled, so puppet draws the ellipse that polygon fill covers,
    the exact one grown by half a grid pixel on both axes, scaled by *k* about the grid pixel's centre."""
    xa, ya, xb, yb = float(pa[0]) / k, float(pa[1]) / k, float(pb[0]) / k, float(pb[1]) / k
    cx, cy = int((xa + xb) / 2), int((ya + yb) / 2)
    half = int(math.hypot(xa - xb, ya - yb) / 2)
    angle = int(math.degrees(math.atan2(ya - yb, xa - xb)))
    if k == 1.0:
        _fill_convex_polygon(canvas, _ellipse_polygon(cx, cy, half, STICK_WIDTH, angle), color)
        return
    _fill_ellipse(
        canvas,
        ((cx + 0.5) * k - 0.5, (cy + 0.5) * k - 0.5),
        (half + 0.5) * k,
        (STICK_WIDTH + 0.5) * k,
        math.radians(angle),
        color,
    )


def _draw_body(canvas: np.ndarray, people: list[np.ndarray], min_score: float, k: float) -> None:
    bodies = [to_openpose18(kp) for kp in people]
    oks = [_confident(b, min_score, strict=True) for b in bodies]
    for i, (a, b) in enumerate(OPENPOSE_LIMBS):  # limb by limb, every person: the annotator's order
        for body, ok in zip(bodies, oks, strict=True):
            if ok[a] and ok[b]:
                _draw_limb(canvas, body[a, :2], body[b, :2], k, OPENPOSE_COLORS[i])
    # Dim the limb layer. Truncation, as an 8-bit multiply-and-cast does.
    canvas[:] = (canvas.astype(np.float64) * LIMB_DIM).astype(np.uint8)
    for j in range(len(OPENPOSE_COLORS)):  # joint by joint, every person
        for body, ok in zip(bodies, oks, strict=True):
            if ok[j]:
                _fill_disc(canvas, (body[j, 0], body[j, 1]), JOINT_RADIUS * k, OPENPOSE_COLORS[j])


def _clear_of_near_edges(points: np.ndarray, k: float) -> np.ndarray:
    """The annotator's near-origin test for hand and face points: it draws one only when its pixel coordinates,
    truncated to integers on its own grid (canvas coordinates divided by *k*), are both at least 1 (its ``eps =
    0.01`` is compared with that integer). So a point less than *k* px from the left or top edge, or off the canvas
    there, is dropped, with every hand edge touching it. Body joints have no such test."""
    return (points[:, :2] / k >= 1.0).all(axis=1)  # NaN compares False


def _draw_hand(canvas: np.ndarray, hand: np.ndarray, min_score: float, k: float) -> None:
    ok = _confident(hand, min_score) & _clear_of_near_edges(hand, k)
    width = max(1, round(HAND_LINE_WIDTH * k))
    edges = [(i, a, b) for i, (a, b) in enumerate(HAND_EDGES) if ok[a] and ok[b]]
    if edges:
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img)
        for i, a, b in edges:
            draw.line(
                [(float(hand[a, 0]), float(hand[a, 1])), (float(hand[b, 0]), float(hand[b, 1]))],
                fill=HAND_EDGE_COLORS[i],
                width=width,
            )
        canvas[:] = np.asarray(img)
    for j in np.flatnonzero(ok):
        _fill_disc(canvas, (hand[j, 0], hand[j, 1]), HAND_DOT_RADIUS * k, HAND_DOT_COLOR)


def _draw_face(canvas: np.ndarray, face: np.ndarray, min_score: float, k: float) -> None:
    ok = _confident(face, min_score) & _clear_of_near_edges(face, k)
    for j in np.flatnonzero(ok):
        _fill_disc(canvas, (face[j, 0], face[j, 1]), FACE_DOT_RADIUS * k, FACE_COLOR)


# ── public API ───────────────────────────────────────────────────────────────────────────────────────────────────


def render_array(
    pose: PoseSource,
    options: RenderOptions | None = None,
    *,
    source_size: tuple[int, int] | None = None,
    canvas_size: tuple[int, int] | None = None,
    primary_only: bool | None = None,
    fit: Fit | None = None,
    margin: float | None = None,
    stroke_scale: float | Literal["auto"] | None = None,
) -> np.ndarray:
    """The control image as an ``(height, width, 3)`` uint8 RGB array. Arguments as :func:`render_pose`."""
    opts = options or RenderOptions()
    primary_only = opts.primary_only if primary_only is None else primary_only
    fit = opts.fit if fit is None else fit
    margin = opts.margin if margin is None else margin
    stroke_scale = opts.stroke_scale if stroke_scale is None else stroke_scale
    people, src = _people_arrays(pose, source_size, primary_only)
    w, h = _canvas_size(src, opts, canvas_size)  # the module function: the argument shadows its name here
    if stroke_scale == "auto":
        k = min(w, h) / REFERENCE_RESOLUTION
    else:
        k = float(stroke_scale)
        if not (math.isfinite(k) and k > 0):
            raise ValueError(f"stroke_scale must be positive, got {stroke_scale!r}")
    points = _drawn_points(people, opts) if fit == "skeleton" else None
    t = canvas_transform(src, (w, h), fit, points=points, margin=margin)
    mapped = [t.apply(kp) for kp in people]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if opts.body:
        _draw_body(canvas, mapped, opts.min_score, k)
    if opts.hands:  # every left hand, then every right hand, as the annotator stacks them
        for hand in [kp[LEFT_HAND] for kp in mapped] + [kp[RIGHT_HAND] for kp in mapped]:
            _draw_hand(canvas, hand, opts.min_score, k)
    if opts.face:
        for kp in mapped:
            _draw_face(canvas, kp[FACE], opts.min_score, k)
    return canvas


def render_pose(
    pose: PoseSource,
    options: RenderOptions | None = None,
    *,
    source_size: tuple[int, int] | None = None,
    canvas_size: tuple[int, int] | None = None,
    primary_only: bool | None = None,
    fit: Fit | None = None,
    margin: float | None = None,
    stroke_scale: float | Literal["auto"] | None = None,
) -> Image.Image:
    """Draw an OpenPose-convention control image.

    *pose* is a :class:`PoseResult` (every person, or the primary only with *primary_only*), a
    :class:`PoseLibraryEntry`, or bare :class:`PersonPose` / ``(133, 3)`` arrays (one, a sequence, or an
    ``(N, 133, 3)`` stack), which need *source_size* ``(width, height)``: the keypoints' pixel frame. With bare
    people, *primary_only* keeps the one flagged primary (else the first).

    The canvas is *canvas_size* ``(width, height)`` if given, else ``options.width`` x ``options.height`` (one
    alone keeps the source aspect), else the source size. *fit* maps the source frame onto it (see the module
    docstring; *margin* applies to ``"skeleton"`` only). *stroke_scale* multiplies every line width and radius;
    ``"auto"`` is ``min(width, height) / 512``. *primary_only*, *fit*, *margin* and *stroke_scale* default to the
    *options* fields of the same name; a value given here overrides the option. ``stroke_scale`` defaults to
    ``"auto"``.
    """
    return Image.fromarray(
        render_array(
            pose,
            options,
            source_size=source_size,
            canvas_size=canvas_size,
            primary_only=primary_only,
            fit=fit,
            margin=margin,
            stroke_scale=stroke_scale,
        ),
        mode="RGB",
    )


def render_png(
    pose: PoseSource,
    options: RenderOptions | None = None,
    *,
    source_size: tuple[int, int] | None = None,
    canvas_size: tuple[int, int] | None = None,
    primary_only: bool | None = None,
    fit: Fit | None = None,
    margin: float | None = None,
    stroke_scale: float | Literal["auto"] | None = None,
) -> bytes:
    """:func:`render_pose` encoded as PNG bytes, with no metadata (same input, same bytes)."""
    img = render_pose(
        pose,
        options,
        source_size=source_size,
        canvas_size=canvas_size,
        primary_only=primary_only,
        fit=fit,
        margin=margin,
        stroke_scale=stroke_scale,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()
