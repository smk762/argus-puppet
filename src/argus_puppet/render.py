"""OpenPose-convention skeleton control images for pose ControlNets.

SD/SDXL pose ControlNets were trained on skeletons drawn in one fixed colour convention (the one the OpenPose and
DWPose annotators produce). This module draws that convention from scratch with numpy and Pillow, treating it as an
interoperability spec:

* a black RGB canvas;
* **body**: the wholebody keypoints become OpenPose-18 (:func:`skeleton.to_openpose18`, the neck synthesised from
  the shoulders). Each of the 17 :data:`skeleton.OPENPOSE_LIMBS` whose two joints are both confident is a filled
  ellipse centred on the limb's midpoint, semi-major half the limb's length, semi-minor :data:`STICK_WIDTH`, in
  ``OPENPOSE_COLORS[i]``. Once every limb of every person is drawn the whole limb layer is multiplied by 0.6
  (:data:`LIMB_DIM`); then each confident joint ``j`` is a filled disc of radius :data:`JOINT_RADIUS` in
  ``OPENPOSE_COLORS[j]`` at full intensity;
* **hands**: per hand, the 20 :data:`skeleton.HAND_EDGES`; edge ``i`` is a :data:`HAND_LINE_WIDTH` px line in
  ``hsv(i / 20, 1, 1)``, drawn when both ends are confident; then each confident hand point is a radius
  :data:`HAND_DOT_RADIUS` disc in pure blue;
* **face**: each confident face landmark (all 68) is a white disc of radius :data:`FACE_DOT_RADIUS`;
* feet are never drawn, and a point scoring below ``min_score`` is dropped, as is one with a non-finite coordinate
  or one more than 1e6 px from the origin (treated as unplaced; it keeps the pixel maths in range).

Draw order is body limbs, the 0.6 dimming, body joints, then each person's hands (edges, then dots) and face. Later
shapes overwrite earlier ones; nothing is blended. Shapes are aliased (hard edges), as the reference annotators draw
them, so every pixel is either black or one of the convention's colours.

**Sizing.** Keypoints live in their source image's pixel frame. The canvas is the source size unless a size is
asked for. ``fit`` decides how the source frame maps onto a canvas of another shape:

* ``"contain"`` (default): one uniform scale so the whole source frame fits, centred (letterboxed). The skeleton is
  never distorted.
* ``"cover"``: one uniform scale so the source frame fills the canvas, centred; parts may fall outside.
* ``"stretch"``: independent x and y scales (distorts the skeleton unless the aspect ratios match).
* ``"skeleton"``: ignore the source frame and fit the drawn points' bounding box into the canvas with a margin,
  uniformly scaled and centred. Used to re-frame library poses for a batch.

A point maps as ``x' = x * scale + offset``. **Line widths and radii are fixed pixel sizes** (``stroke_scale=1``):
4 px stick half-width, radius-4 joints, 2 px hand lines, radius-3 face dots, whatever the canvas size, as the
annotators draw at their working resolution. The annotators usually draw at a ~512 px detect resolution and the
result is resized to the output, so on large canvases their strokes look proportionally thicker; pass
``stroke_scale="auto"`` (``min(width, height) / 512``) to mimic that.
"""

from __future__ import annotations

import colorsys
import io
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from argus_puppet.models import PersonPose, PoseLibraryEntry, PoseResult, RenderOptions
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

Fit = Literal["contain", "cover", "stretch", "skeleton"]
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


def _resolve_canvas(
    source_size: tuple[int, int], options: RenderOptions, canvas_size: tuple[int, int] | None
) -> tuple[int, int]:
    if canvas_size is not None:
        w, h = int(canvas_size[0]), int(canvas_size[1])
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


def _confident(points: np.ndarray, min_score: float) -> np.ndarray:
    """Mask of points that are drawn: finite, sane coordinates and a score at or above *min_score*."""
    xy = np.nan_to_num(points[:, :2], nan=np.inf)
    placed = (np.abs(xy) < _MAX_COORD).all(axis=1)
    return placed & (np.nan_to_num(points[:, 2], nan=-1.0) >= min_score)


def _drawn_points(arrays: list[np.ndarray], options: RenderOptions) -> np.ndarray:
    """Every point the options would draw, in source coordinates (for ``fit="skeleton"``)."""
    out: list[np.ndarray] = []
    for kp in arrays:
        parts: list[np.ndarray] = []
        if options.body:
            parts.append(to_openpose18(kp))
        if options.hands:
            parts += [kp[LEFT_HAND], kp[RIGHT_HAND]]
        if options.face:
            parts.append(kp[FACE])
        for p in parts:
            out.append(p[_confident(p, options.min_score)])
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
    """Fill a disc of *radius* around the pixel the centre falls in (the centre is truncated to an integer pixel,
    as the annotators do)."""
    cx, cy = math.floor(centre[0]), math.floor(centre[1])
    box = _box(cx, cy, radius, radius, canvas.shape[:2])
    if box is None:
        return
    x0, x1, y0, y1 = box
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
    canvas[y0:y1, x0:x1][mask] = color


def _draw_body(canvas: np.ndarray, people: list[np.ndarray], min_score: float, k: float) -> None:
    bodies = [to_openpose18(kp) for kp in people]
    oks = [_confident(b, min_score) for b in bodies]
    stick = STICK_WIDTH * k
    for body, ok in zip(bodies, oks, strict=True):
        for i, (a, b) in enumerate(OPENPOSE_LIMBS):
            if not (ok[a] and ok[b]):
                continue
            (xa, ya), (xb, yb) = body[a, :2], body[b, :2]
            length = math.hypot(xb - xa, yb - ya)
            _fill_ellipse(
                canvas,
                ((xa + xb) / 2, (ya + yb) / 2),
                length / 2,
                stick,
                math.atan2(yb - ya, xb - xa),
                OPENPOSE_COLORS[i],
            )
    # Dim the limb layer. Truncation, as an 8-bit multiply-and-cast does.
    canvas[:] = (canvas.astype(np.float64) * LIMB_DIM).astype(np.uint8)
    for body, ok in zip(bodies, oks, strict=True):
        for j in range(len(body)):
            if ok[j]:
                _fill_disc(canvas, (body[j, 0], body[j, 1]), JOINT_RADIUS * k, OPENPOSE_COLORS[j])


def _draw_hand(canvas: np.ndarray, hand: np.ndarray, min_score: float, k: float) -> None:
    ok = _confident(hand, min_score)
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
    ok = _confident(face, min_score)
    for j in np.flatnonzero(ok):
        _fill_disc(canvas, (face[j, 0], face[j, 1]), FACE_DOT_RADIUS * k, FACE_COLOR)


# ── public API ───────────────────────────────────────────────────────────────────────────────────────────────────


def render_array(
    pose: PoseSource,
    options: RenderOptions | None = None,
    *,
    source_size: tuple[int, int] | None = None,
    canvas_size: tuple[int, int] | None = None,
    primary_only: bool = False,
    fit: Fit = "contain",
    margin: float = _DEFAULT_MARGIN,
    stroke_scale: float | Literal["auto"] = 1.0,
) -> np.ndarray:
    """The control image as an ``(height, width, 3)`` uint8 RGB array. Arguments as :func:`render_pose`."""
    opts = options or RenderOptions()
    people, src = _people_arrays(pose, source_size, primary_only)
    w, h = _resolve_canvas(src, opts, canvas_size)
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
    for kp in mapped:
        if opts.hands:
            _draw_hand(canvas, kp[LEFT_HAND], opts.min_score, k)
            _draw_hand(canvas, kp[RIGHT_HAND], opts.min_score, k)
        if opts.face:
            _draw_face(canvas, kp[FACE], opts.min_score, k)
    return canvas


def render_pose(
    pose: PoseSource,
    options: RenderOptions | None = None,
    *,
    source_size: tuple[int, int] | None = None,
    canvas_size: tuple[int, int] | None = None,
    primary_only: bool = False,
    fit: Fit = "contain",
    margin: float = _DEFAULT_MARGIN,
    stroke_scale: float | Literal["auto"] = 1.0,
) -> Image.Image:
    """Draw an OpenPose-convention control image.

    *pose* is a :class:`PoseResult` (every person, or the primary only with *primary_only*), a
    :class:`PoseLibraryEntry`, or bare :class:`PersonPose` / ``(133, 3)`` arrays (one, a sequence, or an
    ``(N, 133, 3)`` stack), which need *source_size* ``(width, height)``: the keypoints' pixel frame. With bare
    people, *primary_only* keeps the one flagged primary (else the first).

    The canvas is *canvas_size* ``(width, height)`` if given, else ``options.width`` x ``options.height`` (one
    alone keeps the source aspect), else the source size. *fit* maps the source frame onto it (see the module
    docstring; *margin* applies to ``"skeleton"`` only). *stroke_scale* multiplies every line width and radius;
    ``"auto"`` is ``min(width, height) / 512``.
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
    primary_only: bool = False,
    fit: Fit = "contain",
    margin: float = _DEFAULT_MARGIN,
    stroke_scale: float | Literal["auto"] = 1.0,
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
