"""The normalised pose vector of one person, and the mirror-invariant distance between two of them.

A pose vector is the 17 COCO body joints of one person, centred on the mid-hip and divided by a robust torso
scale, with every joint below ``min_joint_score`` masked out (never zero-filled). It is deliberately not
rotation-invariant: a standing and a lying person must stay far apart. It is mirror-invariant through the distance,
which takes the smaller of the distances to the other pose and to its left/right mirror image.

See ``docs/pose-vector.md`` for the maths and the reasoning behind each choice.

Everything here is pure numpy and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argus_puppet.skeleton import (
    BODY,
    FLIP_PERMUTATION,
    L_ANKLE,
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NUM_KEYPOINTS,
    R_ANKLE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)

# ── defaults ─────────────────────────────────────────────────────────────────────────────────────────────────────

# A joint counts as confident at or above this score (DWPose/RTMPose scores; 0.3 is mmpose's usual visibility cut).
DEFAULT_MIN_JOINT_SCORE = 0.3
# Confident body joints (of 17) a person needs to be embedded at all. 8 admits an upper-body shot with the hips in
# frame (head 5 + shoulders 2 + hips 2 = 9 before any limb) and refuses a head-and-shoulders portrait.
DEFAULT_MIN_USABLE_JOINTS = 8
# Joints that must be confident in both poses for a distance to be defined, and how many of them must be limb joints
# (elbows, wrists, knees, ankles). Without the limb rule, two poses sharing only head and torso would look alike
# whatever their arms and legs were doing.
DEFAULT_MIN_SHARED_JOINTS = 6
DEFAULT_MIN_SHARED_LIMB_JOINTS = 2

# ── the vector's joints and weights ──────────────────────────────────────────────────────────────────────────────

NUM_JOINTS = BODY.stop - BODY.start  # 17: the vector uses the body joints only (feet, hands and face stay out)

# Per-joint weight in the distance. The five head points (nose, eyes, ears) sit within a few centimetres of each
# other, so at full weight the head would count five times; at 0.4 each the head as a whole weighs 2 joints.
# Limbs get no extra weight: in torso units a wrist or ankle already moves several times as far as an elbow or knee
# for the same joint rotation, so the lever arm emphasises the extremities on its own.
HEAD_JOINT_WEIGHT = 0.4
JOINT_WEIGHTS: np.ndarray = np.array([HEAD_JOINT_WEIGHT] * 5 + [1.0] * 12, dtype=np.float64)
JOINT_WEIGHTS.setflags(write=False)

LIMB_JOINTS: tuple[int, ...] = (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE)
_IS_LIMB = np.zeros(NUM_JOINTS, dtype=bool)
_IS_LIMB[list(LIMB_JOINTS)] = True
# The 12 non-head body joints: shoulders, elbows, wrists, hips, knees, ankles (the whole limb skeleton). A pose
# comparison that shares all of them covers the whole body; one that does not is a partial comparison.
CORE_JOINTS: tuple[int, ...] = tuple(range(L_SHOULDER, NUM_JOINTS))
NUM_CORE_JOINTS = len(CORE_JOINTS)  # 12
_IS_CORE = np.zeros(NUM_JOINTS, dtype=bool)
_IS_CORE[list(CORE_JOINTS)] = True

# The flip permutation restricted to the body: body joints only ever swap with body joints.
BODY_FLIP: np.ndarray = np.asarray(FLIP_PERMUTATION[:NUM_JOINTS], dtype=np.intp)
BODY_FLIP.setflags(write=False)

# ── the torso scale ──────────────────────────────────────────────────────────────────────────────────────────────

# Lower bounds on the torso length (mid-shoulder to mid-hip) from widths that do not foreshorten with the torso. An
# adult's shoulder keypoints sit about 0.7 torso lengths apart and the hip keypoints about 0.45-0.55, so these
# multipliers stay below the true torso in an upright frontal pose (where the torso itself is the scale) and take
# over when the torso foreshortens (lying or bending toward the camera). Both widths shrink in profile, where the
# torso does not, so the maximum of the three is robust to either kind of foreshortening.
SHOULDER_WIDTH_TO_TORSO = 1.2
HIP_WIDTH_TO_TORSO = 1.5
# A torso scale under this many pixels is noise, not a person.
MIN_SCALE_PX = 4.0

# ── anchoring when one hip or one shoulder is not confident ──────────────────────────────────────────────────────

# An anchor joint (hip or shoulder) below min_joint_score still places the centre and the scale when its score is at
# least this: the estimator still put it somewhere sensible, and dropping it would move the centre by half a hip
# width the moment its score dips under the threshold (0.23 torso units in a frontal view). It stays masked as a
# compared joint. Below this floor its coordinate is treated as unplaced. A doubtful joint only completes a pair whose
# other joint is confident: a pose with no confident hip (or shoulder) is still unusable.
ANCHOR_MIN_SCORE = 0.1
# With the anchor joint gone altogether, it is estimated from the other pair: hip width ~ 0.7 x shoulder width. Adult
# hip keypoints sit 0.45-0.55 torso lengths apart and shoulder keypoints ~0.7 (a ratio of 0.64-0.79); the error of
# the estimate moves the centre by half the error in the width, a few hundredths of a torso unit.
HIP_TO_SHOULDER_WIDTH = 0.7
# Coordinates further than this from the origin (pixels) are treated as unplaced, as the renderer does: it keeps the
# float32 distance kernel in range.
MAX_COORD = 1e6


@dataclass(frozen=True)
class TorsoFrame:
    """The anchor of one person: where the mid-hip is, how big a torso is, and which way it points (pixels)."""

    mid_hip: np.ndarray  # (2,)
    mid_shoulder: np.ndarray  # (2,)
    torso_px: float  # |mid_shoulder - mid_hip| as seen, which foreshortens
    scale_px: float  # the robust torso scale every vector is divided by

    @property
    def up(self) -> np.ndarray:
        """Unit vector from the mid-hip toward the mid-shoulder (image axes, y down)."""
        v = self.mid_shoulder - self.mid_hip
        n = float(np.hypot(*v))
        return v / n if n > 0 else np.array([0.0, -1.0])

    @property
    def foreshortening(self) -> float:
        """torso_px / scale_px, in (0, 1]: 1 when the torso is the scale, small when it points at the camera."""
        return self.torso_px / self.scale_px


def _check(keypoints: np.ndarray) -> np.ndarray:
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected keypoints shaped ({NUM_KEYPOINTS}, 3), got {kp.shape}")
    return kp


def _placed(kp: np.ndarray) -> np.ndarray:
    """Finite coordinates within :data:`MAX_COORD` of the origin."""
    xy = kp[:, :2]
    return np.isfinite(xy).all(axis=1) & (np.abs(np.nan_to_num(xy, nan=np.inf)) < MAX_COORD).all(axis=1)


def confident(keypoints: np.ndarray, min_joint_score: float = DEFAULT_MIN_JOINT_SCORE) -> np.ndarray:
    """Boolean ``(133,)``: score at or above *min_joint_score* and finite coordinates within :data:`MAX_COORD`."""
    kp = _check(keypoints)
    return (kp[:, 2] >= min_joint_score) & _placed(kp)


def _anchor_points(kp: np.ndarray, min_joint_score: float) -> dict[int, np.ndarray] | None:
    """The four anchor joints (hips, shoulders) as the torso frame uses them, or ``None`` when it cannot be placed:
    without at least one confident hip and one confident shoulder.

    A joint is taken as it is when it is confident, or when its score is at least :data:`ANCHOR_MIN_SCORE` (a
    doubtful joint still marks where the body is). A joint missing below that is estimated from the other pair when
    that pair is whole (the missing hip is the present hip plus 0.7 x the shoulder vector on that side, the missing
    shoulder the present shoulder plus the hip vector / 0.7). Otherwise it is left out, and the one present joint of
    the pair stands in for its midpoint: right in profile, where the pair nearly coincides.
    """
    ok = confident(kp, min_joint_score)
    if not (ok[L_HIP] or ok[R_HIP]) or not (ok[L_SHOULDER] or ok[R_SHOULDER]):
        return None  # a doubtful joint completes a pair; it never anchors a pose on its own
    floor = min(min_joint_score, ANCHOR_MIN_SCORE)
    usable = (kp[:, 2] >= floor) & _placed(kp)
    pts = {j: kp[j, :2].copy() for j in (L_HIP, R_HIP, L_SHOULDER, R_SHOULDER) if usable[j]}
    pairs = ((L_HIP, R_HIP, L_SHOULDER, R_SHOULDER, HIP_TO_SHOULDER_WIDTH),)
    pairs += ((L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, 1.0 / HIP_TO_SHOULDER_WIDTH),)
    estimates: dict[int, np.ndarray] = {}
    for a, b, oa, ob, ratio in pairs:
        if oa in pts and ob in pts:
            if a in pts and b not in pts:
                estimates[b] = pts[a] + ratio * (pts[ob] - pts[oa])
            elif b in pts and a not in pts:
                estimates[a] = pts[b] + ratio * (pts[oa] - pts[ob])
    pts.update(estimates)
    if not ({L_HIP, R_HIP} & pts.keys()) or not ({L_SHOULDER, R_SHOULDER} & pts.keys()):
        return None
    return pts


def _mid(pts: dict[int, np.ndarray], a: int, b: int) -> np.ndarray:
    if a in pts and b in pts:
        return (pts[a] + pts[b]) / 2.0
    return (pts[a] if a in pts else pts[b]).copy()


def torso_frame(keypoints: np.ndarray, min_joint_score: float = DEFAULT_MIN_JOINT_SCORE) -> TorsoFrame | None:
    """The anchor and robust scale of one person's ``(133, 3)`` keypoints, or ``None`` when it cannot be placed.

    The centre is the midpoint of the hips and the mid-shoulder the midpoint of the shoulders, read from the anchor
    joints of :func:`_anchor_points`: a hip or shoulder scoring under *min_joint_score* still anchors when it scores
    :data:`ANCHOR_MIN_SCORE` or more, and one missing altogether is estimated from the other pair when that pair is
    whole. Both only ever complete a pair: a pose needs one confident hip and one confident shoulder to be placed at
    all. So a single anchor joint dipping under the threshold moves the vector by a few hundredths of a torso unit
    rather than half a hip width. With one hip (or shoulder) and no way to estimate the other, that one joint is the
    midpoint: right in profile, where the pair nearly coincides. The scale is::

        max(|mid_shoulder - mid_hip|, 1.2 * shoulder_width, 1.5 * hip_width)

    with a width counted when both of its anchor joints are known. ``None`` without a hip and a shoulder to anchor
    on (no confident hip or no confident shoulder), or when the scale is under :data:`MIN_SCALE_PX`.
    """
    kp = _check(keypoints)
    pts = _anchor_points(kp, min_joint_score)
    if pts is None:
        return None
    mid_hip = _mid(pts, L_HIP, R_HIP)
    mid_shoulder = _mid(pts, L_SHOULDER, R_SHOULDER)
    torso = float(np.hypot(*(mid_shoulder - mid_hip)))
    candidates = [torso]
    if L_SHOULDER in pts and R_SHOULDER in pts:
        candidates.append(SHOULDER_WIDTH_TO_TORSO * float(np.hypot(*(pts[L_SHOULDER] - pts[R_SHOULDER]))))
    if L_HIP in pts and R_HIP in pts:
        candidates.append(HIP_WIDTH_TO_TORSO * float(np.hypot(*(pts[L_HIP] - pts[R_HIP]))))
    scale = max(candidates)
    if not np.isfinite(scale) or scale < MIN_SCALE_PX:
        return None
    return TorsoFrame(mid_hip=mid_hip, mid_shoulder=mid_shoulder, torso_px=torso, scale_px=scale)


# ── the vector ───────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PoseVector:
    """One person's normalised body pose.

    ``coords`` is ``(17, 2)`` in torso units, centred on the mid-hip, image axes (x right, y down); a masked joint's
    coordinates are 0 and carry no meaning. ``mask`` is ``(17,)``, True where the joint was confident.
    """

    coords: np.ndarray
    mask: np.ndarray
    scale_px: float

    @property
    def joints(self) -> int:
        """Number of confident joints."""
        return int(self.mask.sum())


def embed(
    keypoints: np.ndarray,
    *,
    min_joint_score: float = DEFAULT_MIN_JOINT_SCORE,
    min_usable_joints: int = DEFAULT_MIN_USABLE_JOINTS,
) -> PoseVector | None:
    """The pose vector of one person's ``(133, 3)`` keypoints, or ``None`` when it is unusable.

    Unusable means: no hip or no shoulder to anchor on (see :func:`torso_frame`), a scale under
    :data:`MIN_SCALE_PX`, or fewer than *min_usable_joints* confident body joints.
    """
    kp = _check(keypoints)
    frame = torso_frame(kp, min_joint_score)
    if frame is None:
        return None
    mask = confident(kp, min_joint_score)[BODY].copy()
    if int(mask.sum()) < min_usable_joints:
        return None
    coords = (kp[BODY, :2] - frame.mid_hip) / frame.scale_px
    coords[~mask] = 0.0
    mask.setflags(write=False)
    coords.setflags(write=False)
    return PoseVector(coords=coords, mask=mask, scale_px=frame.scale_px)


def stack(vectors: list[PoseVector]) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 17, 2)`` coords and ``(N, 17)`` masks from a list of vectors (the input of the matrix functions)."""
    if not vectors:
        return np.zeros((0, NUM_JOINTS, 2)), np.zeros((0, NUM_JOINTS), dtype=bool)
    return np.stack([v.coords for v in vectors]), np.stack([v.mask for v in vectors])


# ── mirror and canonical orientation ─────────────────────────────────────────────────────────────────────────────


def mirror(coords: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The left/right mirror image of ``(..., 17, 2)`` coords and ``(..., 17)`` masks: x negated, left and right
    joints swapped through the skeleton's flip permutation. An involution."""
    c = np.asarray(coords, dtype=np.float64)[..., BODY_FLIP, :].copy()
    c[..., 0] = -c[..., 0]
    return c, np.asarray(mask, dtype=bool)[..., BODY_FLIP]


def mirror_vector(v: PoseVector) -> PoseVector:
    c, m = mirror(v.coords, v.mask)
    return PoseVector(coords=c, mask=m, scale_px=v.scale_px)


# |handedness| under this is treated as a tie and the next criterion decides.
_HANDEDNESS_EPS = 1e-9


def handedness(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A score that changes sign under :func:`mirror`: the weighted mean x of the confident joints, with the mean of
    ``x * y`` as the tie-break. Shape ``(...)``."""
    c = np.asarray(coords, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    w = JOINT_WEIGHTS * m
    total = np.maximum(w.sum(axis=-1), 1e-12)
    first = (w * c[..., 0]).sum(axis=-1) / total
    second = (w * c[..., 0] * c[..., 1]).sum(axis=-1) / total
    return np.where(np.abs(first) > _HANDEDNESS_EPS, first, second)


def canonicalize(coords: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic choice between each pose and its mirror image: the one whose :func:`handedness` is not
    negative (a pose with weight on its right side in the image is mirrored). Works on ``(17, 2)`` or batches.

    Mirror-invariant: ``canonicalize(*mirror(c, m))`` equals ``canonicalize(c, m)`` unless both criteria tie, which
    a real pose does only when it is (numerically) its own mirror image.
    """
    c = np.asarray(coords, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    mc, mm = mirror(c, m)
    flip = handedness(c, m) < -_HANDEDNESS_EPS
    return np.where(flip[..., None, None], mc, c), np.where(flip[..., None], mm, m)


def align_to_mean(coords: np.ndarray, mask: np.ndarray, *, iterations: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalise a set ``(N, 17, 2)`` jointly: start from :func:`canonicalize`, then repeatedly flip each pose to
    whichever orientation is nearer the set's masked mean pose. Order-independent and deterministic; this is what a
    centroid or a projection should average, since near-symmetric poses can sit either side of the per-pose rule."""
    c, m = canonicalize(coords, mask)
    if len(c) < 2:
        return c, m
    mc, mm = mirror(c, m)
    for _ in range(iterations):
        mean, mean_mask = masked_mean(c, m)
        d_keep = _to_template(c, m, mean, mean_mask)
        d_flip = _to_template(mc, mm, mean, mean_mask)
        flip = d_flip < d_keep - 1e-12
        if not flip.any():
            break
        c = np.where(flip[:, None, None], mc, c)
        m = np.where(flip[:, None], mm, m)
        mc, mm = mirror(c, m)
    return c, m


def masked_mean(coords: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint mean of ``(N, 17, 2)`` over the poses where the joint is confident, and which joints had any."""
    m = np.asarray(mask, dtype=bool)
    n = m.sum(axis=0)
    total = (np.asarray(coords) * m[..., None]).sum(axis=0)
    mean = np.where(n[:, None] > 0, total / np.maximum(n, 1)[:, None], 0.0)
    return mean, n > 0


def _to_template(c: np.ndarray, m: np.ndarray, t: np.ndarray, tm: np.ndarray) -> np.ndarray:
    shared = m & tm
    w = JOINT_WEIGHTS * shared
    d = np.linalg.norm(c - t, axis=-1)
    return (w * d).sum(axis=-1) / np.maximum(w.sum(axis=-1), 1e-12)


# ── distance ─────────────────────────────────────────────────────────────────────────────────────────────────────


def _oriented(
    a: np.ndarray,
    ma: np.ndarray,
    b: np.ndarray,
    mb: np.ndarray,
    min_shared: int,
    min_limb: int,
) -> np.ndarray:
    """Distances between every row of a ``(c, 17, 2)`` and every row of b ``(n, 17, 2)``, one orientation, NaN where
    too few joints are shared. ``(c, n)``."""
    fa = ma.astype(_WORK)
    fb = mb.astype(_WORK)
    # Shared-joint counts and weight totals are mask products: (c, J) @ (J, n).
    count = fa @ fb.T
    limbs = (fa * _IS_LIMB) @ fb.T
    wa = fa * JOINT_WEIGHTS.astype(_WORK)
    wsum = wa @ fb.T
    # Per-joint Euclidean distances (c, n, J), in place (np.hypot is several times slower).
    dist = a[:, None, :, 0] - b[None, :, :, 0]
    dy = a[:, None, :, 1] - b[None, :, :, 1]
    dist *= dist
    dy *= dy
    dist += dy
    np.sqrt(dist, out=dist)
    dist *= fb[None]  # only joints confident in b ...
    out = np.matmul(dist, wa[:, :, None])[..., 0]  # ... weighted by a's confident joints
    out /= np.maximum(wsum, 1e-12)
    out[(count < min_shared - 0.5) | (limbs < min_limb - 0.5)] = np.nan
    return out


def _block(
    a: np.ndarray,
    ma: np.ndarray,
    b: np.ndarray,
    mb: np.ndarray,
    bm: np.ndarray,
    mbm: np.ndarray,
    min_shared: int,
    min_limb: int,
) -> np.ndarray:
    d1 = _oriented(a, ma, b, mb, min_shared, min_limb)
    d2 = _oriented(a, ma, bm, mbm, min_shared, min_limb)
    return np.fmin(d1, d2)  # the smaller valid one; NaN only when neither orientation shares enough joints


def _core_shared(ma: np.ndarray, mb: np.ndarray, mbm: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    """Core joints shared in the orientation :func:`_block` chose (``(c, n)``, float)."""
    fa = (ma & _IS_CORE).astype(_WORK)
    c1 = fa @ mb.astype(_WORK).T
    c2 = fa @ mbm.astype(_WORK).T
    use2 = np.isnan(d1) | (d2 < d1)  # np.fmin's choice: d2 when d1 is NaN or d2 is smaller
    return np.where(use2, c2, c1)


@dataclass(frozen=True)
class Nearest:
    """Each query pose's nearest pose in a set (see :func:`nearest_neighbours`); ``-1`` / NaN where there is none.

    ``index``/``distance``/``shared`` are the nearest comparable pose, ``shared`` the core joints (of
    :data:`NUM_CORE_JOINTS`) the comparison shared. ``full_index``/``full_distance`` are the nearest pose among those
    sharing at least ``min_core_joints`` core joints, and ``full_shared`` the core joints that comparison shared.
    """

    index: np.ndarray  # (M,) intp
    distance: np.ndarray  # (M,) float64
    shared: np.ndarray  # (M,) intp
    full_index: np.ndarray  # (M,) intp
    full_distance: np.ndarray  # (M,) float64
    full_shared: np.ndarray  # (M,) intp


def nearest_neighbours(
    coords_a: np.ndarray,
    mask_a: np.ndarray,
    coords_b: np.ndarray,
    mask_b: np.ndarray,
    *,
    min_shared_joints: int = DEFAULT_MIN_SHARED_JOINTS,
    min_shared_limb_joints: int = DEFAULT_MIN_SHARED_LIMB_JOINTS,
    min_core_joints: int = NUM_CORE_JOINTS,
    exclude_self: bool = False,
) -> Nearest:
    """For each of M poses, its nearest of N poses under the mirror-invariant distance, and its nearest among those
    sharing at least *min_core_joints* of the 12 core joints with it (the whole limb skeleton by default).

    Ties go to the lowest index. *exclude_self* (a and b the same set) leaves each pose's own row out: the
    leave-one-out nearest neighbour. Computed in row chunks, so memory stays a few tens of MB and no M x N matrix
    is ever held.
    """
    a = np.asarray(coords_a, dtype=_WORK)
    ma = np.asarray(mask_a, dtype=bool)
    b = np.asarray(coords_b, dtype=_WORK)
    mb = np.asarray(mask_b, dtype=bool)
    if exclude_self and len(a) != len(b):
        raise ValueError("exclude_self needs the same set on both sides")
    m = len(a)
    out = Nearest(
        index=np.full(m, -1, dtype=np.intp),
        distance=np.full(m, np.nan),
        shared=np.zeros(m, dtype=np.intp),
        full_index=np.full(m, -1, dtype=np.intp),
        full_distance=np.full(m, np.nan),
        full_shared=np.zeros(m, dtype=np.intp),
    )
    if m == 0 or len(b) == 0:
        return out
    bm, mbm = mirror(b, mb)
    bm = bm.astype(_WORK)
    step = _rows_per_chunk(len(b))
    for i in range(0, m, step):
        j = slice(i, i + step)
        d1 = _oriented(a[j], ma[j], b, mb, min_shared_joints, min_shared_limb_joints)
        d2 = _oriented(a[j], ma[j], bm, mbm, min_shared_joints, min_shared_limb_joints)
        d = np.fmin(d1, d2)
        core = _core_shared(ma[j], mb, mbm, d1, d2)
        rows = np.arange(d.shape[0])
        if exclude_self:
            d[rows, rows + i] = np.nan
        dd = np.where(np.isfinite(d), d, np.inf)
        k = np.argmin(dd, axis=1)
        found = np.isfinite(dd[rows, k])
        out.index[j] = np.where(found, k, -1)
        out.distance[j] = np.where(found, d[rows, k], np.nan)
        out.shared[j] = np.where(found, np.rint(core[rows, k]), 0)
        dd[core < min_core_joints - 0.5] = np.inf
        k = np.argmin(dd, axis=1)
        found = np.isfinite(dd[rows, k])
        out.full_index[j] = np.where(found, k, -1)
        out.full_distance[j] = np.where(found, d[rows, k], np.nan)
        out.full_shared[j] = np.where(found, np.rint(core[rows, k]), 0)
    return out


def pose_distance(
    a: PoseVector,
    b: PoseVector,
    *,
    min_shared_joints: int = DEFAULT_MIN_SHARED_JOINTS,
    min_shared_limb_joints: int = DEFAULT_MIN_SHARED_LIMB_JOINTS,
) -> float:
    """The mirror-invariant distance between two poses, in torso units; NaN when they are not comparable.

    ``d(a, b) = min(D(a, b), D(a, mirror(b)))`` where ``D`` is the weighted mean, over the joints confident in both,
    of the Euclidean distance between corresponding joints. Not comparable (NaN) when, in both orientations, fewer
    than *min_shared_joints* joints, or fewer than *min_shared_limb_joints* limb joints, are confident in both.
    """
    out = cross_distances(
        a.coords[None],
        a.mask[None],
        b.coords[None],
        b.mask[None],
        min_shared_joints=min_shared_joints,
        min_shared_limb_joints=min_shared_limb_joints,
        dtype=np.float64,
    )
    return float(out[0, 0])


# Sizes one chunk to rows * cols * 34 <= 1.5M: its two (rows, cols, 17) float32 work arrays are ~3 MB each.
_CHUNK_ELEMENTS = 1_500_000
# The distance kernels work in float32: coordinates are O(1) torso units, so 7 significant digits are plenty, and
# the kernels are memory-bound, so halving the width roughly halves the time.
_WORK = np.float32


def _rows_per_chunk(cols: int) -> int:
    return max(1, _CHUNK_ELEMENTS // max(1, cols * NUM_JOINTS * 2))


def cross_distances(
    coords_a: np.ndarray,
    mask_a: np.ndarray,
    coords_b: np.ndarray,
    mask_b: np.ndarray,
    *,
    min_shared_joints: int = DEFAULT_MIN_SHARED_JOINTS,
    min_shared_limb_joints: int = DEFAULT_MIN_SHARED_LIMB_JOINTS,
    dtype: type[np.floating] = np.float64,
) -> np.ndarray:
    """``(M, N)`` mirror-invariant distances from each of M poses to each of N poses; NaN where not comparable.

    Computed in row chunks so the working memory stays a few tens of MB whatever M and N are.
    """
    a = np.asarray(coords_a, dtype=_WORK)
    ma = np.asarray(mask_a, dtype=bool)
    b = np.asarray(coords_b, dtype=_WORK)
    mb = np.asarray(mask_b, dtype=bool)
    bm, mbm = mirror(b, mb)
    bm = bm.astype(_WORK)
    out = np.empty((len(a), len(b)), dtype=dtype)
    if len(a) == 0 or len(b) == 0:
        return out
    step = _rows_per_chunk(len(b))
    for i in range(0, len(a), step):
        j = slice(i, i + step)
        out[j] = _block(a[j], ma[j], b, mb, bm, mbm, min_shared_joints, min_shared_limb_joints)
    return out


def pairwise_distances(
    coords: np.ndarray,
    mask: np.ndarray,
    *,
    min_shared_joints: int = DEFAULT_MIN_SHARED_JOINTS,
    min_shared_limb_joints: int = DEFAULT_MIN_SHARED_LIMB_JOINTS,
    dtype: type[np.floating] = np.float32,
) -> np.ndarray:
    """``(N, N)`` symmetric mirror-invariant distances between N poses, 0 on the diagonal, NaN where not comparable.

    Only the upper triangle is computed (in chunks) and it is copied to the lower one, so the result is exactly
    symmetric. float32 by default: N = 5000 is a 100 MB matrix.
    """
    c = np.asarray(coords, dtype=_WORK)
    m = np.asarray(mask, dtype=bool)
    n = len(c)
    out = np.zeros((n, n), dtype=dtype)
    if n < 2:
        return out
    cm, mm = mirror(c, m)
    cm = cm.astype(_WORK)
    i = 0
    while i < n:
        step = _rows_per_chunk(n - i)
        stop = min(n, i + step)
        block = _block(c[i:stop], m[i:stop], c[i:], m[i:], cm[i:], mm[i:], min_shared_joints, min_shared_limb_joints)
        k = stop - i
        square = np.triu(block[:, :k], k=1)
        out[i:stop, i:stop] = square + square.T  # the diagonal block, from its upper triangle only
        out[i:stop, stop:] = block[:, k:]
        out[stop:, i:stop] = block[:, k:].T
        i = stop
    # A pose shares all its joints with itself: the diagonal is 0 even for a pose too sparse to compare.
    np.fill_diagonal(out, 0.0)
    return out
