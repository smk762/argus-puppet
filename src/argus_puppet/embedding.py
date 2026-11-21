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


def confident(keypoints: np.ndarray, min_joint_score: float = DEFAULT_MIN_JOINT_SCORE) -> np.ndarray:
    """Boolean ``(133,)``: score at or above *min_joint_score* and finite coordinates."""
    kp = _check(keypoints)
    return (kp[:, 2] >= min_joint_score) & np.isfinite(kp[:, :2]).all(axis=1)


def _midpoint(kp: np.ndarray, ok: np.ndarray, a: int, b: int) -> np.ndarray | None:
    if ok[a] and ok[b]:
        return (kp[a, :2] + kp[b, :2]) / 2.0
    if ok[a]:
        return kp[a, :2].copy()
    if ok[b]:
        return kp[b, :2].copy()
    return None


def torso_frame(keypoints: np.ndarray, min_joint_score: float = DEFAULT_MIN_JOINT_SCORE) -> TorsoFrame | None:
    """The anchor and robust scale of one person's ``(133, 3)`` keypoints, or ``None`` when it cannot be placed.

    The centre is the midpoint of the hips, or the one confident hip (in profile the hips nearly coincide, so one is
    a good centre; in a frontal view both are normally visible). The mid-shoulder likewise. The scale is::

        max(|mid_shoulder - mid_hip|, 1.2 * shoulder_width, 1.5 * hip_width)

    with a width only counted when both of its joints are confident. ``None`` without at least one confident hip and
    one confident shoulder, or when the scale is under :data:`MIN_SCALE_PX`.
    """
    kp = _check(keypoints)
    ok = confident(kp, min_joint_score)
    mid_hip = _midpoint(kp, ok, L_HIP, R_HIP)
    mid_shoulder = _midpoint(kp, ok, L_SHOULDER, R_SHOULDER)
    if mid_hip is None or mid_shoulder is None:
        return None
    torso = float(np.hypot(*(mid_shoulder - mid_hip)))
    candidates = [torso]
    if ok[L_SHOULDER] and ok[R_SHOULDER]:
        candidates.append(SHOULDER_WIDTH_TO_TORSO * float(np.hypot(*(kp[L_SHOULDER, :2] - kp[R_SHOULDER, :2]))))
    if ok[L_HIP] and ok[R_HIP]:
        candidates.append(HIP_WIDTH_TO_TORSO * float(np.hypot(*(kp[L_HIP, :2] - kp[R_HIP, :2]))))
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

    Unusable means: no confident hip or no confident shoulder (there is no anchor), a scale under
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
