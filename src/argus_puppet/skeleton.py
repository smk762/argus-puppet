"""The COCO-WholeBody 133 keypoint layout, its flip map and the OpenPose drawing convention.

Pure data plus small numpy helpers; no I/O and no model code. Every module that reads or writes keypoints indexes
them through the names here, so the layout is defined once.

Keypoints are image pixel coordinates with a score each: an array shaped ``(..., 133, 3)`` holding ``x, y, score``
(helpers that only move points accept ``(..., 133, >=2)``). The order is COCO-WholeBody's (Jin et al. 2020, as
mmpose names it): 17 body joints in COCO order, 6 foot points, 68 face landmarks in iBUG-300W order, then 21 points
per hand (root, then thumb, forefinger, middle, ring and pinky, four joints each, from the palm outwards).

The OpenPose constants below are the interoperability spec SDXL pose ControlNets read: the COCO-18 joint order,
the 17 drawn limbs, the 18 joint and limb colours, and the 20 hand edges.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

LAYOUT = "coco_wholebody_133"
NUM_KEYPOINTS = 133

# ── names ────────────────────────────────────────────────────────────────────────────────────────────────────────

BODY_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
FEET_NAMES: tuple[str, ...] = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)
FACE_NAMES: tuple[str, ...] = tuple(f"face-{i}" for i in range(68))
_FINGERS = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")
HAND_POINT_NAMES: tuple[str, ...] = ("hand_root", *(f"{f}{j}" for f in _FINGERS for j in range(1, 5)))  # 21
LEFT_HAND_NAMES: tuple[str, ...] = tuple(f"left_{n}" for n in HAND_POINT_NAMES)
RIGHT_HAND_NAMES: tuple[str, ...] = tuple(f"right_{n}" for n in HAND_POINT_NAMES)

KEYPOINT_NAMES: tuple[str, ...] = (
    *BODY_NAMES,
    *FEET_NAMES,
    *FACE_NAMES,
    *LEFT_HAND_NAMES,
    *RIGHT_HAND_NAMES,
)
KEYPOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# ── parts ────────────────────────────────────────────────────────────────────────────────────────────────────────

BODY = slice(0, 17)
FEET = slice(17, 23)
FACE = slice(23, 91)
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)
HAND_SIZE = 21
FACE_SIZE = 68

# The part names are the keys of models.PersonPose's per-part scores.
PARTS: dict[str, slice] = {
    "body": BODY,
    "feet": FEET,
    "face": FACE,
    "left_hand": LEFT_HAND,
    "right_hand": RIGHT_HAND,
}


def part_of(index: int) -> str:
    """The part (a key of :data:`PARTS`) keypoint *index* belongs to."""
    for name, part in PARTS.items():
        if part.start <= index < part.stop:
            return name
    raise IndexError(f"keypoint index {index} is outside 0..{NUM_KEYPOINTS - 1}")


class Joint(IntEnum):
    """Body (0-16) and foot (17-22) keypoint indices."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    LEFT_BIG_TOE = 17
    LEFT_SMALL_TOE = 18
    LEFT_HEEL = 19
    RIGHT_BIG_TOE = 20
    RIGHT_SMALL_TOE = 21
    RIGHT_HEEL = 22


# Short aliases for arithmetic on body joints (kp[L_SHOULDER]).
NOSE = Joint.NOSE
L_EYE, R_EYE = Joint.LEFT_EYE, Joint.RIGHT_EYE
L_EAR, R_EAR = Joint.LEFT_EAR, Joint.RIGHT_EAR
L_SHOULDER, R_SHOULDER = Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER
L_ELBOW, R_ELBOW = Joint.LEFT_ELBOW, Joint.RIGHT_ELBOW
L_WRIST, R_WRIST = Joint.LEFT_WRIST, Joint.RIGHT_WRIST
L_HIP, R_HIP = Joint.LEFT_HIP, Joint.RIGHT_HIP
L_KNEE, R_KNEE = Joint.LEFT_KNEE, Joint.RIGHT_KNEE
L_ANKLE, R_ANKLE = Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE
LEFT_HAND_ROOT, RIGHT_HAND_ROOT = LEFT_HAND.start, RIGHT_HAND.start

# ── limbs ────────────────────────────────────────────────────────────────────────────────────────────────────────

# The COCO-17 body skeleton (19 limbs), wholebody indices: legs, hips, torso sides, shoulders, arms, then the head.
BODY_LIMBS: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)
# Ankle to each foot point.
FEET_LIMBS: tuple[tuple[int, int], ...] = ((15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22))

# The 20 edges of one hand, as indices into its 21 points (0 = root); add LEFT_HAND.start or RIGHT_HAND.start.
HAND_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
LEFT_HAND_EDGES: tuple[tuple[int, int], ...] = tuple((a + LEFT_HAND.start, b + LEFT_HAND.start) for a, b in HAND_EDGES)
RIGHT_HAND_EDGES: tuple[tuple[int, int], ...] = tuple(
    (a + RIGHT_HAND.start, b + RIGHT_HAND.start) for a, b in HAND_EDGES
)

# ── horizontal flip ──────────────────────────────────────────────────────────────────────────────────────────────

# Face landmark pairs that swap under a horizontal flip, as indices into the 68 face points (iBUG-300W): jaw line,
# brows, nostrils, eyes, outer lips, inner lips. The nose ridge and the lips' centre line map to themselves.
_FACE_FLIP: tuple[tuple[int, int], ...] = (
    *((i, 16 - i) for i in range(8)),  # jaw 0-16, 8 is the chin
    *((17 + i, 26 - i) for i in range(5)),  # brows 17-26
    (31, 35),
    (32, 34),  # nostrils 31-35, 33 is the tip
    (36, 45),
    (37, 44),
    (38, 43),
    (39, 42),
    (40, 47),
    (41, 46),  # eyes 36-47
    (48, 54),
    (49, 53),
    (50, 52),
    (55, 59),
    (56, 58),  # outer lips 48-59
    (60, 64),
    (61, 63),
    (65, 67),  # inner lips 60-67
)

FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 20),
    (18, 21),
    (19, 22),
    *((FACE.start + a, FACE.start + b) for a, b in _FACE_FLIP),
    *((LEFT_HAND.start + i, RIGHT_HAND.start + i) for i in range(HAND_SIZE)),
)  # 61 pairs


def _flip_permutation() -> np.ndarray:
    perm = np.arange(NUM_KEYPOINTS, dtype=np.intp)
    for a, b in FLIP_PAIRS:
        perm[a], perm[b] = b, a
    perm.setflags(write=False)
    return perm


# flipped[..., i, :] = original[..., FLIP_PERMUTATION[i], :] once x is mirrored. An involution.
FLIP_PERMUTATION: np.ndarray = _flip_permutation()


def flip_horizontal(keypoints: np.ndarray, width: float) -> np.ndarray:
    """Mirror *keypoints* ``(..., 133, >=2)`` about the vertical centre line of an image *width* pixels wide.

    ``x`` becomes ``width - 1 - x`` (pixel centres at integers, as mmpose flips), and left and right points swap so
    a left wrist is still a left wrist. Other columns (``y``, the score) move with their point. Returns a new
    float array; NaN coordinates stay NaN.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim < 2 or kp.shape[-2] != NUM_KEYPOINTS or kp.shape[-1] < 2:
        raise ValueError(f"expected keypoints shaped (..., {NUM_KEYPOINTS}, >=2), got {kp.shape}")
    out = kp[..., FLIP_PERMUTATION, :].copy()
    out[..., 0] = (width - 1) - out[..., 0]
    return out


def part_mean_scores(keypoints: np.ndarray) -> dict[str, float]:
    """The mean score of each part (keys of :data:`PARTS`) of one person's ``(133, 3)`` keypoints."""
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.shape != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected keypoints shaped ({NUM_KEYPOINTS}, 3), got {kp.shape}")
    scores = np.nan_to_num(kp[:, 2], nan=0.0)
    return {name: float(scores[part].mean()) for name, part in PARTS.items()}


# ── OpenPose ─────────────────────────────────────────────────────────────────────────────────────────────────────

OPENPOSE18_NAMES: tuple[str, ...] = (
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
)
# The wholebody index each OpenPose-18 joint is read from; None is the neck, which wholebody lacks and
# :func:`to_openpose18` synthesises from the shoulders.
OPENPOSE18_FROM_WHOLEBODY: tuple[int | None, ...] = (0, None, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3)
OPENPOSE_NECK = 1

# The 17 limbs OpenPose draws, as 0-based OpenPose-18 indices (its limbSeq minus one). Limb i is drawn in
# OPENPOSE_COLORS[i]; joint j is drawn in OPENPOSE_COLORS[j].
OPENPOSE_LIMBS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)
# RGB.
OPENPOSE_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)


def to_openpose18(keypoints: np.ndarray) -> np.ndarray:
    """OpenPose-18 body joints ``(..., 18, 3)`` from wholebody keypoints ``(..., 133, 3)``.

    The neck is the midpoint of the shoulders, scored as the lower of their two scores.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim < 2 or kp.shape[-2:] != (NUM_KEYPOINTS, 3):
        raise ValueError(f"expected keypoints shaped (..., {NUM_KEYPOINTS}, 3), got {kp.shape}")
    out = np.zeros((*kp.shape[:-2], len(OPENPOSE18_NAMES), 3), dtype=np.float64)
    for j, i in enumerate(OPENPOSE18_FROM_WHOLEBODY):
        if i is None:
            shoulders = kp[..., [L_SHOULDER, R_SHOULDER], :]
            out[..., j, :2] = shoulders[..., :2].mean(axis=-2)
            out[..., j, 2] = shoulders[..., 2].min(axis=-1)
        else:
            out[..., j, :] = kp[..., i, :]
    return out
