"""Coarse pose labels (stance, arms, facing) from keypoint geometry. Rules, not a classifier.

Every rule reads confident body joints only, measures lengths in torso units (:func:`embedding.torso_frame`'s robust
scale) and angles in degrees, so the labels do not change with translation or uniform scale. Stance and arms are
unchanged by a left/right mirror; facing swaps ``left`` and ``right`` and keeps ``front`` and ``back``.

What one 2D frame can decide is limited, and the rules say ``unknown`` rather than guess:

* **Gravity is assumed to point down the image** (a roughly level camera). Stance reads the torso's angle to the
  image vertical and the legs' angles to "down".
* **Jumping has no ground reference.** Only two silhouettes count: a *tuck* (upright torso, both knees clearly above
  the hips with the feet hanging below them) and a *star* (upright torso, both legs straight and splayed to opposite
  sides, both arms raised or out). A person striking a star pose on the ground reads as jumping; a jump with ordinary
  running legs reads as standing or unknown.
* **Standing needs a whole leg** (hip, knee and ankle) with a visible shin: a straight vertical thigh without its ankle, or with a short shin, could be kneeling
  upright, so it is unknown.

The thresholds are named constants below, each with the reasoning behind it; ``docs/metrics.md`` tabulates them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argus_puppet.embedding import DEFAULT_MIN_JOINT_SCORE, TorsoFrame, confident, torso_frame
from argus_puppet.models import Arms, Facing, PoseLabels, Stance
from argus_puppet.skeleton import (
    FACE,
    L_ANKLE,
    L_EAR,
    L_EYE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_EAR,
    R_EYE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)

# ── stance thresholds ────────────────────────────────────────────────────────────────────────────────────────────

# Torso (mid-hip -> mid-shoulder) angle to the image's "up". Beyond 60 degrees a torso is closer to horizontal than
# to vertical (60 rather than 45 leaves room for a strong lean or a recline that is still upright stance).
TORSO_LYING_MIN_DEG = 60.0
# Past 135 degrees the shoulders are well below the hips (a handstand, a hang): none of the labels fit.
TORSO_INVERTED_DEG = 135.0
# Jump silhouettes need an upright torso; a tuck with the torso leaning further is as likely a squat.
TORSO_UPRIGHT_DEG = 25.0
# Visible torso length / robust scale under this means the torso points at the camera: its angle is meaningless.
TORSO_FORESHORTENED = 0.5

# A thigh (hip -> knee) within 35 degrees of straight down is a standing thigh.
THIGH_VERTICAL_MAX_DEG = 35.0
# A thigh at least 55 degrees from down is near horizontal (seated, squatting, or raised).
THIGH_RAISED_MIN_DEG = 55.0
# A thigh whose visible length is under 0.45 torso units points at the camera (an adult thigh is ~0.9 torso units).
THIGH_FORESHORTENED = 0.45
# A standing thigh shows at least 0.55 torso units: shorter and it is coming toward the camera (seated, kneeling).
THIGH_STANDING_MIN = 0.55
# A standing shin shows at least 0.5 torso units (an adult shin is ~0.85): shorter and it points away from or toward
# the camera, as in a frontal view of someone kneeling upright, whose straight-looking leg is not standing.
SHIN_STANDING_MIN = 0.5

# Interior knee angle (hip-knee-ankle): 180 is a straight leg.
KNEE_STRAIGHT_MIN_DEG = 150.0  # straight enough to stand on
KNEE_BENT_MAX_DEG = 110.0  # clearly flexed (crouch, tuck)
KNEE_SEATED_MIN_DEG = 70.0  # a chair sit bends the knee to ~90; a full squat folds it under ~70
KNEE_LEGS_OUT_MIN_DEG = 140.0  # sitting on the floor with the legs stretched out

# Heights in torso units, positive upwards for knees and downwards for ankles.
TUCK_KNEE_RISE_MIN = 0.25  # tuck: knees at least a quarter torso above the hips (a chair sit keeps them level)
TUCK_ANKLE_DROP_MIN = 0.2  # tuck: the feet still hang clearly below the hips, under knees raised above them
SEATED_KNEE_RISE_MAX = 0.25  # chair sit: knees no higher than this above the hips
SEATED_SHIN_DROP_MIN = 0.4  # chair sit: the ankle hangs at least this far below the knee
SEATED_ANKLE_DROP_MIN = 0.6  # chair sit: the feet are a seat height (a shin, ~0.85) below the hips
LEGS_OUT_ANKLE_DROP_MAX = 0.5  # floor sit: the feet are near hip height
FRONTAL_SEAT_KNEE_RISE_MAX = 0.1  # thighs at the camera: knee at or below hip level
FRONTAL_SEAT_SHIN_DROP_MIN = 0.5  # ... with a visible shin hanging from it
FRONTAL_SEAT_ANKLE_DROP_MIN = 0.7  # ... and the feet a seat height below the hips
CROUCH_ANKLE_DROP_MAX = 0.8  # crouching: hips within 0.8 torso units of the ankles (standing is ~1.75)

# Star jump: each leg (hip -> ankle) at least 15 degrees out from down, ankles 0.9 torso units apart or more.
STAR_LEG_SPLAY_MIN_DEG = 15.0
STAR_ANKLE_SPREAD_MIN = 0.9

# ── arm thresholds (in the torso's frame: "up" is mid-hip -> mid-shoulder) ──────────────────────────────────────

# Wrist at least a quarter torso above its shoulder: the arm is raised.
ARM_UP_RISE_MIN = 0.25
# Wrist at least 0.6 torso units out to the side (a fully extended arm reaches ~1.05) and not more than half a torso
# below the shoulder: the arm is out.
ARM_OUT_LATERAL_MIN = 0.6
ARM_OUT_RISE_MIN = -0.5
# Anything else is "down": hanging, hands on hips, folded across the body, or a hand at the face.

# ── facing thresholds ────────────────────────────────────────────────────────────────────────────────────────────

# Shoulder width / torso scale under this is a profile: frontal adults show ~0.7, a three-quarter view ~0.5.
PROFILE_SHOULDER_RATIO_MAX = 0.35
# In profile the nose sits ahead of the ears (or the shoulders) by at least this many torso units, horizontally.
PROFILE_NOSE_OFFSET_MIN = 0.05
# The face is visible when the nose and an eye are confident, or the 68 face landmarks average this score or more.
FACE_VISIBLE_MEAN_SCORE = DEFAULT_MIN_JOINT_SCORE


@dataclass(frozen=True)
class _Leg:
    thigh_deg: float  # hip -> knee, degrees from straight down
    thigh_len: float  # torso units, as seen
    knee_rise: float  # (hip_y - knee_y) / scale: positive when the knee is above the hip
    knee_deg: float | None  # interior hip-knee-ankle angle; None without the ankle
    ankle_drop: float | None  # (ankle_y - hip_y) / scale: positive when the foot is below the hip
    shin_drop: float | None  # (ankle_y - knee_y) / scale
    shin_len: float | None  # torso units, as seen
    splay_deg: float | None  # hip -> ankle, degrees from down
    ankle_x: float | None  # pixels


def _deg_from(v: np.ndarray, ref: np.ndarray) -> float:
    n = float(np.hypot(*v)) * float(np.hypot(*ref))
    if n == 0.0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v, ref) / n, -1.0, 1.0))))


_DOWN = np.array([0.0, 1.0])
_UP = np.array([0.0, -1.0])


def _leg(kp: np.ndarray, ok: np.ndarray, hip: int, knee: int, ankle: int, scale: float) -> _Leg | None:
    if not (ok[hip] and ok[knee]):
        return None
    h, k = kp[hip, :2], kp[knee, :2]
    thigh = k - h
    knee_deg = ankle_drop = shin_drop = shin_len = splay = ankle_x = None
    if ok[ankle]:
        a = kp[ankle, :2]
        # A zero-length thigh or shin has no angle; call the leg straight rather than folded.
        knee_deg = _deg_from(h - k, a - k) if np.hypot(*(a - k)) > 0 and np.hypot(*(h - k)) > 0 else 180.0
        ankle_drop = float(a[1] - h[1]) / scale
        shin_drop = float(a[1] - k[1]) / scale
        shin_len = float(np.hypot(*(a - k))) / scale
        splay = _deg_from(a - h, _DOWN)
        ankle_x = float(a[0])
    return _Leg(
        thigh_deg=_deg_from(thigh, _DOWN),
        thigh_len=float(np.hypot(*thigh)) / scale,
        knee_rise=float(h[1] - k[1]) / scale,
        knee_deg=knee_deg,
        ankle_drop=ankle_drop,
        shin_drop=shin_drop,
        shin_len=shin_len,
        splay_deg=splay,
        ankle_x=ankle_x,
    )


def _arm_class(kp: np.ndarray, ok: np.ndarray, shoulder: int, wrist: int, frame: TorsoFrame) -> str | None:
    if not (ok[shoulder] and ok[wrist]):
        return None
    up = frame.up
    perp = np.array([-up[1], up[0]])
    v = kp[wrist, :2] - kp[shoulder, :2]
    rise = float(np.dot(v, up)) / frame.scale_px
    lateral = abs(float(np.dot(v, perp))) / frame.scale_px
    if rise >= ARM_UP_RISE_MIN:
        return "up"
    if lateral >= ARM_OUT_LATERAL_MIN and rise >= ARM_OUT_RISE_MIN:
        return "out"
    return "down"


def _arms(kp: np.ndarray, ok: np.ndarray, frame: TorsoFrame | None) -> tuple[Arms, list[str]]:
    if frame is None:
        return "unknown", []
    sides = ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST))
    classes = [c for c in (_arm_class(kp, ok, s, w, frame) for s, w in sides) if c is not None]
    if not classes:
        return "unknown", []
    if len(set(classes)) > 1:
        return "mixed", classes
    return classes[0], classes  # type: ignore[return-value]


def _stance(kp: np.ndarray, ok: np.ndarray, frame: TorsoFrame | None, arm_classes: list[str]) -> Stance:
    if frame is None or frame.foreshortening < TORSO_FORESHORTENED:
        return "unknown"
    s = frame.scale_px
    torso_deg = _deg_from(frame.mid_shoulder - frame.mid_hip, _UP)
    legs = [g for g in (_leg(kp, ok, L_HIP, L_KNEE, L_ANKLE, s), _leg(kp, ok, R_HIP, R_KNEE, R_ANKLE, s)) if g]
    full = [g for g in legs if g.knee_deg is not None]

    def standing_leg(g: _Leg) -> bool:
        return (
            g.thigh_deg <= THIGH_VERTICAL_MAX_DEG
            and g.thigh_len >= THIGH_STANDING_MIN
            and (g.knee_deg is None or (g.knee_deg >= KNEE_STRAIGHT_MIN_DEG and g.shin_len >= SHIN_STANDING_MIN))  # type: ignore[operator]
        )

    if torso_deg > TORSO_INVERTED_DEG:
        return "unknown"
    if torso_deg >= TORSO_LYING_MIN_DEG:
        # A horizontal torso on straight vertical legs is someone bending over, which is a standing stance.
        if full and all(standing_leg(g) for g in legs):
            return "standing"
        return "lying"
    if not full:
        return "unknown"  # no whole leg: an upper-body shot, or a thigh that could be kneeling or standing

    upright = torso_deg <= TORSO_UPRIGHT_DEG
    both = len(full) == 2
    if upright and both:
        a, b = full
        star = (
            all(g.knee_deg >= KNEE_STRAIGHT_MIN_DEG for g in full)  # type: ignore[operator]
            and all(g.splay_deg >= STAR_LEG_SPLAY_MIN_DEG for g in full)  # type: ignore[operator]
            and abs(a.ankle_x - b.ankle_x) >= STAR_ANKLE_SPREAD_MIN * s  # type: ignore[operator]
            and (a.ankle_x - frame.mid_hip[0]) * (b.ankle_x - frame.mid_hip[0]) < 0  # type: ignore[operator]
            and len(arm_classes) == 2
            and all(c in ("up", "out") for c in arm_classes)
        )
        tuck = all(
            g.knee_rise >= TUCK_KNEE_RISE_MIN
            and g.knee_deg <= KNEE_BENT_MAX_DEG  # type: ignore[operator]
            and g.ankle_drop >= TUCK_ANKLE_DROP_MIN  # type: ignore[operator]
            for g in full
        )
        if star or tuck:
            return "jumping"

    if all(standing_leg(g) for g in legs):
        return "standing"

    def seated_leg(g: _Leg) -> bool:
        if g.knee_deg is None:
            return g.thigh_deg >= THIGH_RAISED_MIN_DEG or g.thigh_len < THIGH_FORESHORTENED
        chair = (
            g.thigh_deg >= THIGH_RAISED_MIN_DEG
            and g.knee_rise < SEATED_KNEE_RISE_MAX
            and g.knee_deg >= KNEE_SEATED_MIN_DEG
            and g.shin_drop >= SEATED_SHIN_DROP_MIN  # type: ignore[operator]
            and g.ankle_drop >= SEATED_ANKLE_DROP_MIN  # type: ignore[operator]
        )
        floor = (
            g.thigh_deg >= THIGH_RAISED_MIN_DEG
            and g.knee_deg >= KNEE_LEGS_OUT_MIN_DEG
            and abs(g.ankle_drop) <= LEGS_OUT_ANKLE_DROP_MAX  # type: ignore[arg-type]
        )
        frontal = (
            g.thigh_len < THIGH_FORESHORTENED
            and g.knee_rise <= FRONTAL_SEAT_KNEE_RISE_MAX
            and g.shin_drop >= FRONTAL_SEAT_SHIN_DROP_MIN  # type: ignore[operator]
            and g.ankle_drop >= FRONTAL_SEAT_ANKLE_DROP_MIN  # type: ignore[operator]
        )
        return chair or floor or frontal

    if all(seated_leg(g) for g in legs):
        return "seated"
    if all(g.knee_deg <= KNEE_BENT_MAX_DEG and g.ankle_drop < CROUCH_ANKLE_DROP_MAX for g in full):  # type: ignore[operator]
        return "crouching"
    return "unknown"


def _facing(kp: np.ndarray, ok: np.ndarray, frame: TorsoFrame | None) -> Facing:
    if frame is None:
        return "unknown"
    s = frame.scale_px
    # Shoulder positions across the body (perpendicular to the torso axis; the image's x when upright), so a lying
    # subject's shoulders are compared side to side rather than along the image row.
    up = frame.up
    across = np.array([-up[1], up[0]])
    l_across = float(np.dot(kp[L_SHOULDER, :2], across))
    r_across = float(np.dot(kp[R_SHOULDER, :2], across))
    both_shoulders = bool(ok[L_SHOULDER] and ok[R_SHOULDER])
    one_shoulder = bool(ok[L_SHOULDER]) != bool(ok[R_SHOULDER])
    width = abs(l_across - r_across) if both_shoulders else 0.0
    # One shoulder alone is also what a frontal shot cropped at one side looks like: a face seen from the front
    # (both eyes confident, the nose, if confident, between them across the body) rules the profile out.
    frontal_face = bool(ok[L_EYE] and ok[R_EYE])
    if frontal_face and ok[NOSE]:
        eyes = sorted((float(np.dot(kp[L_EYE, :2], across)), float(np.dot(kp[R_EYE, :2], across))))
        frontal_face = eyes[0] <= float(np.dot(kp[NOSE, :2], across)) <= eyes[1]
    profile = (one_shoulder and not frontal_face) or (both_shoulders and width / s < PROFILE_SHOULDER_RATIO_MAX)
    if profile:
        # "Left" and "right" are the image's; they only mean something for a roughly upright subject.
        if not ok[NOSE] or _deg_from(frame.mid_shoulder - frame.mid_hip, _UP) >= TORSO_LYING_MIN_DEG:
            return "unknown"
        ears = [kp[j, 0] for j in (L_EAR, R_EAR) if ok[j]]
        ref = float(np.mean(ears)) if ears else float(frame.mid_shoulder[0])
        dx = (float(kp[NOSE, 0]) - ref) / s
        if dx <= -PROFILE_NOSE_OFFSET_MIN:
            return "left"
        if dx >= PROFILE_NOSE_OFFSET_MIN:
            return "right"
        return "unknown"
    if not both_shoulders:
        return "unknown"
    face_visible = bool(ok[NOSE] and (ok[L_EYE] or ok[R_EYE])) or float(kp[FACE, 2].mean()) >= FACE_VISIBLE_MEAN_SCORE
    # Seen from the front, the subject's left shoulder is on the image's right; from behind, on its left.
    left_on_right = l_across > r_across
    if face_visible and left_on_right:
        return "front"
    if not face_visible and not left_on_right:
        return "back"
    return "unknown"  # the face and the shoulder order disagree (a mislabelled side, or a face turned away)


def pose_labels(keypoints: np.ndarray, *, min_joint_score: float = DEFAULT_MIN_JOINT_SCORE) -> PoseLabels:
    """Stance, arms and facing of one person's ``(133, 3)`` keypoints; each is ``unknown`` when its rule cannot
    decide from the confident joints."""
    kp = np.asarray(keypoints, dtype=np.float64)
    ok = confident(kp, min_joint_score)
    frame = torso_frame(kp, min_joint_score)
    arms, arm_classes = _arms(kp, ok, frame)
    return PoseLabels(stance=_stance(kp, ok, frame, arm_classes), arms=arms, facing=_facing(kp, ok, frame))
