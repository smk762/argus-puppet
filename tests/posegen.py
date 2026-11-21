"""Parametric stick figures as COCO-WholeBody (133, 3) keypoints, for the embedding, label and metric tests.

A figure is built in body units (torso length 1, y up), then rotated, scaled, translated and flipped into image
pixels (y down). Limb angles are degrees from straight down: positive is outward (front and back views) or forward
(profile views). Hands and face landmarks are left unscored; the feet follow the ankles.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from argus_puppet.models import EstimatorProvenance, PersonPose, PoseResult
from argus_puppet.skeleton import FEET, FLIP_PERMUTATION, NUM_KEYPOINTS, Joint

UPPER_ARM, FOREARM, THIGH, SHIN = 0.55, 0.5, 0.9, 0.85
SCORE = 0.9


@dataclass(frozen=True)
class Figure:
    view: str = "front"  # front, back, left, right (facing the image's left or right)
    rot: float = 0.0  # whole-figure rotation about the mid-hip, degrees counter-clockwise
    lean: float = 0.0  # torso lean, degrees: forward in profile, toward the image right in front/back
    l_arm: tuple[float, float] = (8.0, 5.0)  # upper arm, forearm
    r_arm: tuple[float, float] = (8.0, 5.0)
    l_leg: tuple[float, float, float] = (4.0, 2.0, 1.0)  # thigh, shin, thigh length factor (foreshortening)
    r_leg: tuple[float, float, float] = (4.0, 2.0, 1.0)
    hidden: tuple[int, ...] = field(default_factory=tuple)


def _dir(deg: float, sign: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([sign * np.sin(t), -np.cos(t)])


def render(fig: Figure) -> np.ndarray:
    """The figure in body units: ``(133, 3)``, y up, mid-hip at the origin."""
    kp = np.zeros((NUM_KEYPOINTS, 3))
    profile = fig.view in ("left", "right")
    if profile:
        fwd = -1.0 if fig.view == "left" else 1.0
        l_sign = r_sign = fwd
        half_sh, half_hip = 0.04, 0.03
        left_x = fwd  # the near side; small offsets only
    else:
        left_x = 1.0 if fig.view == "front" else -1.0
        l_sign, r_sign = left_x, -left_x
        half_sh, half_hip = 0.35, 0.23
        fwd = 0.0
    lean_sign = fwd if profile else 1.0
    up = np.array([lean_sign * np.sin(np.radians(fig.lean)), np.cos(np.radians(fig.lean))])
    side = np.array([up[1], -up[0]])  # perpendicular, toward +x when upright
    hip_side = np.array([1.0, 0.0])
    neck = up * 1.0

    def put(j: int, p: np.ndarray, visible: bool = True) -> None:
        kp[j, :2] = p
        kp[j, 2] = SCORE if visible else 0.0

    put(Joint.LEFT_SHOULDER, neck + left_x * half_sh * side)
    put(Joint.RIGHT_SHOULDER, neck - left_x * half_sh * side)
    put(Joint.LEFT_HIP, left_x * half_hip * hip_side)
    put(Joint.RIGHT_HIP, -left_x * half_hip * hip_side)

    # Arms hang from the shoulders in the torso's frame (rotated with the lean).
    lean_rad = np.radians(fig.lean) * lean_sign
    rot = np.array([[np.cos(lean_rad), np.sin(lean_rad)], [-np.sin(lean_rad), np.cos(lean_rad)]])
    for (upper, fore), sign, s, e, w in (
        (fig.l_arm, l_sign, Joint.LEFT_SHOULDER, Joint.LEFT_ELBOW, Joint.LEFT_WRIST),
        (fig.r_arm, r_sign, Joint.RIGHT_SHOULDER, Joint.RIGHT_ELBOW, Joint.RIGHT_WRIST),
    ):
        elbow = kp[s, :2] + UPPER_ARM * (rot @ _dir(upper, sign))
        put(e, elbow)
        put(w, elbow + FOREARM * (rot @ _dir(fore, sign)))
    for (thigh, shin, f), sign, h, k, a, feet in (
        (fig.l_leg, l_sign, Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE, range(FEET.start, FEET.start + 3)),
        (fig.r_leg, r_sign, Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE, range(FEET.start + 3, FEET.stop)),
    ):
        knee = kp[h, :2] + THIGH * f * _dir(thigh, sign)
        put(k, knee)
        ankle = knee + SHIN * _dir(shin, sign)
        put(a, ankle)
        toe = np.array([fwd if profile else 0.0, -0.05])
        for i, off in zip(feet, (0.2, 0.15, -0.05), strict=True):
            put(i, ankle + off * (toe if profile else np.array([0.0, -0.1])) + np.array([0, -0.05]))

    head = neck
    if profile:
        put(Joint.NOSE, head + 0.35 * up + 0.15 * fwd * np.array([1.0, 0.0]))
        near_eye, far_eye = (
            (Joint.LEFT_EYE, Joint.RIGHT_EYE) if fig.view == "left" else (Joint.RIGHT_EYE, Joint.LEFT_EYE)
        )
        near_ear, far_ear = (
            (Joint.LEFT_EAR, Joint.RIGHT_EAR) if fig.view == "left" else (Joint.RIGHT_EAR, Joint.LEFT_EAR)
        )
        put(near_eye, head + 0.42 * up + 0.1 * fwd * np.array([1.0, 0.0]))
        put(far_eye, head + 0.42 * up + 0.1 * fwd * np.array([1.0, 0.0]), visible=False)
        put(near_ear, head + 0.38 * up - 0.03 * fwd * np.array([1.0, 0.0]))
        put(far_ear, head + 0.38 * up, visible=False)
    else:
        face = fig.view == "front"
        put(Joint.NOSE, head + 0.35 * up, visible=face)
        put(Joint.LEFT_EYE, head + 0.42 * up + left_x * 0.07 * side, visible=face)
        put(Joint.RIGHT_EYE, head + 0.42 * up - left_x * 0.07 * side, visible=face)
        put(Joint.LEFT_EAR, head + 0.38 * up + left_x * 0.14 * side)
        put(Joint.RIGHT_EAR, head + 0.38 * up - left_x * 0.14 * side)

    if fig.rot:
        t = np.radians(fig.rot)
        r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        kp[:, :2] = kp[:, :2] @ r.T
    for j in fig.hidden:
        kp[j, 2] = 0.0
    return kp


def to_image(body: np.ndarray, *, scale: float = 100.0, tx: float = 500.0, ty: float = 500.0) -> np.ndarray:
    """Body units (y up) to image pixels (y down)."""
    kp = body.copy()
    kp[:, 0] = tx + scale * body[:, 0]
    kp[:, 1] = ty - scale * body[:, 1]
    return kp


PRESETS: dict[str, Figure] = {
    "standing": Figure(),
    "t_pose": Figure(l_arm=(90, 90), r_arm=(90, 90)),
    "arms_up": Figure(l_arm=(170, 175), r_arm=(170, 175)),
    "wave": Figure(l_arm=(150, 175), r_arm=(8, 5)),
    "hands_on_hips": Figure(l_arm=(50, -60), r_arm=(50, -60)),
    "seated": Figure(view="right", l_arm=(20, 60), r_arm=(20, 60), l_leg=(90, 0, 1.0), r_leg=(90, 0, 1.0)),
    "seated_front": Figure(l_arm=(10, 30), r_arm=(10, 30), l_leg=(5, 3, 0.3), r_leg=(5, 3, 0.3)),
    "floor_sit": Figure(view="left", lean=-10, l_arm=(30, 30), r_arm=(30, 30), l_leg=(90, 90, 1), r_leg=(88, 92, 1)),
    "crouch": Figure(view="right", lean=30, l_arm=(60, 80), r_arm=(60, 80), l_leg=(95, -30, 1), r_leg=(95, -30, 1)),
    "lying": Figure(rot=90),
    "jump_tuck": Figure(l_arm=(150, 160), r_arm=(150, 160), l_leg=(160, 10, 0.5), r_leg=(160, 10, 0.5)),
    "star_jump": Figure(l_arm=(135, 140), r_arm=(135, 140), l_leg=(25, 25, 1), r_leg=(25, 25, 1)),
    "profile": Figure(view="left"),
    "back": Figure(view="back"),
    "walk": Figure(view="right", l_arm=(-25, -10), r_arm=(25, 50), l_leg=(25, 10, 1), r_leg=(-20, -30, 1)),
    "reach": Figure(view="left", l_arm=(90, 95), r_arm=(85, 90)),
}

EXPECTED_STANCE = {
    "standing": "standing",
    "t_pose": "standing",
    "arms_up": "standing",
    "wave": "standing",
    "hands_on_hips": "standing",
    "seated": "seated",
    "seated_front": "seated",
    "floor_sit": "seated",
    "crouch": "crouching",
    "lying": "lying",
    "jump_tuck": "jumping",
    "star_jump": "jumping",
    "profile": "standing",
    "back": "standing",
}
EXPECTED_ARMS = {
    "standing": "down",
    "t_pose": "out",
    "arms_up": "up",
    "wave": "mixed",
    "hands_on_hips": "down",
    "jump_tuck": "up",
    "star_jump": "up",
    "profile": "down",
    "back": "down",
    "reach": "out",
    "lying": "down",
}
EXPECTED_FACING = {
    "standing": "front",
    "t_pose": "front",
    "profile": "left",
    "reach": "left",
    "seated": "right",
    "back": "back",
    "star_jump": "front",
}


def jitter_figure(fig: Figure, rng: np.random.Generator, degrees: float) -> Figure:
    """The figure with every limb angle moved by a uniform +-*degrees*."""

    def j(v: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(x + rng.uniform(-degrees, degrees) if i < 2 else x for i, x in enumerate(v))

    return replace(fig, l_arm=j(fig.l_arm), r_arm=j(fig.r_arm), l_leg=j(fig.l_leg), r_leg=j(fig.r_leg))


def make(
    name: str | Figure,
    *,
    rng: np.random.Generator | None = None,
    angle_jitter: float = 0.0,
    joint_noise: float = 0.0,
    scale: float = 100.0,
    tx: float = 500.0,
    ty: float = 500.0,
    mirror: bool = False,
    dropout: tuple[int, ...] = (),
    width: int = 1000,
) -> np.ndarray:
    """One figure as image keypoints ``(133, 3)``, optionally jittered (limb angles in degrees, joint positions in
    torso units), mirrored about the image's centre line, or with joints dropped (score 0)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    fig = PRESETS[name] if isinstance(name, str) else name
    if angle_jitter:
        fig = jitter_figure(fig, rng, angle_jitter)
    body = render(fig)
    if joint_noise:
        body[:23, :2] += rng.normal(0.0, joint_noise, size=(23, 2))
    kp = to_image(body, scale=scale, tx=tx, ty=ty)
    if mirror:
        kp = kp[FLIP_PERMUTATION].copy()
        kp[:, 0] = (width - 1) - kp[:, 0]
    for j in dropout:
        kp[j, 2] = 0.0
    return kp


def provenance() -> EstimatorProvenance:
    from argus_puppet import registry

    return EstimatorProvenance(
        name="topdown",
        detector=registry.model_ref(registry.DEFAULT_DETECTOR),
        pose=registry.model_ref(registry.DEFAULT_POSE),
        provider="CPUExecutionProvider",
        package_version="0.1.0",
        det_threshold=0.3,
        nms_iou=0.45,
    )


_PROV: EstimatorProvenance | None = None


def result(image: str, keypoints: np.ndarray | None, *, width: int = 1000, height: int = 1000) -> PoseResult:
    """A PoseResult with one primary person (or nobody when *keypoints* is None)."""
    global _PROV
    if _PROV is None:
        _PROV = provenance()
    if keypoints is None:
        return PoseResult(image=image, image_sha256="ab" * 32, width=width, height=height, estimator=_PROV)
    kp = keypoints.copy()
    kp[:, :2] = np.clip(kp[:, :2], -1e6, 1e6)
    person = PersonPose.from_array(kp, bbox=(0.0, 0.0, 10.0, 10.0), det_score=0.9, primary=True)
    return PoseResult(
        image=image, image_sha256="ab" * 32, width=width, height=height, people=[person], primary=0, estimator=_PROV
    )
