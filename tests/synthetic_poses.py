"""Synthetic whole-body people drawn in code, shared by the render / library tests (no image files)."""

from __future__ import annotations

import hashlib

import numpy as np

from argus_puppet import registry
from argus_puppet.models import EstimatorProvenance, PersonPose, PoseResult
from argus_puppet.skeleton import FACE, FEET, LEFT_HAND, RIGHT_HAND

# A standing figure in a 100 x 200 frame (subject's left on the image's right), wholebody body-17 order.
STANDING_BODY = np.array(
    [
        (50, 30),  # nose
        (53, 27),  # left eye
        (47, 27),  # right eye
        (56, 29),  # left ear
        (44, 29),  # right ear
        (62, 50),  # left shoulder
        (38, 50),  # right shoulder
        (66, 75),  # left elbow
        (34, 75),  # right elbow
        (68, 98),  # left wrist
        (32, 98),  # right wrist
        (57, 105),  # left hip
        (43, 105),  # right hip
        (58, 140),  # left knee
        (42, 140),  # right knee
        (58, 175),  # left ankle
        (42, 175),  # right ankle
    ],
    dtype=np.float64,
)


def figure(
    *,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    score: float = 0.9,
    arms_up: bool = False,
    body: np.ndarray | None = None,
) -> np.ndarray:
    """A ``(133, 3)`` person: the standing figure (or *body*), feet near the ankles, a ring of 68 face points round
    the nose and a fan of 21 points per hand below each wrist; scaled about the origin then shifted."""
    b = (STANDING_BODY if body is None else np.asarray(body, dtype=np.float64)).copy()
    if arms_up:
        b[7], b[8] = (68, 30), (32, 30)  # elbows
        b[9], b[10] = (70, 8), (30, 8)  # wrists
    kp = np.zeros((133, 3))
    kp[:17, :2] = b
    kp[FEET, :2] = [b[15] + (3, 4), b[15] + (-2, 4), b[15] + (0, 3), b[16] + (-3, 4), b[16] + (2, 4), b[16] + (0, 3)]
    ang = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    kp[FACE, 0] = b[0, 0] + 7 * np.cos(ang)
    kp[FACE, 1] = b[0, 1] + 7 * np.sin(ang)
    for hand, wrist in ((LEFT_HAND, b[9]), (RIGHT_HAND, b[10])):
        pts = [wrist]
        for f in range(5):
            a = np.pi / 2 + (f - 2) * 0.35
            pts += [wrist + (r * np.cos(a), r * np.sin(a)) for r in (3, 5, 7, 9)]
        kp[hand, :2] = pts
    kp[:, :2] = kp[:, :2] * scale + (dx, dy)
    kp[:, 2] = score
    return kp


def provenance() -> EstimatorProvenance:
    return EstimatorProvenance(
        name="topdown",
        detector=registry.model_ref(registry.DEFAULT_DETECTOR),
        pose=registry.model_ref(registry.DEFAULT_POSE),
        provider="CPUExecutionProvider",
        package_version="0.1.0",
        det_threshold=0.3,
        nms_iou=0.45,
    )


def person(kp: np.ndarray, *, primary: bool = False) -> PersonPose:
    return PersonPose.from_array(kp, bbox=(0.0, 0.0, 10.0, 10.0), det_score=0.9, primary=primary)


def result(
    people: list[np.ndarray], *, image: str = "img.png", width: int = 100, height: int = 200, primary: int | None = 0
) -> PoseResult:
    """A PoseResult for *people*; the image hash is derived from its name."""
    if not people:
        primary = None
    return PoseResult(
        image=image,
        image_sha256=hashlib.sha256(image.encode()).hexdigest(),
        width=width,
        height=height,
        people=[person(kp, primary=(i == primary)) for i, kp in enumerate(people)],
        primary=primary,
        estimator=provenance(),
    )
