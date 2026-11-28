"""Pose adherence: does a generated image hold the pose it was asked for?

Pose conditioning asks a generator for a pose; nothing so far checked that it complied. :func:`pose_adherence` takes
the target pose (the one the control image was drawn from) and the pose found in the output (``POST /pose`` on the
generated image), matches each target person to an output person, and says how far apart they are.

The distance is the suite's pose distance (17 body joints, hip-centred, torso-scaled, low-confidence joints masked;
``docs/pose-vector.md``) with one difference: it is **not mirror-invariant**. For a set's diversity a pose and its
mirror image are the same pose; for adherence a figure flipped left to right did not do what was asked. The mirrored
distance is reported beside it, so a flip is told apart from a miss.

It compares pose shape only: not where in the frame a person stands, how large they are, or their hands and face.
Pure numpy. See ``docs/conditioning.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from argus_puppet.embedding import PoseVector, embed_or_reason, oriented_distances
from argus_puppet.metrics import DEFAULT_PARAMS, shared_estimator
from argus_puppet.models import (
    DEFAULT_ADHERENCE_THRESHOLD,
    AdherenceMatch,
    BarePose,
    MetricParams,
    PoseAdherence,
    PoseResult,
)

Pose = PoseResult | BarePose | Sequence[np.ndarray]


def _people(pose: Pose, primary_only: bool) -> list[np.ndarray]:
    """The people of *pose* as ``(133, 3)`` arrays: everyone, or the primary subject alone (a bare pose's first)."""
    if isinstance(pose, PoseResult):
        if primary_only:
            person = pose.primary_person
            return [] if person is None else [person.keypoints_array()]
        return [p.keypoints_array() for p in pose.people]
    people = pose.people() if isinstance(pose, BarePose) else [np.asarray(p, dtype=np.float64) for p in pose]
    return people[:1] if primary_only else people


def pose_adherence(
    target: Pose,
    output: Pose,
    *,
    threshold: float = DEFAULT_ADHERENCE_THRESHOLD,
    primary_only: bool = False,
    params: MetricParams | None = None,
) -> PoseAdherence:
    """How well *output*'s people hold *target*'s poses.

    Each usable target person is matched to one output person: pairs are taken in order of distance (as posed),
    closest first, each person used once; ties go to the lower target index, then the lower output index. A target
    person is ``close`` when their match is below *threshold*, and ``mirrored`` when it is not but the output
    person's mirror image is. The result ``adheres`` when every usable target person is close and there is at least
    one. A target person who is unusable (too few confident joints: see ``unusable_reason``) is listed and does not
    count either way; an unusable output person is never a match.

    *threshold* defaults to the cluster cut (0.25): the output's pose would join the target's pose cluster. It is
    not calibrated on real ControlNet output, so read ``distance`` and choose your own.
    """
    params = params or DEFAULT_PARAMS
    if not (threshold >= 0 and math.isfinite(threshold)):
        raise ValueError(f"threshold {threshold} must be a finite number >= 0")
    embed = {"min_joint_score": params.min_joint_score, "min_usable_joints": params.min_usable_joints}
    rules = {
        "min_shared_joints": params.min_shared_joints,
        "min_shared_limb_joints": params.min_shared_limb_joints,
    }
    targets = [embed_or_reason(kp, **embed) for kp in _people(target, primary_only)]
    outputs = [embed_or_reason(kp, **embed) for kp in _people(output, primary_only)]

    # Every comparable pair, closest first: (distance as posed, target, output, mirrored distance).
    pairs: list[tuple[float, int, int, float]] = []
    for t, tv in enumerate(targets):
        if not isinstance(tv, PoseVector):
            continue
        for o, ov in enumerate(outputs):
            if isinstance(ov, PoseVector):
                posed, mirrored = oriented_distances(tv, ov, **rules)
                if not math.isnan(posed):
                    pairs.append((posed, t, o, mirrored))
    pairs.sort()
    found: dict[int, tuple[float, int, float]] = {}
    taken: set[int] = set()
    for posed, t, o, mirrored in pairs:
        if t not in found and o not in taken:
            found[t] = (posed, o, mirrored)
            taken.add(o)

    matches: list[AdherenceMatch] = []
    for t, tv in enumerate(targets):
        if not isinstance(tv, PoseVector):
            matches.append(AdherenceMatch(target=t, close=False, unusable_reason=tv))
            continue
        if t not in found:
            matches.append(AdherenceMatch(target=t, close=False))
            continue
        posed, o, mirrored = found[t]
        ov = outputs[o]
        assert isinstance(ov, PoseVector)
        close = posed < threshold
        matches.append(
            AdherenceMatch(
                target=t,
                output=o,
                distance=posed,
                mirrored_distance=None if math.isnan(mirrored) else mirrored,
                shared_joints=int((tv.mask & ov.mask).sum()),
                close=close,
                mirrored=not close and not math.isnan(mirrored) and mirrored < threshold,
            )
        )

    comparable = sum(1 for tv in targets if isinstance(tv, PoseVector))
    distances = [m.distance for m in matches if m.distance is not None]
    n_close = sum(1 for m in matches if m.close)
    both = [p for p in (target, output) if isinstance(p, PoseResult)]
    return PoseAdherence(
        adheres=comparable > 0 and n_close == comparable,
        threshold=threshold,
        primary_only=primary_only,
        target_people=len(targets),
        output_people=len(outputs),
        comparable=comparable,
        matched=len(distances),
        close=n_close,
        distance=max(distances) if distances else None,
        mean_distance=sum(distances) / len(distances) if distances else None,
        matches=matches,
        params=params,
        estimator=shared_estimator(both) if len(both) == 2 else None,
    )
