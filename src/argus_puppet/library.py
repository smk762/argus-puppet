"""The pose library: poses extracted from a folder of images, deduplicated and sampled for conditioned generation.

A :class:`~argus_puppet.models.PoseLibrary` holds one entry per source image (its primary person), with the source
image's name and SHA-256 as provenance. This module builds one from :class:`~argus_puppet.models.PoseResult` objects,
saves and loads it as JSON, samples *N* diverse poses from it, and renders a sampled batch as control images.

**The pose distance** is the suite's mirror-invariant distance (:func:`metrics.keypoint_distance_matrix`, in torso
units; see ``docs/pose-vector.md``), entries are labelled with :func:`labels.pose_labels`, and clustered by
:func:`metrics.average_linkage` at the metrics' cluster threshold. Each is a parameter: ``distance_matrix`` (a list
of ``(133, 3)`` arrays to their ``(n, n)`` distance matrix, NaN for pairs it cannot compare), ``labeler`` (one array
to :class:`~argus_puppet.models.PoseLabels`) and ``clusterer`` (the matrix to one cluster id per pose); pass
``None`` as the labeler or clusterer to skip that step.

**The age fail-safe** (``age_gate``): with real-person sources, every candidate image is sent to the suite's age
check (:mod:`argus_puppet.age`) and kept only when every face and body in it reads 18 or over. It fails closed: an
image the check cannot vouch for (service down, an unreadable answer, nobody to estimate, an image that cannot be
read or no longer matches its pose's SHA-256) is excluded too.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import structlog
from PIL import Image

from argus_puppet.age import ADULT_AGE, AgeEstimator, AgeUnavailable
from argus_puppet.compare import DEFAULT_CLOSE_THRESHOLD
from argus_puppet.embedding import CORE_JOINTS, PoseVector, embed, mirror
from argus_puppet.labels import pose_labels
from argus_puppet.metrics import (
    DEFAULT_PARAMS,
    average_linkage,
    cluster_ids,
    keypoint_distance_matrix,
    shared_estimator,
)
from argus_puppet.models import (
    Arms,
    Facing,
    PoseLabels,
    PoseLibrary,
    PoseLibraryEntry,
    PoseResult,
    RenderOptions,
    Stance,
)
from argus_puppet.render import render_pose
from argus_puppet.sampling import deduplicate, farthest_point_sample

__all__ = [
    "DEFAULT_DEDUPE_THRESHOLD",
    "DEFAULT_MIN_BODY_SCORE",
    "AgeGate",
    "Clusterer",
    "DistanceMatrixFn",
    "LibraryBuild",
    "Labeler",
    "build_library",
    "default_clusterer",
    "dedupe_matrix",
    "default_distance_matrix",
    "entry_from_result",
    "entry_id",
    "load_library",
    "render_entries",
    "sample",
    "save_library",
]

log = structlog.get_logger(__name__)

DistanceMatrixFn = Callable[[Sequence[np.ndarray]], np.ndarray]
Labeler = Callable[[np.ndarray], PoseLabels]
Clusterer = Callable[[np.ndarray], Sequence[int | None]]

# A primary person whose mean body-joint score is below this is not a usable pose.
DEFAULT_MIN_BODY_SCORE = 0.3
# Poses closer than this, over the whole limb skeleton, collapse to the best-scored one. The library's "same pose" is
# the comparison's close pose match (compare.DEFAULT_CLOSE_THRESHOLD), so the suite has one definition. Measured on
# 60 synthetic poses plus two re-estimated copies of 20 of them: every copy with 0.01 torso units of joint noise and
# three in four with 0.02 are removed, while poses of distinct people in one stance with 8-10 degrees of limb
# variation are kept (at most 1 of 60 lost); 0.08 dropped 17 of 60 broad-mix poses and 52 of 60 same-stance ones.
# Below ~5 degrees of variation two people's poses are indistinguishable from a re-estimated copy and the dedupe
# merges them. See docs/conditioning.md.
DEFAULT_DEDUPE_THRESHOLD = DEFAULT_CLOSE_THRESHOLD


def default_distance_matrix(poses: Sequence[np.ndarray]) -> np.ndarray:
    """The suite's mirror-invariant pose distance under the default metric params (NaN where not comparable)."""
    return keypoint_distance_matrix(poses, DEFAULT_PARAMS)


def default_clusterer(dist: np.ndarray) -> list[int | None]:
    """Average-linkage clusters cut at the metrics' default threshold, as compact ids (0 for the largest)."""
    return list(cluster_ids(average_linkage(np.asarray(dist), DEFAULT_PARAMS.cluster_threshold)))


# ── entries ──────────────────────────────────────────────────────────────────────────────────────────────────────


def entry_id(source_sha256: str, keypoints: np.ndarray) -> str:
    """A stable, content-derived entry id: the source image hash prefix, then a hash of the keypoints (rounded to
    0.01 px so float noise does not change it). Matches ``PoseLibraryEntry.id``'s pattern."""
    kp = np.round(np.asarray(keypoints, dtype=np.float64), 2) + 0.0  # + 0.0 folds -0.0 into 0.0
    digest = hashlib.sha256(json.dumps(kp.tolist(), separators=(",", ":")).encode()).hexdigest()
    return f"{source_sha256[:16]}-{digest[:8]}"


def entry_from_result(
    result: PoseResult, *, min_body_score: float = DEFAULT_MIN_BODY_SCORE, labeler: Labeler | None = None
) -> PoseLibraryEntry | None:
    """The library entry for *result*'s primary person, or ``None`` when there is no primary person or their mean
    body score is below *min_body_score*."""
    person = result.primary_person
    if person is None or person.part_scores.body < min_body_score:
        return None
    kp = person.keypoints_array()
    return PoseLibraryEntry(
        id=entry_id(result.image_sha256, kp),
        source_image=result.image,
        source_sha256=result.image_sha256,
        width=result.width,
        height=result.height,
        keypoints=person.keypoints,
        labels=labeler(kp) if labeler is not None else PoseLabels(),
    )


# ── building ─────────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgeGate:
    """The age fail-safe for :func:`build_library`: *estimator* reads an image, *load_image* gives the bytes of the
    image a result came from (raising ``OSError`` or ``LookupError`` when it cannot)."""

    estimator: AgeEstimator
    load_image: Callable[[PoseResult], bytes]


@dataclass
class LibraryBuild:
    """A built library and what happened to each input."""

    library: PoseLibrary
    considered: int = 0
    no_person: int = 0  # no primary person
    low_score: int = 0  # primary person below min_body_score
    unembeddable: int = (
        0  # primary person the pose vector cannot use under the default params (no anchor, too few joints)
    )
    duplicate_ids: int = 0  # the same image (and pose) seen twice
    under_age: int = 0  # age check on: a face or body read under 18
    age_unchecked: int = 0  # age check on: the check could not vouch for the image (fails closed)
    age_passed: int = 0  # age check on: every face and body read 18 or over
    near_duplicates: int = 0  # dropped by the distance dedupe
    sampled_out: int = 0  # dropped by max_entries
    dropped: list[str] = field(default_factory=list)  # source images dropped as near-duplicates or sampled out
    age_excluded: list[tuple[str, str]] = field(default_factory=list)  # (source image, why) the age check kept out

    @property
    def age_checked(self) -> int:
        """Images the age check gave a verdict on (passed or under age)."""
        return self.age_passed + self.under_age


def _age_verdict(gate: AgeGate, result: PoseResult) -> str | None:
    """``None`` when the image passes, else why it is kept out ('under 18' or why it could not be checked)."""
    try:
        data = gate.load_image(result)
    except (OSError, LookupError) as exc:
        return f"unchecked: the image could not be read ({exc})"
    if hashlib.sha256(data).hexdigest() != result.image_sha256:
        return "unchecked: the image no longer matches its pose's SHA-256"
    try:
        youngest = gate.estimator.min_age(data)
    except AgeUnavailable as exc:
        return f"unchecked: {exc}"
    if youngest is None:
        return "unchecked: no face or body to estimate"
    if youngest < ADULT_AGE:
        return "under 18"
    return None


def build_library(
    results: Iterable[PoseResult],
    *,
    min_body_score: float = DEFAULT_MIN_BODY_SCORE,
    created_from: str = "",
    distance_matrix: DistanceMatrixFn | None = None,
    labeler: Labeler | None = pose_labels,
    clusterer: Clusterer | None = default_clusterer,
    dedupe_threshold: float | None = DEFAULT_DEDUPE_THRESHOLD,
    max_entries: int | None = None,
    seed: int | None = None,
    age_gate: AgeGate | None = None,
) -> LibraryBuild:
    """Build a library from pose results: one entry per image's usable primary person, labelled by *labeler*. A
    primary person is usable when their mean body score is at least *min_body_score* and the pose vector can embed
    them under the default metric params (``unembeddable`` counts the ones it cannot: no hip or shoulder to anchor
    on, or too few confident joints; such a pose could never be compared, deduplicated or clustered).

    Then, in order: entries with the same id (same image, same pose) collapse to the first; with *age_gate*, every
    remaining image goes through the age check and is kept only when it passes (failing closed, see the module
    docstring); with *dedupe_threshold*, poses closer than it to a better-scored kept pose, over the whole limb
    skeleton (:func:`dedupe_matrix`), are dropped (best mean body score wins); with *clusterer*, each entry gets a cluster id; with *max_entries*, a diverse subset of that
    size is kept by farthest-point sampling, stratified by cluster (deterministic, or seeded with *seed*). The distance
    steps see the entries sorted by source image and id, so the result does not depend on the input order; entries
    keep input order. *distance_matrix* defaults to :func:`default_distance_matrix`; it is only computed when a step
    needs it. ``library.age_checked`` records whether the age check ran.
    """
    pairs: list[tuple[PoseResult, PoseLibraryEntry]] = []
    vectors: list[PoseVector] = []
    seen: set[str] = set()
    build = LibraryBuild(library=PoseLibrary(created_from=created_from))
    all_results: list[PoseResult] = []
    for result in results:
        build.considered += 1
        all_results.append(result)
        if result.primary_person is None:
            build.no_person += 1
            continue
        entry = entry_from_result(result, min_body_score=min_body_score, labeler=labeler)
        if entry is None:
            build.low_score += 1
            continue
        vector = _embed(entry.keypoints_array())
        if vector is None:
            build.unembeddable += 1
            continue
        if entry.id in seen:
            build.duplicate_ids += 1
            continue
        seen.add(entry.id)
        if age_gate is not None:
            why = _age_verdict(age_gate, result)
            if why is not None:
                if why == "under 18":
                    build.under_age += 1
                else:
                    build.age_unchecked += 1
                build.age_excluded.append((result.image, why))
                continue
            build.age_passed += 1
        pairs.append((result, entry))
        vectors.append(vector)

    # The distance steps see the entries in a canonical order (by source image, then id) so that ties in the dedupe,
    # the clustering and the sampling never depend on the order the results arrived in; entries keep input order.
    order = sorted(range(len(pairs)), key=lambda i: (pairs[i][1].source_image, pairs[i][1].id))
    entries = [pairs[i][1] for i in order]
    quality = [pairs[i][0].primary_person.part_scores.body for i in order]  # type: ignore[union-attr]
    vectors = [vectors[i] for i in order]
    needs_distance = dedupe_threshold is not None or clusterer is not None or max_entries is not None
    if entries and needs_distance:
        dist_fn = distance_matrix or default_distance_matrix
        d = _checked_matrix(dist_fn([e.keypoints_array() for e in entries]), len(entries))
        keep = list(range(len(entries)))
        if dedupe_threshold is not None:
            keep = deduplicate(dedupe_matrix(d, vectors), dedupe_threshold, quality=quality)
            build.near_duplicates = len(entries) - len(keep)
            build.dropped += [entries[i].source_image for i in sorted(set(range(len(entries))) - set(keep))]
        sub = d[np.ix_(keep, keep)]
        entries = [entries[i] for i in keep]
        clusters: list[int | None] | None = None
        if clusterer is not None and entries:
            clusters = list(clusterer(sub))
            if len(clusters) != len(entries):
                raise ValueError(f"clusterer returned {len(clusters)} ids for {len(entries)} poses")
            entries = [e.model_copy(update={"cluster": c}) for e, c in zip(entries, clusters, strict=True)]
        if max_entries is not None and len(entries) > max_entries:
            picked = set(farthest_point_sample(sub, max_entries, seed=seed, clusters=clusters))
            build.sampled_out = len(entries) - len(picked)
            build.dropped += [e.source_image for i, e in enumerate(entries) if i not in picked]
            entries = [e for i, e in enumerate(entries) if i in picked]

    position = {e.id: i for i, (_, e) in enumerate(pairs)}
    entries.sort(key=lambda e: position[e.id])  # back to input order
    build.library = PoseLibrary(
        created_from=created_from,
        estimator=shared_estimator(all_results),
        entries=entries,
        age_checked=age_gate is not None,
    )
    log.info(
        "pose_library.built",
        considered=build.considered,
        entries=len(entries),
        no_person=build.no_person,
        low_score=build.low_score,
        unembeddable=build.unembeddable,
        duplicate_ids=build.duplicate_ids,
        age_check=age_gate is not None,
        under_age=build.under_age,
        age_unchecked=build.age_unchecked,
        near_duplicates=build.near_duplicates,
        sampled_out=build.sampled_out,
    )
    return build


def _embed(keypoints: np.ndarray) -> PoseVector | None:
    return embed(
        keypoints, min_joint_score=DEFAULT_PARAMS.min_joint_score, min_usable_joints=DEFAULT_PARAMS.min_usable_joints
    )


def dedupe_matrix(
    d: np.ndarray, vectors: Sequence[PoseVector], min_close_joints: int = DEFAULT_PARAMS.min_close_joints
) -> np.ndarray:
    """*d* with every pair that shares fewer than *min_close_joints* of the 12 non-head joints set to NaN (never a
    duplicate). With the default 12 a duplicate needs both poses to show the whole limb skeleton, so a pose dropped as
    a duplicate of a kept one was compared over the same joints as every other pose that pose absorbed."""
    core = list(CORE_JOINTS)
    c = np.stack([v.mask[core] for v in vectors]).astype(np.int64)
    cf = np.stack([mirror(v.coords, v.mask)[1][core] for v in vectors]).astype(np.int64)
    shared = np.maximum(c @ c.T, c @ cf.T)
    return np.where(shared >= min_close_joints, d, np.nan)


def _checked_matrix(d: np.ndarray, n: int) -> np.ndarray:
    m = np.asarray(d, dtype=np.float64)
    if m.shape != (n, n):
        raise ValueError(f"distance_matrix returned shape {m.shape} for {n} poses")
    return m


# ── persistence ──────────────────────────────────────────────────────────────────────────────────────────────────


def save_library(library: PoseLibrary, path: str | os.PathLike[str]) -> Path:
    """Write *library* as JSON (sorted keys, compact) atomically: a temp file beside *path*, then ``os.replace``."""
    target = Path(path)
    text = json.dumps(library.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def load_library(path: str | os.PathLike[str]) -> PoseLibrary:
    """Read and validate a library written by :func:`save_library` (any supported ``puppet_version``)."""
    return PoseLibrary.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ── sampling and rendering ───────────────────────────────────────────────────────────────────────────────────────


def sample(
    library: PoseLibrary,
    n: int,
    *,
    seed: int | None = None,
    distance_matrix: DistanceMatrixFn | None = None,
    stratify: bool = True,
    stance: Stance | None = None,
    arms: Arms | None = None,
    facing: Facing | None = None,
    start: int | Literal["medoid", "random"] | None = None,
) -> list[PoseLibraryEntry]:
    """*n* mutually distant entries, in pick order (farthest-point sampling over the pose distance).

    Entries are first filtered to the given *stance* / *arms* / *facing* labels. With *stratify* and clustered
    entries, every cluster is sampled once before any twice. Unseeded, the first pick is the most central pose and
    the result is deterministic; with *seed*, the first pick is random (reproducibly). ``n >= len(pool)`` returns
    the whole pool. *start* indexes the filtered pool. *distance_matrix* defaults to
    :func:`default_distance_matrix`.
    """
    pool = [
        e
        for e in library.entries
        if (stance is None or e.labels.stance == stance)
        and (arms is None or e.labels.arms == arms)
        and (facing is None or e.labels.facing == facing)
    ]
    if n <= 0 or not pool:
        return []
    if len(pool) == 1:
        return pool[:1]
    d = _checked_matrix((distance_matrix or default_distance_matrix)([e.keypoints_array() for e in pool]), len(pool))
    clusters = [e.cluster for e in pool] if stratify and any(e.cluster is not None for e in pool) else None
    return [pool[i] for i in farthest_point_sample(d, n, seed=seed, clusters=clusters, start=start)]


def render_entries(
    entries: Sequence[PoseLibraryEntry],
    size: tuple[int, int],
    options: RenderOptions | None = None,
    *,
    margin: float | None = None,
    stroke_scale: float | Literal["auto"] | None = None,
) -> list[Image.Image]:
    """One control image per entry on a ``(width, height)`` canvas, each skeleton re-fitted into it: the bounding
    box of the points that will be drawn, scaled uniformly (aspect preserved) to fit inside *margin* (a fraction of
    each side) and centred. ``options.width``/``height`` and ``options.fit`` are ignored in favour of *size* and
    ``"skeleton"``; *margin* and *stroke_scale* default to the options' fields."""
    return [
        render_pose(e, options, canvas_size=size, fit="skeleton", margin=margin, stroke_scale=stroke_scale)
        for e in entries
    ]
