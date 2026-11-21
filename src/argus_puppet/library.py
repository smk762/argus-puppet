"""The pose library: poses extracted from a folder of images, deduplicated and sampled for conditioned generation.

A :class:`~argus_puppet.models.PoseLibrary` holds one entry per source image (its primary person), with the source
image's name and SHA-256 as provenance. This module builds one from :class:`~argus_puppet.models.PoseResult` objects,
saves and loads it as JSON, samples *N* diverse poses from it, and renders a sampled batch as control images.

**Pose distance is injected.** The suite's pose distance (mirror-invariant, over ``(133, 3)`` keypoints) lives in
the metrics module; pass it as ``distance_matrix``: a callable taking a list of ``(133, 3)`` arrays and returning
their ``(n, n)`` pairwise distance matrix (NaN for pairs it cannot compare). Optional ``labeler`` (one ``(133, 3)``
array to :class:`~argus_puppet.models.PoseLabels`) and ``clusterer`` (the ``(n, n)`` distance matrix to one cluster
id per pose, ``None`` for unclustered) fill each entry's labels and cluster. Without them,
:func:`fallback_distance_matrix` is used: a deliberately simple **stand-in** so the library works on its own.
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

from argus_puppet.models import (
    Arms,
    EstimatorProvenance,
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
from argus_puppet.skeleton import BODY, FLIP_PERMUTATION, L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

__all__ = [
    "DEFAULT_MIN_BODY_SCORE",
    "FALLBACK_DEDUPE_THRESHOLD",
    "Clusterer",
    "DistanceMatrixFn",
    "LibraryBuild",
    "Labeler",
    "build_library",
    "entry_from_result",
    "entry_id",
    "fallback_distance_matrix",
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
# A joint counts in the fallback distance at or above this score.
FALLBACK_MIN_JOINT_SCORE = 0.3
# The fallback distance needs this many joints confident in both poses; fewer and the pair is NaN.
FALLBACK_MIN_COMMON_JOINTS = 6
# A dedupe threshold that suits the fallback distance (mean joint offset in torso lengths). The injected distance
# has its own scale: pick its threshold from its docs, not this.
FALLBACK_DEDUPE_THRESHOLD = 0.1


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


# ── the stand-in distance ────────────────────────────────────────────────────────────────────────────────────────

_BODY_FLIP = FLIP_PERMUTATION[BODY]  # body joints map onto body joints under a flip


def _normalise_body(kp: np.ndarray, min_score: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Body-17 coordinates centred on the hips and scaled by torso length, with the confident-joint mask; ``None``
    when no hip or no shoulder is confident, or the torso has no length."""
    body = np.asarray(kp, dtype=np.float64)[BODY]
    ok = np.isfinite(body[:, :2]).all(axis=1) & (np.nan_to_num(body[:, 2], nan=0.0) >= min_score)
    hips, shoulders = [j for j in (L_HIP, R_HIP) if ok[j]], [j for j in (L_SHOULDER, R_SHOULDER) if ok[j]]
    if not hips or not shoulders:
        return None
    hip_c, sh_c = body[hips, :2].mean(axis=0), body[shoulders, :2].mean(axis=0)
    scale = float(np.linalg.norm(sh_c - hip_c))
    if not np.isfinite(scale) or scale < 1e-6:
        return None
    xy = np.where(ok[:, None], (body[:, :2] - hip_c) / scale, 0.0)
    return xy, ok


def fallback_distance_matrix(poses: Sequence[np.ndarray]) -> np.ndarray:
    """**Stand-in** pose distance: mean body-joint offset, in torso lengths, after hip-centring.

    Each pose's 17 body joints are centred on the mean of its confident hips and divided by its hip-to-shoulder
    distance; the distance between two poses is the mean Euclidean offset over the joints confident (score >= 0.3)
    in both, minimised over mirroring one of them (so a pose and its mirror image are the same pose). NaN when a pose
    cannot be normalised or the pair shares fewer than 6 confident joints. Simple, scale- and translation-invariant
    but not rotation-invariant; the suite's real metric replaces it.
    """
    norm = [_normalise_body(p, FALLBACK_MIN_JOINT_SCORE) for p in poses]
    n = len(norm)
    d = np.full((n, n), np.nan)
    for i in range(n):
        if norm[i] is None:
            continue
        d[i, i] = 0.0
        a, ma = norm[i]  # type: ignore[misc]
        for j in range(i + 1, n):
            if norm[j] is None:
                continue
            b, mb = norm[j]  # type: ignore[misc]
            fb = b[_BODY_FLIP] * np.array([-1.0, 1.0])
            fmb = mb[_BODY_FLIP]
            best = np.nan
            for other, mask in ((b, mb), (fb, fmb)):
                common = ma & mask
                if common.sum() >= FALLBACK_MIN_COMMON_JOINTS:
                    v = float(np.linalg.norm(a[common] - other[common], axis=1).mean())
                    best = v if np.isnan(best) else min(best, v)
            d[i, j] = d[j, i] = best
    return d


# ── building ─────────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class LibraryBuild:
    """A built library and what happened to each input."""

    library: PoseLibrary
    considered: int = 0
    no_person: int = 0  # no primary person
    low_score: int = 0  # primary person below min_body_score
    duplicate_ids: int = 0  # the same image (and pose) seen twice
    near_duplicates: int = 0  # dropped by the distance dedupe
    sampled_out: int = 0  # dropped by max_entries
    dropped: list[str] = field(default_factory=list)  # source images dropped as near-duplicates or sampled out


def _shared_estimator(results: list[PoseResult]) -> EstimatorProvenance | None:
    if not results:
        return None
    first = results[0].estimator
    return first if all(r.estimator == first for r in results[1:]) else None


def build_library(
    results: Iterable[PoseResult],
    *,
    min_body_score: float = DEFAULT_MIN_BODY_SCORE,
    created_from: str = "",
    distance_matrix: DistanceMatrixFn | None = None,
    labeler: Labeler | None = None,
    clusterer: Clusterer | None = None,
    dedupe_threshold: float | None = None,
    max_entries: int | None = None,
    seed: int | None = None,
) -> LibraryBuild:
    """Build a library from pose results: one entry per image's usable primary person.

    Then, in order: entries with the same id (same image, same pose) collapse to the first; with
    *dedupe_threshold*, poses closer than it to a better-scored kept pose are dropped (best mean body score wins);
    with *clusterer*, each entry gets a cluster id; with *max_entries*, a diverse subset of that size is kept by
    farthest-point sampling, stratified by cluster (deterministic, or seeded with *seed*). Entries keep input order.
    *distance_matrix* defaults to :func:`fallback_distance_matrix`; it is only computed when a step needs it.
    """
    pairs: list[tuple[PoseResult, PoseLibraryEntry]] = []
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
        if entry.id in seen:
            build.duplicate_ids += 1
            continue
        seen.add(entry.id)
        pairs.append((result, entry))

    entries = [e for _, e in pairs]
    quality = [r.primary_person.part_scores.body for r, _ in pairs]  # type: ignore[union-attr]
    needs_distance = dedupe_threshold is not None or clusterer is not None or max_entries is not None
    if entries and needs_distance:
        dist_fn = distance_matrix or fallback_distance_matrix
        d = _checked_matrix(dist_fn([e.keypoints_array() for e in entries]), len(entries))
        keep = list(range(len(entries)))
        if dedupe_threshold is not None:
            keep = deduplicate(d, dedupe_threshold, quality=quality)
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

    build.library = PoseLibrary(created_from=created_from, estimator=_shared_estimator(all_results), entries=entries)
    log.info(
        "pose_library.built",
        considered=build.considered,
        entries=len(entries),
        no_person=build.no_person,
        low_score=build.low_score,
        duplicate_ids=build.duplicate_ids,
        near_duplicates=build.near_duplicates,
        sampled_out=build.sampled_out,
    )
    return build


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
    :func:`fallback_distance_matrix`.
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
    d = _checked_matrix((distance_matrix or fallback_distance_matrix)([e.keypoints_array() for e in pool]), len(pool))
    clusters = [e.cluster for e in pool] if stratify and any(e.cluster is not None for e in pool) else None
    return [pool[i] for i in farthest_point_sample(d, n, seed=seed, clusters=clusters, start=start)]


def render_entries(
    entries: Sequence[PoseLibraryEntry],
    size: tuple[int, int],
    options: RenderOptions | None = None,
    *,
    margin: float = 0.05,
    stroke_scale: float | Literal["auto"] = 1.0,
) -> list[Image.Image]:
    """One control image per entry on a ``(width, height)`` canvas, each skeleton re-fitted into it: the bounding
    box of the points that will be drawn, scaled uniformly (aspect preserved) to fit inside *margin* (a fraction of
    each side) and centred. ``options.width``/``height`` are ignored in favour of *size*."""
    return [
        render_pose(e, options, canvas_size=size, fit="skeleton", margin=margin, stroke_scale=stroke_scale)
        for e in entries
    ]
