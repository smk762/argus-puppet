"""Pose diversity and coverage of one image set: :func:`pose_set_stats`.

Each image contributes its primary person. Usable poses (see :func:`embedding.embed`) are compared pairwise with
the mirror-invariant distance, grouped by average-linkage clustering cut at ``cluster_threshold``, projected to 2D
for a scatter plot, and summarised by ``diversity``, the number argus-proof's pose-diversity gate reads. Labels
(:mod:`argus_puppet.labels`) feed the coverage histograms.

Everything is deterministic and independent of the input order: the set is sorted by image name first, and the
output's items are listed in that order. See ``docs/metrics.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from argus_puppet.embedding import (
    JOINT_WEIGHTS,
    PoseVector,
    align_to_mean,
    embed,
    masked_mean,
    pairwise_distances,
    stack,
)
from argus_puppet.labels import pose_labels
from argus_puppet.models import (
    DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_DIVERSITY_SCALE,
    Arms,
    Coverage,
    EstimatorProvenance,
    Facing,
    MetricParams,
    PoseCluster,
    PoseLabels,
    PoseResult,
    PoseSetItem,
    PoseSetStats,
    Stance,
)

# ── defaults ─────────────────────────────────────────────────────────────────────────────────────────────────────

# Every default lives on MetricParams (the wire contract), with the reasoning beside it in models.py; these names are
# kept for callers that want one number.
DEFAULT_PARAMS = MetricParams()
DIVERSITY_SCALE = DEFAULT_DIVERSITY_SCALE
__all__ = [
    "DEFAULT_CLUSTER_THRESHOLD",
    "DEFAULT_PARAMS",
    "DIVERSITY_SCALE",
    "EmbeddedSet",
    "average_linkage",
    "cluster_ids",
    "coverage",
    "diversity",
    "embed_set",
    "keypoint_distance_matrix",
    "mean_pairwise_distance",
    "pair_diversity",
    "pose_set_stats",
    "project_2d",
    "set_distances",
    "shared_estimator",
]

STANCES: tuple[Stance, ...] = ("standing", "jumping", "crouching", "seated", "lying", "unknown")
ARMS: tuple[Arms, ...] = ("up", "out", "down", "mixed", "unknown")
FACINGS: tuple[Facing, ...] = ("front", "back", "left", "right", "unknown")


# ── building blocks ──────────────────────────────────────────────────────────────────────────────────────────────


def pair_diversity(d: np.ndarray, scale: float = DIVERSITY_SCALE) -> np.ndarray:
    """The per-pair term of :func:`diversity`: ``1 - exp(-(d / scale)^2)``, in [0, 1)."""
    return -np.expm1(-np.square(np.asarray(d, dtype=np.float64) / scale))


def _upper_rows(dist: np.ndarray, chunk: int = 256):
    """Yield the strictly-upper-triangle values of a square matrix, a block of rows at a time (no N^2 temporaries)."""
    n = len(dist)
    for i in range(0, n, chunk):
        block = np.asarray(dist[i : i + chunk], dtype=np.float64)
        cols = np.arange(n)
        rows = np.arange(i, min(n, i + chunk))[:, None]
        yield block[cols[None, :] > rows]


def diversity(dist: np.ndarray, scale: float = DIVERSITY_SCALE) -> float:
    """Pose diversity in [0, 1] from a square distance matrix: the mean of :func:`pair_diversity` over the comparable
    (non-NaN) pairs. 0 with fewer than two poses or no comparable pair."""
    total, count = 0.0, 0
    for vals in _upper_rows(dist):
        vals = vals[np.isfinite(vals)]
        total += float(pair_diversity(vals, scale).sum())
        count += len(vals)
    return min(1.0, max(0.0, total / count)) if count else 0.0


def mean_pairwise_distance(dist: np.ndarray) -> float:
    """Mean distance over the comparable pairs of a square distance matrix; 0 with none."""
    total, count = 0.0, 0
    for vals in _upper_rows(dist):
        vals = vals[np.isfinite(vals)]
        total += float(vals.sum())
        count += len(vals)
    return total / count if count else 0.0


def average_linkage(dist: np.ndarray, threshold: float) -> np.ndarray:
    """Flat clusters from average-linkage (UPGMA) agglomeration, merging while the two nearest clusters are closer
    than *threshold*. Returns one integer label per row (the lowest member index of its cluster).

    NaN distances (not comparable) count as infinite, so a cluster never holds an incomparable pair. Runs the
    nearest-neighbour-chain algorithm with the Lance-Williams update: O(N^2) time on one N x N working copy. Ties
    resolve by index, so the result depends only on the matrix (callers sort their items first).
    """
    n = len(dist)
    parent = np.arange(n)
    if n < 2:
        return parent
    d = np.array(dist, dtype=np.float32 if np.asarray(dist).dtype == np.float32 else np.float64, copy=True)
    d[~np.isfinite(d)] = np.inf
    np.fill_diagonal(d, np.inf)
    size = np.ones(n)
    alive = np.ones(n, dtype=bool)
    chain: list[int] = []
    while True:
        if not chain:
            rest = np.flatnonzero(alive)
            if len(rest) == 0:
                break
            chain.append(int(rest[0]))
        a = chain[-1]
        b = int(np.argmin(d[a]))
        best = d[a, b]
        if len(chain) > 1 and d[a, chain[-2]] <= best:
            b, best = chain[-2], d[a, chain[-2]]
        if not best < threshold:
            # a's nearest cluster is too far, and (average linkage being reducible) merges elsewhere can only move
            # the others further from it: a is final.
            alive[a] = False
            d[a, :] = np.inf
            d[:, a] = np.inf
            chain.pop()
            continue
        if len(chain) > 1 and b == chain[-2]:
            chain.pop()
            chain.pop()
            keep, drop = min(a, b), max(a, b)
            merged = (size[keep] * d[keep] + size[drop] * d[drop]) / (size[keep] + size[drop])
            d[keep, :] = merged
            d[:, keep] = merged
            d[keep, keep] = np.inf
            d[drop, :] = np.inf
            d[:, drop] = np.inf
            size[keep] += size[drop]
            parent[drop] = keep
            alive[drop] = False
        else:
            chain.append(b)
    # Resolve each item to its cluster's root.
    for i in range(n):
        r = i
        while parent[r] != r:
            r = parent[r]
        parent[i] = r
    return parent


def project_2d(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(N, 2)`` PCA projection of pose vectors for a scatter plot.

    The set is first canonicalised jointly (:func:`embedding.align_to_mean`), masked joints are imputed with the
    set's mean for that joint, each joint is weighted by the square root of its distance weight (so squared
    Euclidean geometry matches the distance's weighting), and the two leading principal components are taken by
    SVD. Each component's sign is fixed so that its largest-magnitude loading is positive. Rows beyond the set's
    rank are 0.
    """
    n = len(coords)
    out = np.zeros((n, 2))
    if n < 2:
        return out
    c, m = align_to_mean(coords, mask)
    mean, _ = masked_mean(c, m)
    filled = np.where(m[..., None], c, mean[None])
    x = (filled * np.sqrt(JOINT_WEIGHTS)[None, :, None]).reshape(n, -1)
    x = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(2, int((s > 1e-9 * max(1.0, float(s[0]) if len(s) else 1.0)).sum()))
    for i in range(k):
        j = int(np.argmax(np.abs(vt[i])))
        sign = 1.0 if vt[i, j] >= 0 else -1.0
        out[:, i] = sign * u[:, i] * s[i]
    return out


def cluster_ids(roots: Sequence[int] | np.ndarray) -> list[int]:
    """:func:`average_linkage`'s per-item roots as compact cluster ids: 0 for the largest cluster, equal sizes ordered
    by their first member's index."""
    groups: dict[int, list[int]] = {}
    for k, root in enumerate(roots):
        groups.setdefault(int(root), []).append(k)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    out = [0] * len(roots)
    for cid, members in enumerate(ordered):
        for k in members:
            out[k] = cid
    return out


def coverage(labels: Sequence[PoseLabels]) -> Coverage:
    """Counts per label, every label listed (zero or not)."""
    return Coverage(
        stance={k: sum(1 for x in labels if x.stance == k) for k in STANCES},
        arms={k: sum(1 for x in labels if x.arms == k) for k in ARMS},
        facing={k: sum(1 for x in labels if x.facing == k) for k in FACINGS},
    )


@dataclass(frozen=True)
class EmbeddedSet:
    """A set's primary poses, sorted by image name: what the metrics and the comparison both start from."""

    images: list[str]
    labels: list[PoseLabels]
    vectors: list[PoseVector | None]

    @property
    def usable_index(self) -> list[int]:
        return [i for i, v in enumerate(self.vectors) if v is not None]

    def usable_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return stack([v for v in self.vectors if v is not None])


def set_distances(coords: np.ndarray, mask: np.ndarray, params: MetricParams) -> np.ndarray:
    """:func:`embedding.pairwise_distances` under *params*' shared-joint rules."""
    return pairwise_distances(
        coords,
        mask,
        min_shared_joints=params.min_shared_joints,
        min_shared_limb_joints=params.min_shared_limb_joints,
    )


def keypoint_distance_matrix(poses: Sequence[np.ndarray], params: MetricParams | None = None) -> np.ndarray:
    """The ``(n, n)`` mirror-invariant pose distance between ``(133, 3)`` keypoint arrays (float64).

    A pose that cannot be embedded under *params* is NaN against everything, itself included; any other pair that
    shares too few joints is NaN too. The pose library's default distance."""
    params = params or DEFAULT_PARAMS
    vectors = [
        embed(np.asarray(p), min_joint_score=params.min_joint_score, min_usable_joints=params.min_usable_joints)
        for p in poses
    ]
    usable = [i for i, v in enumerate(vectors) if v is not None]
    out = np.full((len(vectors), len(vectors)), np.nan)
    if usable:
        coords, mask = stack([vectors[i] for i in usable])
        out[np.ix_(usable, usable)] = set_distances(coords, mask, params)
    return out


def shared_estimator(results: Sequence[PoseResult]) -> EstimatorProvenance | None:
    """The estimator every result names, when they all name the same one; else ``None`` (also for no results)."""
    if not results:
        return None
    first = results[0].estimator
    return first if all(r.estimator == first for r in results[1:]) else None


def _sort_key(r: PoseResult) -> tuple:
    p = r.primary_person
    return (r.image, r.image_sha256, b"" if p is None else p.keypoints_array().tobytes())


def embed_set(results: Sequence[PoseResult], params: MetricParams) -> EmbeddedSet:
    """Sort *results* by image name and embed and label each primary person."""
    images: list[str] = []
    labels: list[PoseLabels] = []
    vectors: list[PoseVector | None] = []
    for r in sorted(results, key=_sort_key):
        p = r.primary_person
        images.append(r.image)
        if p is None:
            labels.append(PoseLabels())
            vectors.append(None)
            continue
        kp = p.keypoints_array()
        labels.append(pose_labels(kp, min_joint_score=params.min_joint_score))
        vectors.append(embed(kp, min_joint_score=params.min_joint_score, min_usable_joints=params.min_usable_joints))
    return EmbeddedSet(images=images, labels=labels, vectors=vectors)


# ── the set stats ────────────────────────────────────────────────────────────────────────────────────────────────


def pose_set_stats(results: Sequence[PoseResult], *, params: MetricParams | None = None) -> PoseSetStats:
    """Diversity, clusters, coverage and a 2D projection of a set's primary poses.

    Images without a primary person, or whose primary person cannot be embedded, are listed with ``usable`` false
    (their labels are still filled where the rules can decide) and are left out of every number. Items come back
    sorted by image name; cluster ids run from 0 for the largest cluster, ties broken by the alphabetically first
    image in each.
    """
    params = params or DEFAULT_PARAMS
    es = embed_set(results, params)
    usable = es.usable_index
    coords, mask = es.usable_arrays()
    dist = set_distances(coords, mask, params)

    raw = average_linkage(dist, params.cluster_threshold)
    groups: dict[int, list[int]] = {}
    for k, root in enumerate(raw):
        groups.setdefault(int(root), []).append(k)  # k indexes the usable list, already in image-name order
    ordered = sorted(groups.values(), key=lambda g: (-len(g), es.images[usable[g[0]]]))
    n_usable = len(usable)

    clusters: list[PoseCluster] = []
    cluster_of = np.full(n_usable, -1)
    for cid, members in enumerate(ordered):
        cluster_of[members] = cid
        sub = np.asarray(dist[np.ix_(members, members)], dtype=np.float64)
        medoid = members[int(np.argmin(np.nansum(sub, axis=1)))]  # first by name on ties
        clusters.append(
            PoseCluster(id=cid, size=len(members), share=len(members) / n_usable, medoid=es.images[usable[medoid]])
        )

    xy = project_2d(coords, mask)
    pos = {i: k for k, i in enumerate(usable)}
    items = [
        PoseSetItem(
            image=es.images[i],
            usable=i in pos,
            cluster=int(cluster_of[pos[i]]) if i in pos else None,
            labels=es.labels[i],
            xy=(float(xy[pos[i], 0]), float(xy[pos[i], 1])) if i in pos else None,
        )
        for i in range(len(es.images))
    ]
    return PoseSetStats(
        count=len(es.images),
        usable=n_usable,
        diversity=diversity(dist, params.diversity_scale),
        mean_pairwise_distance=mean_pairwise_distance(dist),
        cluster_count=len(clusters),
        largest_cluster_share=clusters[0].share if clusters else 0.0,
        clusters=clusters,
        coverage=coverage([es.labels[i] for i in usable]),
        items=items,
        params=params,
        estimator=shared_estimator(results),
    )
