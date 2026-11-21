"""Set selection over generic items: deduplication and diverse (farthest-point) sampling.

Pure numpy and deterministic: the same inputs (and seed) give the same selection. Items are indices ``0..n-1``;
distances come from a precomputed ``(n, n)`` matrix or a callable ``distance(i, j) -> float``. Distances are assumed
symmetric; the diagonal is ignored.

**Unusable pairs.** A NaN or infinite distance means "these two cannot be compared" (e.g. a pose missing the
joints the metric needs), not "infinitely far apart":

* :func:`deduplicate` never treats an unusable pair as a duplicate;
* :func:`farthest_point_sample` scores a candidate only on the selected items it *can* be compared with. Candidates
  comparable with none of them rank after every comparable candidate (they carry no evidence of being different),
  so an item that is unusable against everything is picked only once the comparable ones run out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

__all__ = [
    "DistanceFn",
    "DistanceSource",
    "deduplicate",
    "euclidean_distances",
    "farthest_point_sample",
]

DistanceFn = Callable[[int, int], float]
DistanceSource = np.ndarray | DistanceFn


class _Rows:
    """Row access to a distance matrix or callable, with unusable entries (and the diagonal) as NaN."""

    def __init__(self, distance: DistanceSource, n: int | None) -> None:
        if callable(distance):
            if n is None:
                raise ValueError("n is required with a distance callable")
            self._fn: DistanceFn | None = distance
            self._matrix: np.ndarray | None = None
            self.n = int(n)
        else:
            m = np.asarray(distance, dtype=np.float64)
            if m.ndim != 2 or m.shape[0] != m.shape[1]:
                raise ValueError(f"distance matrix must be square, got shape {m.shape}")
            if n is not None and n != m.shape[0]:
                raise ValueError(f"n={n} disagrees with the {m.shape[0]}x{m.shape[0]} distance matrix")
            self._fn, self._matrix, self.n = None, m, m.shape[0]
        if self.n < 0:
            raise ValueError("n must be >= 0")
        self._cache: dict[int, np.ndarray] = {}

    def row(self, i: int) -> np.ndarray:
        cached = self._cache.get(i)
        if cached is not None:
            return cached
        if self._matrix is not None:
            r = self._matrix[i].copy()
        else:
            assert self._fn is not None
            r = np.array([np.nan if j == i else float(self._fn(i, j)) for j in range(self.n)], dtype=np.float64)
        r[~np.isfinite(r)] = np.nan
        r[i] = np.nan
        self._cache[i] = r
        return r


def euclidean_distances(vectors: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances between the rows of an ``(n, d)`` array. A row with any non-finite value is
    unusable: its distances are NaN (the diagonal is 0 for usable rows)."""
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected an (n, d) array, got shape {x.shape}")
    ok = np.isfinite(x).all(axis=1)
    safe = np.where(ok[:, None], x, 0.0)
    diff = safe[:, None, :] - safe[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=-1))
    d[~ok, :] = np.nan
    d[:, ~ok] = np.nan
    return d


def _quality_order(n: int, quality: Sequence[float] | np.ndarray | None) -> list[int]:
    """Indices best quality first; ties (and NaN quality, which ranks last) by index."""
    if quality is None:
        return list(range(n))
    q = np.asarray(quality, dtype=np.float64)
    if q.shape != (n,):
        raise ValueError(f"quality must have one value per item ({n}), got shape {q.shape}")
    key = np.where(np.isnan(q), -np.inf, q)
    return sorted(range(n), key=lambda i: (-key[i], i))


def deduplicate(
    distance: DistanceSource,
    threshold: float,
    *,
    quality: Sequence[float] | np.ndarray | None = None,
    n: int | None = None,
) -> list[int]:
    """Indices to keep after dropping near-duplicates, ascending.

    Items are visited best *quality* first (ties, or no quality, by index); an item is kept unless it is strictly
    closer than *threshold* to an item already kept. So from each group of duplicates the best one survives. An
    unusable (NaN/inf) distance never makes a duplicate. *n* is required with a distance callable.
    """
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(f"threshold must be a finite number >= 0, got {threshold}")
    rows = _Rows(distance, n)
    kept: list[int] = []
    for i in _quality_order(rows.n, quality):
        if kept:
            d = rows.row(i)[kept]
            if np.any(d < threshold):  # NaN compares False: never a duplicate
                continue
        kept.append(i)
    return sorted(kept)


def _medoid(rows: _Rows, candidates: np.ndarray) -> int:
    """The candidate with the smallest mean usable distance to the others; one with no usable distance ranks last.
    Ties by index."""
    best, best_key = int(candidates[0]), (1, np.inf)
    for i in candidates:
        r = rows.row(int(i))[candidates]
        r = r[~np.isnan(r)]
        key = (0, float(r.mean())) if len(r) else (1, np.inf)
        if key < best_key:
            best, best_key = int(i), key
    return best


def farthest_point_sample(
    distance: DistanceSource,
    k: int,
    *,
    n: int | None = None,
    start: int | Literal["medoid", "random"] | None = None,
    seed: int | None = None,
    clusters: Sequence[int | None] | np.ndarray | None = None,
    candidates: Sequence[int] | np.ndarray | None = None,
) -> list[int]:
    """Pick *k* mutually distant items by greedy max-min (farthest-point) sampling; returns them in pick order.

    Each step picks the candidate whose distance to its nearest already-picked item is largest. Ties go to the
    lowest index, or are broken by the seeded generator when *seed* is given.

    * *start*: the first pick. An index; ``"medoid"`` (the most central item); ``"random"`` (needs *seed*). The
      default is ``"random"`` when a *seed* is given and ``"medoid"`` otherwise, so the unseeded result is a pure
      function of the distances.
    * *clusters*: optional stratification, one cluster id per item (``None`` is its own group). The pick is made
      only among clusters represented least so far, so every cluster gets one pick before any gets a second, and
      so on. The start is the medoid (or random pick) of the largest cluster when stratifying, unless given.
    * *candidates*: restrict selection to these indices (the distances still index the full set).
    * ``k >= len(candidates)`` returns every candidate, still in farthest-point order; ``k <= 0`` returns ``[]``.

    *n* is required with a distance callable.
    """
    rows = _Rows(distance, n)
    pool = np.arange(rows.n) if candidates is None else np.unique(np.asarray(candidates, dtype=np.intp))
    if len(pool) and (pool.min() < 0 or pool.max() >= rows.n):
        raise ValueError("candidate index out of range")
    if k <= 0 or len(pool) == 0:
        return []
    k = min(int(k), len(pool))
    rng = np.random.default_rng(seed) if seed is not None else None

    groups: np.ndarray
    if clusters is None:
        groups = np.zeros(rows.n, dtype=np.int64)
    else:
        if len(clusters) != rows.n:
            raise ValueError(f"clusters must have one id per item ({rows.n}), got {len(clusters)}")
        ids = sorted({c for c in clusters if c is not None})
        none_id = (max(ids) + 1) if ids else 0
        groups = np.array([none_id if c is None else int(c) for c in clusters], dtype=np.int64)

    if start is None:
        start = "random" if rng is not None else "medoid"
    if isinstance(start, str):
        first_pool = pool
        if clusters is not None:
            gids, counts = np.unique(groups[pool], return_counts=True)
            first_pool = pool[groups[pool] == gids[np.argmax(counts)]]  # largest cluster; ties to the lowest id
        if start == "random":
            if rng is None:
                raise ValueError('start="random" needs a seed')
            first = int(rng.choice(first_pool))
        elif start == "medoid":
            first = _medoid(rows, first_pool)
        else:
            raise ValueError(f"unknown start {start!r}")
    else:
        first = int(start)
        if first not in set(pool.tolist()):
            raise ValueError(f"start {first} is not a candidate")

    picked = [first]
    remaining = np.zeros(rows.n, dtype=bool)
    remaining[pool] = True
    remaining[first] = False
    # Nearest usable distance to the picked set; NaN until some picked item is comparable.
    nearest = rows.row(first).copy()
    group_counts: dict[int, int] = {int(g): 0 for g in np.unique(groups[pool])}
    group_counts[int(groups[first])] += 1

    while len(picked) < k:
        live = np.flatnonzero(remaining)
        if clusters is not None:
            live_groups = {int(g) for g in groups[live]}
            least = min(group_counts[g] for g in live_groups)
            live = live[np.array([group_counts[int(g)] == least for g in groups[live]])]
        known = live[~np.isnan(nearest[live])]
        if len(known):
            vals = nearest[known]
            best = known[vals == vals.max()]
        else:
            best = live
        choice = int(best[0]) if rng is None or len(best) == 1 else int(rng.choice(best))
        picked.append(choice)
        remaining[choice] = False
        group_counts[int(groups[choice])] += 1
        r = rows.row(choice)
        nearest = np.fmin(nearest, r)  # fmin ignores NaN on either side
    return picked
