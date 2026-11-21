from __future__ import annotations

import numpy as np
import pytest

from argus_puppet.sampling import deduplicate, euclidean_distances, farthest_point_sample


def blobs(centres: list[tuple[float, float]], per: int, spread: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([np.asarray(c) + rng.normal(0, spread, (per, 2)) for c in centres])


# ── distances ────────────────────────────────────────────────────────────────────────────────────────────────────


def test_euclidean_distances() -> None:
    x = np.array([[0.0, 0.0], [3.0, 4.0], [np.nan, 1.0]])
    d = euclidean_distances(x)
    assert d[0, 1] == d[1, 0] == 5.0 and d[0, 0] == 0.0
    assert np.isnan(d[2]).all() and np.isnan(d[:, 2]).all()
    with pytest.raises(ValueError):
        euclidean_distances(np.zeros(3))


# ── dedupe ───────────────────────────────────────────────────────────────────────────────────────────────────────


def test_dedupe_keeps_one_per_planted_group() -> None:
    x = blobs([(0, 0), (5, 0), (0, 5)], per=4)
    kept = deduplicate(euclidean_distances(x), 1.0)
    assert kept == [0, 4, 8]  # the first of each group, no quality given


def test_dedupe_keeps_the_best_quality() -> None:
    x = blobs([(0, 0), (5, 0)], per=3)
    quality = [0.1, 0.9, 0.5, 0.2, 0.3, 0.8]
    assert deduplicate(euclidean_distances(x), 1.0, quality=quality) == [1, 5]
    # NaN quality ranks last
    assert deduplicate(euclidean_distances(x), 1.0, quality=[np.nan, 0.1, 0.0, 0, 0, 0]) == [1, 3]


def test_dedupe_threshold_is_strict_and_zero_keeps_all() -> None:
    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert deduplicate(d, 1.0) == [0, 1]
    assert deduplicate(d, 1.01) == [0]
    assert deduplicate(np.zeros((3, 3)), 0.0) == [0, 1, 2]


def test_dedupe_never_treats_an_unusable_pair_as_a_duplicate() -> None:
    d = np.array([[0.0, np.nan, np.inf], [np.nan, 0.0, 0.1], [np.inf, 0.1, 0.0]])
    assert deduplicate(d, 1.0) == [0, 1]


def test_dedupe_with_a_callable() -> None:
    x = blobs([(0, 0), (5, 0)], per=3)
    d = euclidean_distances(x)
    assert deduplicate(lambda i, j: d[i, j], 1.0, n=6) == deduplicate(d, 1.0)
    with pytest.raises(ValueError):
        deduplicate(lambda i, j: 0.0, 1.0)  # n missing
    with pytest.raises(ValueError):
        deduplicate(d, -1.0)
    with pytest.raises(ValueError):
        deduplicate(d, 1.0, quality=[1.0])


def test_dedupe_empty() -> None:
    assert deduplicate(np.zeros((0, 0)), 1.0) == []


# ── farthest-point sampling ──────────────────────────────────────────────────────────────────────────────────────


def test_fps_picks_one_per_planted_cluster() -> None:
    centres = [(0, 0), (10, 0), (0, 10), (10, 10), (5, 5)]
    x = blobs(centres, per=6)
    picks = farthest_point_sample(euclidean_distances(x), 5)
    assert sorted(i // 6 for i in picks) == [0, 1, 2, 3, 4]


def test_fps_unseeded_is_deterministic_and_starts_at_the_medoid() -> None:
    x = blobs([(0, 0), (10, 0), (0, 10), (10, 10), (5, 5)], per=6)
    d = euclidean_distances(x)
    a, b = farthest_point_sample(d, 4), farthest_point_sample(d, 4)
    assert a == b
    assert a[0] // 6 == 4  # the central blob holds the medoid


def test_fps_seed_is_reproducible_and_varies_the_start() -> None:
    x = blobs([(0, 0), (10, 0), (0, 10), (10, 10)], per=10)
    d = euclidean_distances(x)
    assert farthest_point_sample(d, 3, seed=7) == farthest_point_sample(d, 3, seed=7)
    starts = {farthest_point_sample(d, 1, seed=s)[0] for s in range(20)}
    assert len(starts) > 5


def test_fps_explicit_start_and_errors() -> None:
    d = euclidean_distances(np.arange(10.0)[:, None])
    assert farthest_point_sample(d, 3, start=0) == [0, 9, 4]
    with pytest.raises(ValueError):
        farthest_point_sample(d, 3, start="random")  # no seed
    with pytest.raises(ValueError):
        farthest_point_sample(d, 3, start="middle")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        farthest_point_sample(np.zeros((2, 3)), 1)


def test_fps_on_a_line_spreads_out() -> None:
    d = euclidean_distances(np.arange(11.0)[:, None])
    picks = farthest_point_sample(d, 3)
    assert picks[0] == 5 and sorted(picks[1:]) == [0, 10]


def test_fps_n_at_or_above_population_returns_everything() -> None:
    d = euclidean_distances(np.arange(5.0)[:, None])
    for k in (5, 50):
        picks = farthest_point_sample(d, k)
        assert sorted(picks) == [0, 1, 2, 3, 4] and len(picks) == 5
    assert farthest_point_sample(d, 0) == [] and farthest_point_sample(d, -3) == []
    assert farthest_point_sample(np.zeros((0, 0)), 3) == []


def test_fps_stratifies_across_clusters() -> None:
    # A big tight cluster (ids 0) and two small ones; plain FPS would take the big cluster's spread first.
    big = np.c_[np.linspace(0, 100, 20), np.zeros(20)]
    small = np.array([[50.0, 1.0], [50.0, 1.5], [51.0, 2.0]])
    x = np.r_[big, small]
    clusters = [0] * 20 + [1, 1, 2]
    picks = farthest_point_sample(euclidean_distances(x), 3, clusters=clusters)
    assert sorted(clusters[i] for i in picks) == [0, 1, 2]
    picks = farthest_point_sample(euclidean_distances(x), 6, clusters=clusters)
    counts = np.bincount([clusters[i] for i in picks])
    assert counts.tolist() == [3, 2, 1]  # cluster 2 is exhausted after one; then the others alternate
    unstratified = farthest_point_sample(euclidean_distances(x), 3)
    assert sorted(clusters[i] for i in unstratified) != [0, 1, 2]


def test_fps_none_cluster_is_its_own_group() -> None:
    x = np.c_[np.linspace(0, 100, 10), np.zeros(10)]
    clusters: list[int | None] = [0] * 9 + [None]
    picks = farthest_point_sample(euclidean_distances(x), 2, clusters=clusters)
    assert 9 in picks
    with pytest.raises(ValueError):
        farthest_point_sample(euclidean_distances(x), 2, clusters=[0])


def test_fps_unusable_items_come_last() -> None:
    x = np.array([[0.0], [10.0], [np.nan], [5.0], [np.nan]])
    d = euclidean_distances(x)
    picks = farthest_point_sample(d, 5)
    assert picks[:3] == [3, 0, 1] and sorted(picks[3:]) == [2, 4]
    assert 2 not in farthest_point_sample(d, 3) and 4 not in farthest_point_sample(d, 3)


def test_fps_partly_unusable_pairs_use_the_known_ones() -> None:
    # 0 and 2 cannot be compared, but 2 is known to sit right next to 1.
    d = np.array(
        [
            [0.0, 10.0, np.nan, 5.0],
            [10.0, 0.0, 0.1, 5.0],
            [np.nan, 0.1, 0.0, 5.0],
            [5.0, 5.0, 5.0, 0.0],
        ]
    )
    picks = farthest_point_sample(d, 3, start=0)
    assert picks == [0, 1, 3]


def test_fps_candidates_restrict_the_pool() -> None:
    d = euclidean_distances(np.arange(10.0)[:, None])
    picks = farthest_point_sample(d, 10, candidates=[2, 3, 4])
    assert sorted(picks) == [2, 3, 4]
    with pytest.raises(ValueError):
        farthest_point_sample(d, 2, candidates=[0, 99])
    with pytest.raises(ValueError):
        farthest_point_sample(d, 2, candidates=[1, 2], start=5)


def test_fps_with_a_callable_matches_the_matrix() -> None:
    x = blobs([(0, 0), (10, 0), (0, 10)], per=5)
    d = euclidean_distances(x)
    calls: list[tuple[int, int]] = []

    def dist(i: int, j: int) -> float:
        calls.append((i, j))
        return float(d[i, j])

    assert farthest_point_sample(dist, 3, n=15, start=0) == farthest_point_sample(d, 3, start=0)
    assert calls  # it really used the callable
