from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
from posegen import PRESETS, make, result

from argus_puppet.metrics import (
    ARMS,
    DEFAULT_PARAMS,
    FACINGS,
    STANCES,
    average_linkage,
    diversity,
    pose_set_stats,
)
from argus_puppet.models import SIDECAR_SUFFIX, MetricParams, PoseSetStats
from argus_puppet.sidecars import SidecarError, load_sidecars

NAMES = list(PRESETS)


def sample(rng: np.random.Generator, pool: list[str], n: int, *, angle: float = 5.0, noise: float = 0.03, prefix="i"):
    """n jittered poses drawn from *pool*, each at a random scale, position and mirroring."""
    return [
        result(
            f"{prefix}{k:05d}.png",
            make(
                pool[int(rng.integers(len(pool)))],
                rng=rng,
                angle_jitter=angle,
                joint_noise=noise,
                scale=float(rng.uniform(50, 300)),
                tx=float(rng.uniform(200, 800)),
                ty=float(rng.uniform(200, 800)),
                mirror=bool(rng.integers(2)),
            ),
        )
        for k in range(n)
    ]


# ── basic shape ──────────────────────────────────────────────────────────────────────────────────────────────────


def test_empty_set() -> None:
    s = pose_set_stats([])
    assert (s.count, s.usable, s.diversity, s.mean_pairwise_distance, s.cluster_count) == (0, 0, 0.0, 0.0, 0)
    assert s.largest_cluster_share == 0.0 and s.items == [] and s.params == DEFAULT_PARAMS
    assert set(s.coverage.stance) == set(STANCES) and set(s.coverage.arms) == set(ARMS)
    assert set(s.coverage.facing) == set(FACINGS)


def test_one_usable_pose() -> None:
    s = pose_set_stats([result("a.png", make("standing")), result("b.png", None)])
    assert (s.count, s.usable, s.diversity, s.cluster_count, s.largest_cluster_share) == (2, 1, 0.0, 1, 1.0)
    a, b = s.items
    assert a.usable and a.cluster == 0 and a.xy == (0.0, 0.0)
    assert not b.usable and b.cluster is None and b.xy is None and b.labels.stance == "unknown"
    assert s.clusters[0].medoid == "a.png"


def test_unusable_items_keep_their_labels_but_stay_out_of_the_numbers() -> None:
    upper_body_only = make("arms_up", dropout=(1, 2, 3, 4, 13, 14, 15, 16, 7, 8))  # 7 joints: unusable
    s = pose_set_stats([result("a.png", upper_body_only), result("b.png", make("t_pose"))])
    assert s.usable == 1
    item = s.items[0]
    assert not item.usable and item.labels.arms == "up"
    assert sum(s.coverage.arms.values()) == 1 and s.coverage.arms["out"] == 1


def test_n_identical_poses_have_zero_diversity_and_one_cluster() -> None:
    s = pose_set_stats([result(f"{i}.png", make("star_jump", tx=100 + i, scale=80 + i)) for i in range(10)])
    assert s.diversity == pytest.approx(0.0, abs=1e-9)
    assert s.mean_pairwise_distance == pytest.approx(0.0, abs=1e-5)
    assert s.cluster_count == 1 and s.largest_cluster_share == 1.0


def test_validates_against_the_wire_model() -> None:
    s = pose_set_stats(sample(np.random.default_rng(0), NAMES, 40))
    again = PoseSetStats.model_validate_json(s.model_dump_json())
    assert again == s
    assert sum(c.size for c in s.clusters) == s.usable
    assert all(0.0 <= c.share <= 1.0 for c in s.clusters)
    assert json.loads(s.model_dump_json())["puppet_version"]


def test_items_are_sorted_and_stats_do_not_depend_on_input_order() -> None:
    rng = np.random.default_rng(1)
    rs = sample(rng, NAMES, 60, angle=10)
    rs.append(result("zz-nobody.png", None))
    a = pose_set_stats(rs)
    b = pose_set_stats([rs[i] for i in rng.permutation(len(rs))])
    assert a.model_dump() == b.model_dump()
    assert [it.image for it in a.items] == sorted(it.image for it in a.items)


# ── clustering ───────────────────────────────────────────────────────────────────────────────────────────────────


def test_clusters_separate_distinct_poses_and_merge_jittered_copies() -> None:
    rng = np.random.default_rng(2)
    rs = []
    for g, (name, n) in enumerate((("lying", 12), ("star_jump", 9), ("seated", 6))):
        rs += sample(rng, [name], n, angle=4, noise=0.02, prefix=f"g{g}-")
    s = pose_set_stats(rs)
    assert [c.size for c in s.clusters] == [12, 9, 6]
    for item in s.items:
        assert item.cluster == int(item.image[1])  # largest first: g0 -> 0, g1 -> 1, g2 -> 2
    for c in s.clusters:
        assert c.medoid.startswith(f"g{c.id}-")
    assert s.largest_cluster_share == pytest.approx(12 / 27)


def test_cluster_ids_break_size_ties_by_first_image_name() -> None:
    rng = np.random.default_rng(3)
    rs = sample(rng, ["lying"], 5, prefix="b-") + sample(rng, ["arms_up"], 5, prefix="a-")
    s = pose_set_stats(rs)
    assert [c.size for c in s.clusters] == [5, 5]
    assert s.clusters[0].medoid.startswith("a-") and s.clusters[1].medoid.startswith("b-")


def _naive_upgma(d: np.ndarray, threshold: float) -> list[frozenset[int]]:
    clusters = [frozenset([i]) for i in range(len(d))]
    while len(clusters) > 1:
        best, pair = np.inf, None
        for a, b in itertools.combinations(range(len(clusters)), 2):
            v = np.mean([d[i, j] for i in clusters[a] for j in clusters[b]])
            if v < best:
                best, pair = v, (a, b)
        if not best < threshold:
            break
        a, b = pair
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)] + [clusters[a] | clusters[b]]
    return sorted(clusters, key=min)


@pytest.mark.parametrize("seed", range(8))
def test_average_linkage_matches_the_textbook_algorithm(seed: int) -> None:
    rng = np.random.default_rng(seed)
    pts = np.concatenate([rng.normal(c, 0.4, size=(8, 2)) for c in ((0, 0), (3, 0), (0, 3))])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    for threshold in (0.5, 1.0, 2.0, 4.0):
        labels = average_linkage(d, threshold)
        got = sorted((frozenset(np.flatnonzero(labels == r).tolist()) for r in set(labels.tolist())), key=min)
        assert got == _naive_upgma(d, threshold)


def test_average_linkage_never_joins_an_incomparable_pair() -> None:
    d = np.array([[0, 0.1, np.nan], [0.1, 0, 0.1], [np.nan, 0.1, 0]])
    labels = average_linkage(d, 1.0)
    assert labels[0] == labels[1] and labels[2] != labels[0]


def test_cluster_threshold_is_read_from_params() -> None:
    rs = sample(np.random.default_rng(4), NAMES, 40, angle=8)
    tight = pose_set_stats(rs, params=MetricParams(min_joint_score=0.3, min_usable_joints=8, cluster_threshold=0.01))
    loose = pose_set_stats(rs, params=MetricParams(min_joint_score=0.3, min_usable_joints=8, cluster_threshold=50))
    assert tight.cluster_count == tight.usable
    assert loose.cluster_count == 1


# ── diversity ────────────────────────────────────────────────────────────────────────────────────────────────────


def div(rs) -> float:
    return pose_set_stats(rs).diversity


def test_diversity_calibration_one_jump_shot_scores_low() -> None:
    rng = np.random.default_rng(5)
    for pose in ("star_jump", "jump_tuck"):
        d = div(sample(rng, [pose], 80))
        assert 0.03 <= d <= 0.15, (pose, d)


def test_diversity_calibration_a_broad_mix_scores_high() -> None:
    assert div(sample(np.random.default_rng(6), NAMES, 120, angle=10)) >= 0.7


def test_diversity_is_monotone_in_spread() -> None:
    rng = np.random.default_rng(7)
    values = [div(sample(rng, ["wave"], 60, angle=a, noise=0.01)) for a in (0, 4, 8, 16, 32)]
    assert values == sorted(values) and values[0] < values[-1]


def test_diversity_is_not_dominated_by_one_outlier() -> None:
    rng = np.random.default_rng(8)
    base = sample(rng, ["standing"], 40)
    with_outlier = base + [result("zz-outlier.png", make("lying"))]
    assert div(with_outlier) - div(base) < 0.06


def test_diversity_is_stable_as_n_grows() -> None:
    small = div(sample(np.random.default_rng(9), NAMES, 40, angle=10))
    large = div(sample(np.random.default_rng(10), NAMES, 400, angle=10))
    assert abs(small - large) < 0.06


def test_diversity_is_zero_without_comparable_pairs() -> None:
    assert diversity(np.array([[0.0, np.nan], [np.nan, 0.0]])) == 0.0
    assert diversity(np.zeros((1, 1))) == 0.0


# ── projection ───────────────────────────────────────────────────────────────────────────────────────────────────


def test_projection_separates_groups_and_is_deterministic() -> None:
    rng = np.random.default_rng(11)
    rs = sample(rng, ["lying"], 10, prefix="a-") + sample(rng, ["arms_up"], 10, prefix="b-")
    s = pose_set_stats(rs)
    xy = np.array([it.xy for it in s.items])
    assert np.isfinite(xy).all()
    a, b = xy[:10].mean(axis=0), xy[10:].mean(axis=0)
    spread = max(xy[:10].std(axis=0).max(), xy[10:].std(axis=0).max())
    assert np.linalg.norm(a - b) > 5 * spread
    assert pose_set_stats(list(reversed(rs))).items == s.items


def test_projection_ignores_mirroring() -> None:
    rs = [result(f"{i}.png", make("wave", mirror=bool(i % 2), tx=300 + 10 * i)) for i in range(6)]
    xy = np.array([it.xy for it in pose_set_stats(rs).items])
    assert np.ptp(xy, axis=0).max() < 1e-6


# ── sidecars ─────────────────────────────────────────────────────────────────────────────────────────────────────


def test_load_sidecars_feeds_the_stats(tmp_path) -> None:
    rs = sample(np.random.default_rng(12), NAMES, 5)
    (tmp_path / "sub").mkdir()
    for k, r in enumerate(rs):
        where = tmp_path / "sub" if k % 2 else tmp_path
        (where / f"{k}{SIDECAR_SUFFIX}").write_text(r.model_dump_json())
    (tmp_path / "other.json").write_text("{}")  # another package's sidecar: ignored
    loaded = load_sidecars(tmp_path)
    assert len(loaded) == 5
    assert pose_set_stats(loaded).model_dump() == pose_set_stats(rs).model_dump()
    assert len(load_sidecars(tmp_path, recursive=False)) == 3


def test_load_sidecars_reports_or_skips_a_bad_file(tmp_path) -> None:
    (tmp_path / f"ok{SIDECAR_SUFFIX}").write_text(result("ok.png", make("standing")).model_dump_json())
    (tmp_path / f"bad{SIDECAR_SUFFIX}").write_text("{not json")
    with pytest.raises(SidecarError, match="bad"):
        load_sidecars(tmp_path)
    assert [r.image for r in load_sidecars(tmp_path, skip_invalid=True)] == ["ok.png"]


def test_load_sidecars_needs_a_directory(tmp_path) -> None:
    with pytest.raises(NotADirectoryError):
        load_sidecars(tmp_path / "missing")
