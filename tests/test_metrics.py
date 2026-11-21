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


def _all_upgma_partitions(d: np.ndarray, threshold: float) -> set[frozenset[frozenset[int]]]:
    """Every partition the textbook algorithm can reach when exact ties may be broken either way (NaN = inf)."""
    d = np.where(np.isfinite(d), d, np.inf)
    seen: set[frozenset[frozenset[int]]] = set()
    out: set[frozenset[frozenset[int]]] = set()

    def link(a: frozenset[int], b: frozenset[int]) -> float:
        return float(np.mean([d[i, j] for i in a for j in b]))

    def walk(clusters: frozenset[frozenset[int]]) -> None:
        if clusters in seen:
            return
        seen.add(clusters)
        pairs = [(link(a, b), a, b) for a, b in itertools.combinations(sorted(clusters, key=min), 2)]
        best = min((v for v, _, _ in pairs), default=np.inf)
        if not best < threshold:
            out.add(clusters)
            return
        for v, a, b in pairs:
            if np.isclose(v, best, rtol=0, atol=1e-9):
                walk(clusters - {a, b} | {a | b})

    walk(frozenset(frozenset([i]) for i in range(len(d))))
    return out


def _partition(labels: np.ndarray) -> frozenset[frozenset[int]]:
    return frozenset(frozenset(np.flatnonzero(labels == r).tolist()) for r in set(labels.tolist()))


@pytest.mark.parametrize("kind", ["nan", "ties", "duplicates"])
def test_average_linkage_matches_the_textbook_algorithm_on_awkward_matrices(kind: str) -> None:
    rng = np.random.default_rng({"nan": 50, "ties": 51, "duplicates": 52}[kind])
    for _ in range(60):
        n = int(rng.integers(3, 9))
        p = rng.normal(size=(n, 2)) * rng.uniform(0.3, 2)
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        if kind == "nan":
            m = rng.random((n, n)) < 0.2
            m = np.triu(m, 1)
            d[m | m.T] = np.nan
        elif kind == "ties":
            d = np.round(d * 2)  # integer distances: many exact ties
            np.fill_diagonal(d, 0.0)
        else:
            idx = rng.integers(0, n, size=n)
            d = d[np.ix_(idx, idx)]  # repeated rows: zero distances between duplicates
        for threshold in (0.5, 1.0, 2.0, 3.0):
            labels = average_linkage(d, threshold)
            assert _partition(labels) in _all_upgma_partitions(d, threshold)
            assert np.array_equal(labels, average_linkage(d.copy(), threshold))  # deterministic


def test_tied_clusters_do_not_depend_on_input_order_once_sorted() -> None:
    # Exact ties make UPGMA's partition depend on the order it meets them; pose_set_stats sorts by image name first,
    # so the stats do not depend on the order the results arrive in even then.
    tri = [make("standing", tx=100 + 37 * k) for k in range(3)]  # three identical poses: every pair ties at 0
    rs = [result(f"{k}.png", kp) for k, kp in enumerate([*tri, make("t_pose"), make("t_pose", tx=50)])]
    rng = np.random.default_rng(9)
    first = pose_set_stats(rs).model_dump()
    for _ in range(5):
        assert pose_set_stats([rs[i] for i in rng.permutation(len(rs))]).model_dump() == first


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


def gate_rejects(s) -> bool:
    """The gate docs/metrics.md suggests to argus-proof (on sets with enough usable, comparable poses)."""
    return s.diversity < 0.3 or s.largest_cluster_share >= 0.75


def mixed(seed: int, share: float, n: int = 200):
    """*share* of the set one jump shot (5 deg, 0.03), the rest the broad mix."""
    rng = np.random.default_rng(seed)
    k = round(share * n)
    return sample(rng, ["star_jump"], k, prefix="a") + sample(rng, NAMES, n - k, angle=10, prefix="b")


def test_diversity_calibration_pins_the_scale_and_the_formula() -> None:
    # Bands tight enough that diversity_scale 0.25 (0.094 / 0.228) or 0.35 (0.049 / 0.126), or the non-squared
    # 1 - exp(-d / s) (0.224 / 0.339), would fall outside them.
    jump = div(sample(np.random.default_rng(5), ["star_jump"], 80))
    assert 0.055 <= jump <= 0.08, jump
    loose = div(sample(np.random.default_rng(7), ["star_jump"], 200, angle=8, noise=0.05))
    assert 0.14 <= loose <= 0.19, loose
    tuck = div(sample(np.random.default_rng(55), ["jump_tuck"], 80))
    assert 0.03 <= tuck <= 0.065, tuck


def test_diversity_calibration_a_broad_mix_scores_high() -> None:
    assert 0.86 <= div(sample(np.random.default_rng(6), NAMES, 120, angle=10)) <= 0.9


def test_suggested_gate_rejects_collapse_and_passes_variety() -> None:
    for share, seed in ((1.0, 7), (0.9, 13), (0.85, 14), (0.8, 15)):
        s = pose_set_stats(mixed(seed, share))
        assert gate_rejects(s), (share, s.diversity, s.largest_cluster_share)
    assert pose_set_stats(mixed(13, 0.9)).diversity < 0.3  # diversity alone rejects 90 % one pose
    for pool, seed in ((["star_jump", "seated"], 17), (["standing", "t_pose"], 31)):
        s = pose_set_stats(sample(np.random.default_rng(seed), pool, 100))
        assert not gate_rejects(s), (pool, s.diversity, s.largest_cluster_share)
    assert not gate_rejects(pose_set_stats(mixed(16, 0.7)))


def test_one_hip_lost_in_half_the_frames_does_not_lift_a_collapsed_set_over_the_gate() -> None:
    # Regression: falling back to the one confident hip scored this set 0.35, over the 0.3 gate.
    rng = np.random.default_rng(70)
    rs = sample(rng, ["star_jump"], 200, angle=8, noise=0.05)
    base = div(rs)
    for score in (0.2, 0.0):
        dropped = []
        for k, r in enumerate(rs):
            kp = r.people[0].keypoints_array()
            if k % 2:
                kp[11 + k % 4 // 2, 2] = score  # one hip, alternating sides
            dropped.append(result(r.image, kp))
        s = pose_set_stats(dropped)
        assert s.diversity - base < 0.05 and s.diversity < 0.25 and s.largest_cluster_share == 1.0


def test_diversity_is_monotone_in_spread() -> None:
    rng = np.random.default_rng(7)
    values = [div(sample(rng, ["wave"], 60, angle=a, noise=0.01)) for a in (0, 4, 8, 16, 32)]
    assert values == sorted(values) and values[0] < values[-1]


def test_diversity_is_not_dominated_by_one_outlier() -> None:
    rng = np.random.default_rng(8)
    n = 40
    base = sample(rng, ["standing"], n)
    with_outlier = base + [result("zz-outlier.png", make("lying"))]
    a, b = pose_set_stats(base), pose_set_stats(with_outlier)
    # One pose touches n of the n(n+1)/2 pairs, so it can add at most 2(1 - diversity) / (n + 1); a lying pose is
    # different from every standing one, so it adds nearly that, and no more.
    bound = 2 * (1 - a.diversity) / (n + 1)
    assert 0.9 * bound <= b.diversity - a.diversity <= bound + 1e-9
    # The raw mean distance, by contrast, is pulled up by far more than its share of the pairs.
    assert b.mean_pairwise_distance > 1.5 * a.mean_pairwise_distance


def test_diversity_is_stable_as_n_grows() -> None:
    # An unbiased (U-) statistic: the mean over many 6-pose sets matches one 400-pose set. A V-statistic counting
    # the diagonal would come out (n - 1) / n = 5/6 as large here (0.74 against 0.87).
    small = np.mean([div(sample(np.random.default_rng(1000 + k), NAMES, 6, angle=10)) for k in range(40)])
    large = div(sample(np.random.default_rng(10), NAMES, 400, angle=10))
    assert abs(small - large) < 0.04


def test_comparable_pairs_expose_the_denominator() -> None:
    # Regression: 98 unusable images and 2 different poses scored 0.9 with nothing saying how little was measured.
    rs = [result(f"u{i:02d}.png", None) for i in range(98)]
    rs += [result("a.png", make("standing")), result("b.png", make("arms_up"))]
    s = pose_set_stats(rs)
    assert (s.usable, s.comparable_pairs, s.comparable_share) == (2, 1, 1.0)
    legs_only = make("standing", dropout=(7, 8, 9, 10))
    arms_only = make("standing", dropout=(13, 14, 15, 16))
    s = pose_set_stats([result(f"{k}.png", kp) for k, kp in enumerate((legs_only, legs_only, arms_only, arms_only))])
    assert s.usable == 4 and s.comparable_pairs == 2 and s.comparable_share == pytest.approx(2 / 6)
    assert pose_set_stats([result("a.png", make("standing"))]).comparable_share == 0.0


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


def test_every_param_is_honoured_and_recorded() -> None:
    rng = np.random.default_rng(11)
    rs = sample(rng, NAMES, 12)
    base = pose_set_stats(rs)
    assert base.params == DEFAULT_PARAMS == MetricParams()
    wide = pose_set_stats(rs, params=MetricParams(diversity_scale=3.0))
    assert wide.params.diversity_scale == 3.0 and wide.diversity < base.diversity
    assert wide.mean_pairwise_distance == pytest.approx(base.mean_pairwise_distance)
    # Hide both ankles everywhere: with 16 shared joints required, no pair is comparable any more.
    hidden = []
    for r in rs:
        kp = r.people[0].keypoints_array()
        kp[[15, 16], 2] = 0.0
        hidden.append(result(r.image, kp))
    strict = pose_set_stats(hidden, params=MetricParams(min_shared_joints=16))
    assert strict.params.min_shared_joints == 16
    assert strict.mean_pairwise_distance == 0.0 and strict.cluster_count == strict.usable  # every pose alone
    assert pose_set_stats(hidden).mean_pairwise_distance > 0  # the default (6 shared) still compares them
    no_limbs = pose_set_stats(hidden, params=MetricParams(min_shared_limb_joints=7))
    assert no_limbs.mean_pairwise_distance == 0.0  # only 6 limb joints left in each pose


# ── stored labels are never trusted (wire 1.1) ──────────────────────────────────────────────────────────────────


def _with_labels(rs, make_labels):
    return [
        r.model_copy(update={"people": [p.model_copy(update={"labels": make_labels(p)}) for p in r.people]}, deep=True)
        for r in rs
    ]


@pytest.mark.parametrize("params", [MetricParams(), MetricParams(min_joint_score=0.6, min_usable_joints=6)])
def test_stored_person_labels_never_change_stats_compare_or_library(params) -> None:
    from argus_puppet.compare import compare_pose_sets
    from argus_puppet.labels import pose_labels
    from argus_puppet.library import build_library
    from argus_puppet.models import PoseLabels

    rng = np.random.default_rng(31)
    v10 = sample(rng, NAMES, 30)  # a 1.0 sidecar: no labels
    ref10 = sample(rng, NAMES, 20, prefix="r")
    assert all(p.labels is None for r in v10 + ref10 for p in r.people)

    def v11(rs):  # what extraction writes today
        return _with_labels(rs, lambda p: pose_labels(p.keypoints_array()))

    def poisoned(rs):  # stale or tampered labels
        return _with_labels(rs, lambda p: PoseLabels(stance="lying", arms="mixed", facing="back"))

    want_stats = pose_set_stats(v10, params=params).model_dump()
    want_cmp = compare_pose_sets(v10, ref10, params=params).model_dump()
    want_lib = build_library(v10).library.model_dump()
    for variant in (v11, poisoned):
        assert pose_set_stats(variant(v10), params=params).model_dump() == want_stats
        assert compare_pose_sets(variant(v10), variant(ref10), params=params).model_dump() == want_cmp
        assert build_library(variant(v10)).library.model_dump() == want_lib
    assert {i["labels"]["stance"] for i in want_stats["items"]} - {"lying", "unknown"}  # the poison is a change
