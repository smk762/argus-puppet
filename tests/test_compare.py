from __future__ import annotations

import numpy as np
import pytest
from posegen import PRESETS, make, result
from test_metrics import sample

from argus_puppet.compare import (
    DEFAULT_CLOSE_THRESHOLD,
    compare_pose_sets,
    coverage_divergence,
    js_divergence,
)
from argus_puppet.embedding import embed
from argus_puppet.metrics import coverage
from argus_puppet.models import MetricParams, PoseLabels, PoseSetComparison

NAMES = list(PRESETS)
# The reference library leaves two presets out, so outputs drawn from them are genuinely new poses.
HELD_OUT = ("walk", "floor_sit")
REF_NAMES = [n for n in NAMES if n not in HELD_OUT]
WAIST_UP = (13, 14, 15, 16)  # knees and ankles out of frame


def reference_set(rng: np.random.Generator, n: int = 40):
    return [
        result(f"train/{k:03d}.png", make(REF_NAMES[k % len(REF_NAMES)], rng=rng, angle_jitter=12, joint_noise=0.02))
        for k in range(n)
    ]


def copies(rng: np.random.Generator, refs, n: int, *, noise: float = 0.02, prefix: str = "copy"):
    """Re-estimated copies of reference frames: per-joint noise of *noise* torso units, another scale and place."""
    out = []
    for k in range(n):
        kp = refs[int(rng.integers(len(refs)))].people[0].keypoints_array().copy()
        kp[:23, :2] += rng.normal(0, noise * embed(kp).scale_px, size=(23, 2))
        kp[:, :2] = kp[:, :2] * rng.uniform(0.5, 2.0) + rng.uniform(-100, 100, 2)
        out.append(result(f"{prefix}{k:04d}.png", kp))
    return out


def test_finds_the_planted_near_duplicate() -> None:
    rng = np.random.default_rng(0)
    ref = reference_set(rng)
    planted = ref[17].primary_person.keypoints_array()
    copy = planted.copy()
    copy[:23, :2] += rng.normal(0, 0.01 * 100, size=(23, 2))  # estimator-level noise, 1% of a torso
    copy[:, :2] = copy[:, :2] * 0.6 + 40  # another scale and position
    outputs = [
        result("out/copy.png", copy),
        result("out/fresh-a.png", make("walk", angle_jitter=25, rng=rng)),
        result("out/fresh-b.png", make("floor_sit", angle_jitter=25, rng=rng)),
    ]
    cmp = compare_pose_sets(outputs, ref)
    by_image = {m.image: m for m in cmp.matches}
    assert by_image["out/copy.png"].nearest == "train/017.png"
    assert by_image["out/copy.png"].close and by_image["out/copy.png"].shared_joints == 12
    assert by_image["out/copy.png"].distance < DEFAULT_CLOSE_THRESHOLD
    assert not by_image["out/fresh-a.png"].close and not by_image["out/fresh-b.png"].close
    assert cmp.close_share == pytest.approx(1 / 3) and cmp.close_eligible == 3
    assert (cmp.output_count, cmp.matched_count, cmp.reference_count) == (3, 3, 40)
    assert PoseSetComparison.model_validate_json(cmp.model_dump_json()) == cmp


def test_a_mirrored_copy_is_still_a_copy() -> None:
    ref = reference_set(np.random.default_rng(1))
    target = [r for r in ref if r.image == "train/005.png"][0].primary_person.keypoints_array()
    cmp = compare_pose_sets([result("o.png", target)], ref)
    assert cmp.matches[0].distance == 0.0
    flipped = make(NAMES[5], mirror=True, scale=60.0)
    cmp2 = compare_pose_sets([result("o.png", flipped)], [result("t.png", make(NAMES[5]))])
    assert cmp2.matches[0].distance == pytest.approx(0.0, abs=1e-5) and cmp2.matches[0].close


def test_same_stance_different_pose_is_not_close() -> None:
    cmp = compare_pose_sets([result("o.png", make("hands_on_hips"))], [result("t.png", make("standing"))])
    assert cmp.matches[0].distance > DEFAULT_CLOSE_THRESHOLD and not cmp.matches[0].close


def test_independent_draws_of_one_stance_show_no_close_excess() -> None:
    # Regression: at 0.08 and with no baseline, independent draws of one generic pose were flagged "memorised" at
    # 0.58-1.00, and a broad mix against a broad mix at 0.87.
    for pose in ("standing", "hands_on_hips"):
        for m in (10, 50, 200):
            rng = np.random.default_rng(31)
            out = sample(rng, [pose], 200, angle=8, noise=0.03, prefix="o")
            ref = sample(rng, [pose], m, angle=8, noise=0.03, prefix="r")
            cmp = compare_pose_sets(out, ref)
            assert cmp.close_excess is not None and cmp.close_excess <= 0.08, (pose, m, cmp.close_excess)
            assert cmp.close_share <= 0.1
    rng = np.random.default_rng(33)
    broad = compare_pose_sets(sample(rng, NAMES, 300, angle=10, prefix="o"), sample(rng, NAMES, 200, angle=10))
    assert broad.close_share <= 0.05 and broad.close_excess <= 0.05


def test_planted_copies_show_a_large_close_excess() -> None:
    rng = np.random.default_rng(36)
    ref = sample(rng, ["standing"], 50, angle=8, noise=0.03, prefix="r")
    all_copies = compare_pose_sets(copies(rng, ref, 200), ref)
    assert all_copies.close_share >= 0.75 and all_copies.close_excess >= 0.7
    assert all_copies.output_nn_distance.p50 < 0.7 * all_copies.reference_nn_distance.p50
    rng = np.random.default_rng(35)
    ref = sample(rng, NAMES, 200, angle=10, prefix="r")
    some = compare_pose_sets(copies(rng, ref, 90) + sample(rng, NAMES, 210, angle=10, prefix="o"), ref)
    assert some.close_excess >= 0.15  # 30 % copies


def test_a_waist_up_output_is_never_close_to_a_full_body_frame() -> None:
    # Regression: only the shared upper-body joints were compared, so a waist-up standing output matched a full-body
    # seated frame at 0.051 (and a full-body standing frame at 0).
    waist_up = make("standing", dropout=WAIST_UP)
    for ref in ("standing", "seated_front"):
        cmp = compare_pose_sets([result("o.png", waist_up)], [result("t.png", make(ref))])
        m = cmp.matches[0]
        assert m.nearest == "t.png" and m.shared_joints == 8 and not m.close
        assert cmp.close_eligible == 0 and cmp.close_share == 0.0
    loose = MetricParams(min_close_joints=8)
    assert (
        compare_pose_sets([result("o.png", waist_up)], [result("t.png", make("standing"))], params=loose)
        .matches[0]
        .close
    )


def test_a_close_full_frame_wins_over_a_nearer_partial_one() -> None:
    out = make("standing")
    partial = make("standing", dropout=WAIST_UP)  # distance 0 over 8 shared joints
    full = out.copy()
    full[9, 0] += 2.0  # a wrist 0.02 torso units off: close over all 12
    cmp = compare_pose_sets([result("o.png", out)], [result("a-partial.png", partial), result("b-full.png", full)])
    m = cmp.matches[0]
    assert m.close and m.nearest == "b-full.png" and m.shared_joints == 12


def test_baseline_is_leave_one_out_and_null_without_two_reference_poses() -> None:
    ref = reference_set(np.random.default_rng(2), 20)
    same = compare_pose_sets(ref, ref)
    assert same.close_share == 1.0 and all(m.nearest == m.image and m.distance == 0.0 for m in same.matches)
    assert same.baseline_close_share is not None and same.baseline_close_share < 1.0  # never against itself
    assert same.close_excess == pytest.approx(1.0 - same.baseline_close_share)
    assert same.baseline_eligible == 20 and same.reference_nn_distance.n == 20
    twins = [result("t1.png", make("standing")), result("t2.png", make("standing", tx=10))]
    assert compare_pose_sets([result("o.png", make("t_pose"))], twins).baseline_close_share == 1.0
    one = compare_pose_sets([result("o.png", make("standing"))], [result("t.png", make("standing"))])
    assert one.baseline_close_share is None and one.close_excess is None and one.reference_nn_distance is None
    assert one.close_share == 1.0 and one.baseline_eligible == 0


def test_unusable_outputs_are_listed_unmatched_and_left_out_of_the_share() -> None:
    sparse = make("t_pose")
    sparse[:17, 2] = 0.0
    sparse[[5, 6, 11, 12, 0, 1, 2, 3, 4], 2] = 0.9  # usable (9 joints) but no limb joint: comparable with nothing
    outputs = [result("none.png", None), result("ok.png", make("t_pose")), result("sparse.png", sparse)]
    cmp = compare_pose_sets(outputs, [result("t.png", make("t_pose"))])
    assert [m.image for m in cmp.matches] == ["none.png", "ok.png", "sparse.png"]
    by = {m.image: m for m in cmp.matches}
    for unmatched in ("none.png", "sparse.png"):
        m = by[unmatched]
        assert m.nearest is None and m.distance is None and m.shared_joints is None and not m.close
    assert by["ok.png"].nearest == "t.png" and by["ok.png"].close
    # only the unusable output carries a reason: sparse.png is usable, just comparable with nothing
    assert [by[i].unusable_reason for i in ("none.png", "ok.png", "sparse.png")] == ["no_person", None, None]
    assert cmp.output_unusable_reasons == {"no_person": 1} and cmp.reference_unusable_reasons == {}
    assert (cmp.output_count, cmp.matched_count, cmp.close_eligible) == (3, 1, 1)
    assert cmp.close_share == 1.0  # over the eligible output only
    assert PoseSetComparison.model_validate_json(cmp.model_dump_json()) == cmp


def test_unusable_reasons_on_both_sides() -> None:
    legs = make("standing", dropout=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    outputs = [result("legs.png", legs), result("ok.png", make("t_pose"))]
    reference = [result("r_legs.png", legs), result("r_none.png", None), result("t.png", make("t_pose"))]
    cmp = compare_pose_sets(outputs, reference)
    by = {m.image: m for m in cmp.matches}
    assert by["legs.png"].unusable_reason == "no_confident_shoulder" and by["legs.png"].nearest is None
    assert by["ok.png"].unusable_reason is None
    assert cmp.output_unusable_reasons == {"no_confident_shoulder": 1}
    assert cmp.reference_unusable_reasons == {"no_person": 1, "no_confident_shoulder": 1}
    assert cmp.reference_count == 1
    data = cmp.model_dump()
    data["output_unusable_reasons"] = {"too_few_joints": 1}
    with pytest.raises(ValueError, match="output_unusable_reasons"):
        PoseSetComparison.model_validate(data)


def test_params_are_honoured_and_recorded() -> None:
    out, ref = [result("o.png", make("hands_on_hips"))], [result("t.png", make("standing"))]
    strict = MetricParams(min_shared_joints=17)  # every body joint shared: the synthetic figures hide none
    assert compare_pose_sets(out, ref, params=strict).params == strict
    hidden = make("standing")
    hidden[15:17, 2] = 0.0  # no ankles
    cmp = compare_pose_sets(out, [result("t.png", hidden)], params=strict)
    assert cmp.matches[0].nearest is None and cmp.matched_count == 0
    assert compare_pose_sets(out, [result("t.png", hidden)]).matched_count == 1  # the default (6 shared) matches


def test_empty_reference() -> None:
    cmp = compare_pose_sets([result("o.png", make("standing"))], [])
    assert [(m.image, m.nearest, m.distance) for m in cmp.matches] == [("o.png", None, None)]
    assert cmp.close_share == 0.0 and cmp.reference_count == 0 and cmp.baseline_close_share is None
    assert (cmp.output_count, cmp.matched_count) == (1, 0)
    assert cmp.coverage_divergence == 1.0  # one side has poses, the other has none


def test_identical_sets_have_zero_coverage_divergence() -> None:
    ref = reference_set(np.random.default_rng(2), 20)
    assert compare_pose_sets(ref, ref).coverage_divergence == 0.0


def test_threshold_is_respected_and_recorded() -> None:
    cmp = compare_pose_sets([result("o.png", make("hands_on_hips"))], [result("t.png", make("standing"))], threshold=1)
    assert cmp.threshold == 1 and cmp.matches[0].close
    with pytest.raises(ValueError):
        compare_pose_sets([], [], threshold=-1)


def test_matches_do_not_depend_on_input_order() -> None:
    rng = np.random.default_rng(3)
    ref = reference_set(rng)
    out = [result(f"o{k}.png", make(NAMES[k], rng=rng, angle_jitter=20)) for k in range(len(NAMES))]
    a = compare_pose_sets(out, ref)
    b = compare_pose_sets(list(reversed(out)), [ref[i] for i in rng.permutation(len(ref))])
    assert a.model_dump() == b.model_dump()


def test_nearest_matches_brute_force_with_partial_poses() -> None:
    from argus_puppet.embedding import pose_distance

    rng = np.random.default_rng(30)
    outs = sample(rng, NAMES, 40, angle=10, prefix="o")
    refs = sample(rng, NAMES, 50, angle=10, prefix="r")
    for rs in (outs, refs):
        for k, r in enumerate(rs):
            kp = r.people[0].keypoints_array()
            kp[rng.choice(np.r_[7:11, 13:17], size=rng.integers(0, 6), replace=False), 2] = 0
            rs[k] = result(r.image, kp)
    cmp = compare_pose_sets(outs, refs)
    vr = {r.image: embed(r.people[0].keypoints_array()) for r in refs}
    for m, o in zip(cmp.matches, sorted(outs, key=lambda r: r.image), strict=True):
        vo = embed(o.people[0].keypoints_array())
        cands = [(pose_distance(vo, vr[n]), n) for n in sorted(vr) if vo is not None and vr[n] is not None]
        cands = [(d, n) for d, n in cands if np.isfinite(d)]
        if not cands:
            assert m.nearest is None
            continue
        if not m.close:
            d, n = min(cands, key=lambda c: c[0])
            assert m.nearest == n and m.distance == pytest.approx(d, abs=1e-5)


# ── Jensen-Shannon ───────────────────────────────────────────────────────────────────────────────────────────────


def test_js_divergence_bounds() -> None:
    assert js_divergence([3, 1, 0], [6, 2, 0]) == 0.0
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    assert js_divergence({"a": 2}, {"b": 5}) == pytest.approx(1.0)
    assert js_divergence([0, 0], [0, 0]) == 0.0
    mid = js_divergence([1, 1], [1, 0])
    assert 0.0 < mid < 1.0
    assert js_divergence([1, 0], [1, 1]) == pytest.approx(mid)  # symmetric
    with pytest.raises(ValueError):
        js_divergence([1, 2], [1])


def test_coverage_divergence_is_the_mean_over_the_three_kinds() -> None:
    a = coverage([PoseLabels(stance="standing", arms="up", facing="front")] * 4)
    b = coverage([PoseLabels(stance="lying", arms="up", facing="front")] * 4)
    assert coverage_divergence(a, a) == 0.0
    assert coverage_divergence(a, b) == pytest.approx(1 / 3)
