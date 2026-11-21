from __future__ import annotations

import numpy as np
import pytest
from posegen import PRESETS, make, result

from argus_puppet.compare import (
    DEFAULT_MEMORISATION_THRESHOLD,
    compare_pose_sets,
    coverage_divergence,
    js_divergence,
)
from argus_puppet.metrics import coverage
from argus_puppet.models import MetricParams, PoseLabels, PoseSetComparison

NAMES = list(PRESETS)
# The reference library leaves two presets out, so outputs drawn from them are genuinely new poses.
HELD_OUT = ("walk", "floor_sit")
REF_NAMES = [n for n in NAMES if n not in HELD_OUT]


def reference_set(rng: np.random.Generator, n: int = 40):
    return [
        result(f"train/{k:03d}.png", make(REF_NAMES[k % len(REF_NAMES)], rng=rng, angle_jitter=12, joint_noise=0.02))
        for k in range(n)
    ]


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
    assert by_image["out/copy.png"].memorised
    assert by_image["out/copy.png"].distance < DEFAULT_MEMORISATION_THRESHOLD
    assert not by_image["out/fresh-a.png"].memorised and not by_image["out/fresh-b.png"].memorised
    assert cmp.memorised_share == pytest.approx(1 / 3)
    assert (cmp.output_count, cmp.matched_count, cmp.reference_count) == (3, 3, 40)
    assert PoseSetComparison.model_validate_json(cmp.model_dump_json()) == cmp


def test_a_mirrored_copy_is_still_a_copy() -> None:
    ref = reference_set(np.random.default_rng(1))
    target = [r for r in ref if r.image == "train/005.png"][0].primary_person.keypoints_array()
    cmp = compare_pose_sets([result("o.png", target)], ref)
    assert cmp.matches[0].distance == 0.0
    flipped = make(NAMES[5], mirror=True, scale=60.0)
    cmp2 = compare_pose_sets([result("o.png", flipped)], [result("t.png", make(NAMES[5]))])
    assert cmp2.matches[0].distance == pytest.approx(0.0, abs=1e-5) and cmp2.matches[0].memorised


def test_same_stance_different_pose_is_not_memorised() -> None:
    cmp = compare_pose_sets([result("o.png", make("hands_on_hips"))], [result("t.png", make("standing"))])
    assert cmp.matches[0].distance > DEFAULT_MEMORISATION_THRESHOLD and not cmp.matches[0].memorised


def test_unusable_outputs_are_listed_unmatched_and_left_out_of_the_share() -> None:
    sparse = make("t_pose")
    sparse[:17, 2] = 0.0
    sparse[[5, 6, 11, 12, 0, 1, 2, 3, 4], 2] = 0.9  # usable (9 joints) but no limb joint: comparable with nothing
    outputs = [result("none.png", None), result("ok.png", make("t_pose")), result("sparse.png", sparse)]
    cmp = compare_pose_sets(outputs, [result("t.png", make("t_pose"))])
    assert [m.image for m in cmp.matches] == ["none.png", "ok.png", "sparse.png"]
    by = {m.image: m for m in cmp.matches}
    for unmatched in ("none.png", "sparse.png"):
        assert by[unmatched].nearest is None and by[unmatched].distance is None and not by[unmatched].memorised
    assert by["ok.png"].nearest == "t.png" and by["ok.png"].memorised
    assert (cmp.output_count, cmp.matched_count) == (3, 1)
    assert cmp.memorised_share == 1.0  # over the matched output only
    assert PoseSetComparison.model_validate_json(cmp.model_dump_json()) == cmp


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
    assert cmp.memorised_share == 0.0 and cmp.reference_count == 0
    assert (cmp.output_count, cmp.matched_count) == (1, 0)
    assert cmp.coverage_divergence == 1.0  # one side has poses, the other has none


def test_identical_sets_have_zero_coverage_divergence_and_are_all_memorised() -> None:
    ref = reference_set(np.random.default_rng(2), 20)
    cmp = compare_pose_sets(ref, ref)
    assert cmp.coverage_divergence == 0.0 and cmp.memorised_share == 1.0
    assert all(m.nearest == m.image and m.distance == 0.0 for m in cmp.matches)


def test_threshold_is_respected_and_recorded() -> None:
    cmp = compare_pose_sets([result("o.png", make("hands_on_hips"))], [result("t.png", make("standing"))], threshold=1)
    assert cmp.threshold == 1 and cmp.matches[0].memorised
    with pytest.raises(ValueError):
        compare_pose_sets([], [], threshold=-1)


def test_matches_do_not_depend_on_input_order() -> None:
    rng = np.random.default_rng(3)
    ref = reference_set(rng)
    out = [result(f"o{k}.png", make(NAMES[k], rng=rng, angle_jitter=20)) for k in range(len(NAMES))]
    a = compare_pose_sets(out, ref)
    b = compare_pose_sets(list(reversed(out)), [ref[i] for i in rng.permutation(len(ref))])
    assert a.model_dump() == b.model_dump()


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
