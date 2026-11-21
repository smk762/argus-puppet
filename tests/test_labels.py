from __future__ import annotations

import numpy as np
import pytest
from posegen import EXPECTED_ARMS, EXPECTED_FACING, EXPECTED_STANCE, PRESETS, Figure, make

from argus_puppet.labels import pose_labels
from argus_puppet.models import PoseLabels
from argus_puppet.skeleton import FACE, Joint

MIRRORED_FACING = {"left": "right", "right": "left"}
VARIANTS = [
    {},
    {"scale": 40.0, "tx": 100.0, "ty": 900.0},
    {"scale": 450.0, "tx": -2000.0, "ty": 30.0},
    {"mirror": True},
    {"mirror": True, "scale": 70.0, "tx": 750.0},
]


def labels(name: str | Figure, **kw) -> PoseLabels:
    return pose_labels(make(name, **kw))


@pytest.mark.parametrize("name", sorted(EXPECTED_STANCE))
@pytest.mark.parametrize("variant", range(len(VARIANTS)))
def test_stance(name: str, variant: int) -> None:
    assert labels(name, **VARIANTS[variant]).stance == EXPECTED_STANCE[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_ARMS))
@pytest.mark.parametrize("variant", range(len(VARIANTS)))
def test_arms(name: str, variant: int) -> None:
    assert labels(name, **VARIANTS[variant]).arms == EXPECTED_ARMS[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_FACING))
@pytest.mark.parametrize("variant", range(len(VARIANTS)))
def test_facing_swaps_left_and_right_only_under_a_mirror(name: str, variant: int) -> None:
    kw = VARIANTS[variant]
    want = EXPECTED_FACING[name]
    if kw.get("mirror"):
        want = MIRRORED_FACING.get(want, want)
    assert labels(name, **kw).facing == want


@pytest.mark.parametrize("name", sorted(EXPECTED_STANCE))
def test_stance_survives_small_jitter(name: str) -> None:
    rng = np.random.default_rng(7)
    got = [labels(name, rng=rng, angle_jitter=4, joint_noise=0.01).stance for _ in range(20)]
    assert got.count(EXPECTED_STANCE[name]) >= 18, got


def test_every_preset_is_invariant_to_translation_scale_and_mirror() -> None:
    for name in PRESETS:
        base = labels(name)
        moved = labels(name, scale=222.0, tx=10.0, ty=10.0)
        flipped = labels(name, mirror=True)
        assert moved == base, name
        assert flipped.stance == base.stance and flipped.arms == base.arms, name
        assert flipped.facing == MIRRORED_FACING.get(base.facing, base.facing), name


# ── unknown rather than a guess ──────────────────────────────────────────────────────────────────────────────────


def test_nothing_confident_is_all_unknown() -> None:
    kp = make("standing")
    kp[:, 2] = 0.0
    assert pose_labels(kp) == PoseLabels()


def test_no_ankles_means_stance_unknown() -> None:
    # A vertical thigh without its shin could be standing or kneeling upright.
    assert labels("standing", dropout=(Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)).stance == "unknown"


def test_upper_body_shot_has_arms_but_no_stance() -> None:
    lab = labels("arms_up", dropout=(13, 14, 15, 16))
    assert lab.stance == "unknown" and lab.arms == "up" and lab.facing == "front"


def test_no_wrists_means_arms_unknown() -> None:
    assert labels("t_pose", dropout=(Joint.LEFT_WRIST, Joint.RIGHT_WRIST)).arms == "unknown"


def test_one_visible_arm_decides_alone() -> None:
    assert labels("t_pose", dropout=(Joint.RIGHT_WRIST,)).arms == "out"


def test_profile_without_a_nose_is_facing_unknown() -> None:
    assert labels("profile", dropout=(Joint.NOSE,)).facing == "unknown"


def test_back_view_with_a_hallucinated_face_is_unknown() -> None:
    kp = make("back")
    kp[[Joint.NOSE, Joint.LEFT_EYE], 2] = 0.9  # a face where the shoulder order says we see the back
    assert pose_labels(kp).facing == "unknown"


def test_face_landmarks_count_as_a_visible_face() -> None:
    kp = make("standing", dropout=(Joint.NOSE, Joint.LEFT_EYE, Joint.RIGHT_EYE))
    assert pose_labels(kp).facing == "unknown"
    kp[FACE, 2] = 0.8
    assert pose_labels(kp).facing == "front"


def test_torso_pointing_at_the_camera_is_stance_unknown() -> None:
    kp = make("standing")
    kp[[Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER], 1] = kp[Joint.LEFT_HIP, 1] - 10.0
    assert pose_labels(kp).stance == "unknown"


def test_bending_over_on_straight_legs_is_standing() -> None:
    fig = Figure(view="right", lean=85, l_arm=(0, 0), r_arm=(0, 0))
    assert labels(fig).stance == "standing"


def test_inverted_is_unknown() -> None:
    assert labels(Figure(rot=180)).stance == "unknown"


def test_wide_stance_with_arms_down_is_not_a_star_jump() -> None:
    fig = Figure(l_leg=(25, 25, 1), r_leg=(25, 25, 1))
    assert labels(fig).stance == "standing"


def test_squat_with_knees_level_is_crouching_not_a_tuck() -> None:
    fig = Figure(l_arm=(60, 80), r_arm=(60, 80), l_leg=(95, -30, 1), r_leg=(95, -30, 1), view="left")
    assert labels(fig).stance == "crouching"


def test_min_joint_score_is_respected() -> None:
    kp = make("t_pose")
    kp[:, 2] = np.where(kp[:, 2] > 0, 0.5, 0.0)
    assert pose_labels(kp, min_joint_score=0.4).arms == "out"
    assert pose_labels(kp, min_joint_score=0.6) == PoseLabels()
