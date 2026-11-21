from __future__ import annotations

import math

import numpy as np
import pytest
from posegen import PRESETS, make

from argus_puppet import embedding
from argus_puppet.embedding import (
    BODY_FLIP,
    PoseVector,
    align_to_mean,
    canonicalize,
    cross_distances,
    embed,
    mirror,
    mirror_vector,
    pairwise_distances,
    pose_distance,
    stack,
    torso_frame,
)
from argus_puppet.skeleton import FACE, LEFT_HAND, Joint, flip_horizontal


def vec(name: str, **kw) -> PoseVector:
    v = embed(make(name, **kw))
    assert v is not None
    return v


# ── the vector ───────────────────────────────────────────────────────────────────────────────────────────────────


def test_vector_is_centred_on_the_mid_hip_and_in_torso_units() -> None:
    v = vec("standing", scale=137.0, tx=321.0, ty=654.0)
    mid_hip = (v.coords[Joint.LEFT_HIP] + v.coords[Joint.RIGHT_HIP]) / 2
    mid_sh = (v.coords[Joint.LEFT_SHOULDER] + v.coords[Joint.RIGHT_SHOULDER]) / 2
    assert np.allclose(mid_hip, 0.0)
    assert math.isclose(float(np.hypot(*(mid_sh - mid_hip))), 1.0, rel_tol=1e-9)
    assert math.isclose(v.scale_px, 137.0, rel_tol=1e-9)
    assert v.coords.shape == (17, 2) and v.mask.all()


def test_low_confidence_joints_are_masked_not_zero_filled_into_the_distance() -> None:
    kp = make("standing")
    kp[Joint.LEFT_WRIST, 2] = 0.29
    v = embed(kp)
    assert v is not None and not v.mask[Joint.LEFT_WRIST]
    assert np.all(v.coords[Joint.LEFT_WRIST] == 0.0)
    # Moving a masked joint anywhere changes nothing.
    kp2 = kp.copy()
    kp2[Joint.LEFT_WRIST, :2] = (-5000.0, 9000.0)
    assert pose_distance(v, embed(kp2)) == 0.0


def test_feet_hands_and_face_stay_out_of_the_vector() -> None:
    kp = make("t_pose")
    other = kp.copy()
    other[17:, :2] += 333.0
    other[LEFT_HAND, 2] = 1.0
    other[FACE, 2] = 1.0
    a, b = embed(kp), embed(other)
    assert np.array_equal(a.coords, b.coords) and np.array_equal(a.mask, b.mask)


@pytest.mark.parametrize(
    "drop",
    [
        (Joint.LEFT_HIP, Joint.RIGHT_HIP),
        (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER),
    ],
)
def test_no_anchor_is_unusable(drop: tuple[int, ...]) -> None:
    assert embed(make("standing", dropout=drop)) is None


def test_doubtful_anchor_joints_never_anchor_a_pose_on_their_own() -> None:
    for pair in ((Joint.LEFT_HIP, Joint.RIGHT_HIP), (Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER)):
        kp = make("standing")
        kp[list(pair), 2] = 0.2  # both doubtful: above the anchor floor, below the joint threshold
        assert embed(kp) is None and torso_frame(kp) is None


def test_one_hip_and_one_shoulder_are_enough_to_anchor() -> None:
    assert embed(make("profile", dropout=(Joint.RIGHT_HIP, Joint.RIGHT_SHOULDER))) is not None


def test_too_few_joints_is_unusable() -> None:
    limbs = (7, 8, 9, 10, 13, 14, 15, 16, 1, 2, 3, 4)
    assert embed(make("standing", dropout=limbs)) is None  # 5 joints left
    assert embed(make("standing", dropout=limbs[:8]), min_usable_joints=9) is not None
    assert embed(make("standing", dropout=limbs[:8]), min_usable_joints=10) is None


def test_a_tiny_person_is_unusable() -> None:
    assert embed(make("standing", scale=2.0)) is None


def test_foreshortened_torso_takes_the_scale_from_the_shoulders() -> None:
    kp = make("standing", scale=100.0)
    # The subject leans toward the camera: the shoulders drop to just above the hips; widths are unchanged.
    kp[[Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER], 1] = kp[Joint.LEFT_HIP, 1] - 15.0
    frame = torso_frame(kp)
    assert frame is not None
    assert math.isclose(frame.scale_px, embedding.SHOULDER_WIDTH_TO_TORSO * 70.0, rel_tol=1e-9)
    assert frame.foreshortening < 0.25


def test_mirror_is_an_involution_and_swaps_sides() -> None:
    v = vec("wave")
    c, m = mirror(v.coords, v.mask)
    assert np.allclose(c[Joint.LEFT_WRIST], [-v.coords[Joint.RIGHT_WRIST][0], v.coords[Joint.RIGHT_WRIST][1]])
    c2, m2 = mirror(c, m)
    assert np.array_equal(c2, v.coords) and np.array_equal(m2, v.mask)
    assert np.array_equal(BODY_FLIP[BODY_FLIP], np.arange(17))


def test_embedding_a_flipped_image_is_the_mirror_vector() -> None:
    for drop in ((), (Joint.RIGHT_HIP,), (Joint.LEFT_SHOULDER,)):  # also with an estimated anchor joint
        kp = make("wave", dropout=drop)
        a = mirror_vector(embed(kp))
        b = embed(flip_horizontal(kp, 1000))
        assert np.allclose(a.coords, b.coords) and np.array_equal(a.mask, b.mask)
        assert pose_distance(b, mirror_vector(b)) == 0.0
        assert pose_distance(embed(kp), b) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name", ["standing", "t_pose", "back", "profile", "seated", "lying", "star_jump"])
@pytest.mark.parametrize("joint", [Joint.LEFT_HIP, Joint.RIGHT_HIP, Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER])
@pytest.mark.parametrize("score", [0.29, 0.05, 0.0])
def test_one_anchor_joint_crossing_the_threshold_barely_moves_the_vector(name: str, joint: int, score: float) -> None:
    # Regression: falling back to the one confident hip moved a frontal pose's centre by half a hip width (0.23).
    kp = make(name)
    dipped = kp.copy()
    dipped[joint, 2] = score
    v = embed(dipped)
    assert v is not None and not v.mask[joint]  # still masked as a compared joint
    assert pose_distance(embed(kp), v) <= 0.03


def test_one_anchor_joint_crossing_the_threshold_with_jitter_stays_small() -> None:
    rng = np.random.default_rng(12)
    names = ["standing", "t_pose", "back", "profile", "seated", "crouch", "star_jump", "hands_on_hips"]
    ds = []
    for k in range(200):
        kp = make(names[k % len(names)], rng=rng, angle_jitter=10, joint_noise=0.02)
        for joint in (Joint.LEFT_HIP, Joint.RIGHT_HIP, Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER):
            dipped = kp.copy()
            dipped[joint, 2] = 0.0
            ds.append(pose_distance(embed(kp), embed(dipped)))
    assert np.percentile(ds, 95) <= 0.06 and max(ds) <= 0.1


def test_a_shoulder_dipping_keeps_the_scale_of_a_foreshortened_torso() -> None:
    kp = make("standing", scale=100.0)
    kp[:11, 1] = kp[Joint.LEFT_HIP, 1] + (kp[:11, 1] - kp[Joint.LEFT_HIP, 1]) * 0.3  # torso toward the camera
    for score in (0.29, 0.0):
        dipped = kp.copy()
        dipped[Joint.RIGHT_SHOULDER, 2] = score
        assert pose_distance(embed(kp), embed(dipped)) <= 0.06  # was 0.18: the shoulder width set the scale


def test_one_hip_and_one_shoulder_without_an_estimate_anchor_on_the_joints_themselves() -> None:
    kp = make("profile", dropout=(Joint.RIGHT_HIP, Joint.RIGHT_SHOULDER))
    frame = torso_frame(kp)
    assert np.allclose(frame.mid_hip, kp[Joint.LEFT_HIP, :2]) and np.allclose(
        frame.mid_shoulder, kp[Joint.LEFT_SHOULDER, :2]
    )


def test_huge_finite_coordinates_are_unplaced() -> None:
    base = make("wave")
    for value in (1e40, 1e300, 2e6):
        kp = base.copy()
        kp[Joint.LEFT_EAR, :2] = value
        v = embed(kp)
        assert not v.mask[Joint.LEFT_EAR]
        assert pose_distance(v, embed(base)) == pytest.approx(0.0, abs=1e-6)  # was inf (float32 overflow)


# ── distance ─────────────────────────────────────────────────────────────────────────────────────────────────────


def test_distance_is_zero_on_identity_and_on_the_mirror_image() -> None:
    for name in PRESETS:
        v = vec(name)
        assert pose_distance(v, v) == 0.0
        assert pose_distance(v, vec(name, mirror=True)) == pytest.approx(0.0, abs=1e-5)
        assert pose_distance(v, mirror_vector(v)) == 0.0


def test_distance_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    names = list(PRESETS)
    for _ in range(30):
        a = vec(names[rng.integers(len(names))], rng=rng, angle_jitter=10, joint_noise=0.03)
        b = vec(names[rng.integers(len(names))], rng=rng, angle_jitter=10, joint_noise=0.03)
        assert pose_distance(a, b) == pytest.approx(pose_distance(b, a), abs=1e-6)


def test_distance_is_invariant_to_translation_and_uniform_scale() -> None:
    a = vec("star_jump", rng=np.random.default_rng(2), angle_jitter=6)
    b = vec("arms_up")
    b2 = vec("arms_up", scale=37.0, tx=12.0, ty=900.0)
    assert pose_distance(a, b) == pytest.approx(pose_distance(a, b2), abs=1e-5)


def test_distance_grows_with_joint_displacement() -> None:
    base = make("standing")
    v = embed(base)
    last = 0.0
    for dx in (5, 10, 20, 40, 80):
        kp = base.copy()
        kp[Joint.LEFT_WRIST, 0] += dx
        kp[Joint.LEFT_ANKLE, 1] -= dx
        d = pose_distance(v, embed(kp))
        assert d > last
        last = d


def test_distance_is_not_rotation_invariant() -> None:
    assert pose_distance(vec("standing"), vec("lying")) > 1.0


def test_too_few_shared_joints_is_nan() -> None:
    legs = (Joint.LEFT_KNEE, Joint.RIGHT_KNEE, Joint.LEFT_ANKLE, Joint.RIGHT_ANKLE)
    arms = (Joint.LEFT_ELBOW, Joint.RIGHT_ELBOW, Joint.LEFT_WRIST, Joint.RIGHT_WRIST)
    a = embed(make("standing", dropout=legs), min_usable_joints=6)
    b = embed(make("standing", dropout=arms), min_usable_joints=6)
    # Head and torso are shared, but no limb joint is.
    assert math.isnan(pose_distance(a, b))
    assert pose_distance(a, b, min_shared_limb_joints=0) == pytest.approx(0.0, abs=1e-6)


def test_matrices_agree_with_the_pair_function_across_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(3)
    names = list(PRESETS)
    vs = []
    for _ in range(23):
        drop = tuple(int(j) for j in rng.choice(np.arange(5, 17), size=rng.integers(0, 5), replace=False))
        drop = tuple(j for j in drop if j not in (Joint.LEFT_HIP, Joint.LEFT_SHOULDER))
        v = embed(make(names[rng.integers(len(names))], rng=rng, angle_jitter=10, dropout=drop), min_usable_joints=6)
        if v is not None:
            vs.append(v)
    c, m = stack(vs)
    full = pairwise_distances(c, m, dtype=np.float64)
    monkeypatch.setattr(embedding, "_CHUNK_ELEMENTS", 100)  # one row per chunk
    chunked = pairwise_distances(c, m, dtype=np.float64)
    cross = cross_distances(c, m, c, m)
    assert np.array_equal(full, full.T)
    np.testing.assert_allclose(full, chunked, atol=1e-6, equal_nan=True)
    for i, a in enumerate(vs):
        for j, b in enumerate(vs):
            want = 0.0 if i == j else pose_distance(a, b)
            np.testing.assert_allclose(full[i, j], want, atol=1e-6, equal_nan=True)
            if i != j:
                np.testing.assert_allclose(cross[i, j], want, atol=1e-6, equal_nan=True)


def test_empty_and_single_matrices() -> None:
    c, m = stack([])
    assert pairwise_distances(c, m).shape == (0, 0)
    assert cross_distances(c, m, c, m).shape == (0, 0)
    c1, m1 = stack([vec("standing")])
    assert pairwise_distances(c1, m1).tolist() == [[0.0]]


# ── canonical orientation ────────────────────────────────────────────────────────────────────────────────────────


def test_canonicalize_is_mirror_invariant() -> None:
    for name in PRESETS:
        v = vec(name, rng=np.random.default_rng(4), angle_jitter=5)
        a = canonicalize(v.coords, v.mask)
        b = canonicalize(*mirror(v.coords, v.mask))
        assert np.allclose(a[0], b[0]) and np.array_equal(a[1], b[1]), name


def test_align_to_mean_is_order_independent_and_mirror_invariant() -> None:
    rng = np.random.default_rng(5)
    vs = [vec("wave", rng=rng, angle_jitter=8, mirror=bool(rng.integers(2))) for _ in range(12)]
    c, m = stack(vs)
    a, am = align_to_mean(c, m)
    perm = rng.permutation(len(vs))
    b, bm = align_to_mean(c[perm], m[perm])
    assert np.allclose(a[perm], b) and np.array_equal(am[perm], bm)
    # Every aligned copy of the same (jittered) pose ends up on the same side.
    assert np.ptp(a[:, Joint.LEFT_WRIST, 1]) < 0.5


def test_nearest_neighbours_agree_with_the_matrix_across_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_puppet.embedding import CORE_JOINTS, nearest_neighbours

    rng = np.random.default_rng(7)
    names = list(PRESETS)
    vs = []
    for k in range(30):
        drop = tuple(int(j) for j in rng.choice([7, 8, 9, 10, 13, 14, 15, 16], size=rng.integers(0, 3), replace=False))
        vs.append(vec(names[k % len(names)], rng=rng, angle_jitter=10, joint_noise=0.02, dropout=drop))
    c, m = stack(vs)
    full = pairwise_distances(c, m, dtype=np.float64)
    for chunk in (1_500_000, 100):
        monkeypatch.setattr(embedding, "_CHUNK_ELEMENTS", chunk)
        nn = nearest_neighbours(c, m, c, m, exclude_self=True)
        for i in range(len(vs)):
            row = full[i].copy()
            row[i] = np.nan
            dd = np.where(np.isfinite(row), row, np.inf)
            assert nn.index[i] == int(np.argmin(dd)) and nn.distance[i] == pytest.approx(dd.min(), abs=1e-6)
            j = int(nn.index[i])
            assert nn.shared[i] == int((m[i][list(CORE_JOINTS)] & m[j][list(CORE_JOINTS)]).sum()) or (
                nn.shared[i] == int((m[i] & mirror(c[j], m[j])[1])[list(CORE_JOINTS)].sum())
            )
            whole = [k for k in range(len(vs)) if k != i and m[k][list(CORE_JOINTS)].all()]
            if m[i][list(CORE_JOINTS)].all() and whole:
                best = min(whole, key=lambda k: (full[i, k], k))
                assert nn.full_index[i] == best and nn.full_shared[i] == 12
            else:
                assert nn.full_index[i] == -1 and np.isnan(nn.full_distance[i])
    empty = nearest_neighbours(c[:0], m[:0], c, m)
    assert len(empty.index) == 0
    with pytest.raises(ValueError):
        nearest_neighbours(c[:3], m[:3], c, m, exclude_self=True)
