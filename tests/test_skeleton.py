from __future__ import annotations

import numpy as np
import pytest

from argus_puppet import skeleton as sk


def test_names_are_133_and_unique() -> None:
    assert len(sk.KEYPOINT_NAMES) == sk.NUM_KEYPOINTS == 133
    assert len(set(sk.KEYPOINT_NAMES)) == 133
    assert sk.KEYPOINT_NAMES[0] == "nose" and sk.KEYPOINT_NAMES[16] == "right_ankle"
    assert sk.KEYPOINT_NAMES[17] == "left_big_toe" and sk.KEYPOINT_NAMES[22] == "right_heel"
    assert sk.KEYPOINT_NAMES[23] == "face-0" and sk.KEYPOINT_NAMES[90] == "face-67"
    assert sk.KEYPOINT_NAMES[91] == "left_hand_root" and sk.KEYPOINT_NAMES[112] == "right_hand_root"
    assert sk.KEYPOINT_NAMES[132] == "right_pinky_finger4"
    assert all(sk.KEYPOINT_INDEX[n] == i for i, n in enumerate(sk.KEYPOINT_NAMES))


def test_parts_partition_the_layout() -> None:
    covered = [i for part in sk.PARTS.values() for i in range(sk.NUM_KEYPOINTS)[part]]
    assert covered == list(range(133))
    sizes = {name: len(range(133)[part]) for name, part in sk.PARTS.items()}
    assert sizes == {"body": 17, "feet": 6, "face": 68, "left_hand": 21, "right_hand": 21}
    assert sk.part_of(0) == "body" and sk.part_of(22) == "feet" and sk.part_of(90) == "face"
    assert sk.part_of(111) == "left_hand" and sk.part_of(132) == "right_hand"
    with pytest.raises(IndexError):
        sk.part_of(133)


def test_named_joints_match_names() -> None:
    for joint in sk.Joint:
        assert sk.KEYPOINT_NAMES[joint] == joint.name.lower()
    assert sk.L_SHOULDER == 5 and sk.R_HIP == 12 and sk.R_ANKLE == 16


def test_flip_pairs_are_disjoint_and_swap_left_for_right() -> None:
    assert len(sk.FLIP_PAIRS) == 61
    flat = [i for pair in sk.FLIP_PAIRS for i in pair]
    assert len(flat) == len(set(flat))
    assert all(0 <= i < 133 for i in flat)
    for a, b in sk.FLIP_PAIRS:
        na, nb = sk.KEYPOINT_NAMES[a], sk.KEYPOINT_NAMES[b]
        if na.startswith(("left", "right")):
            assert na.replace("left", "X").replace("right", "left").replace("X", "right") == nb


def test_flip_permutation_is_an_involution() -> None:
    perm = sk.FLIP_PERMUTATION
    assert perm.shape == (133,)
    assert sorted(perm.tolist()) == list(range(133))
    assert (perm[perm] == np.arange(133)).all()
    unpaired = {i for i in range(133)} - {i for pair in sk.FLIP_PAIRS for i in pair}
    assert all(perm[i] == i for i in unpaired)
    assert sk.NOSE in unpaired and sk.FACE.start + 8 in unpaired  # nose and chin stay put
    with pytest.raises(ValueError):
        perm[0] = 1  # read-only


def test_flip_horizontal() -> None:
    rng = np.random.default_rng(0)
    kp = np.concatenate([rng.uniform(0, 100, (2, 133, 2)), rng.uniform(0, 1, (2, 133, 1))], axis=-1)
    flipped = sk.flip_horizontal(kp, 100)
    assert flipped.shape == kp.shape
    assert flipped[0, sk.R_WRIST, 0] == pytest.approx(99 - kp[0, sk.L_WRIST, 0])
    assert flipped[0, sk.R_WRIST, 1:].tolist() == kp[0, sk.L_WRIST, 1:].tolist()
    assert np.allclose(sk.flip_horizontal(flipped, 100), kp)
    with pytest.raises(ValueError):
        sk.flip_horizontal(np.zeros((17, 3)), 100)


def test_limbs_and_hand_edges() -> None:
    assert len(sk.BODY_LIMBS) == 19 and all(a < 17 and b < 17 for a, b in sk.BODY_LIMBS)
    assert len(sk.HAND_EDGES) == 20 and {i for e in sk.HAND_EDGES for i in e} == set(range(21))
    assert sk.LEFT_HAND_EDGES[0] == (91, 92) and sk.RIGHT_HAND_EDGES[-1] == (131, 132)


def test_openpose_constants() -> None:
    assert len(sk.OPENPOSE18_NAMES) == len(sk.OPENPOSE18_FROM_WHOLEBODY) == 18
    assert len(sk.OPENPOSE_LIMBS) == 17 and len(sk.OPENPOSE_COLORS) == 18
    one_based = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13]]
    one_based += [[13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
    assert [list(limb) for limb in sk.OPENPOSE_LIMBS] == [[a - 1, b - 1] for a, b in one_based]
    for name, src in zip(sk.OPENPOSE18_NAMES, sk.OPENPOSE18_FROM_WHOLEBODY, strict=True):
        assert (src is None) == (name == "neck")
        if src is not None:
            assert sk.KEYPOINT_NAMES[src] == name
    assert sk.OPENPOSE_COLORS[0] == (255, 0, 0) and sk.OPENPOSE_COLORS[-1] == (255, 0, 85)


def test_to_openpose18() -> None:
    kp = np.zeros((133, 3))
    kp[:, 0] = np.arange(133)
    kp[:, 1] = np.arange(133) * 2
    kp[:, 2] = 0.9
    kp[sk.L_SHOULDER] = (10.0, 20.0, 0.8)
    kp[sk.R_SHOULDER] = (30.0, 40.0, 0.6)
    op = sk.to_openpose18(kp)
    assert op.shape == (18, 3)
    assert op[sk.OPENPOSE_NECK].tolist() == pytest.approx([20.0, 30.0, 0.6])
    assert op[0].tolist() == kp[0].tolist()
    assert op[2].tolist() == kp[sk.R_SHOULDER].tolist()  # OpenPose index 2 is the right shoulder
    assert op[17].tolist() == kp[sk.L_EAR].tolist()
    batched = sk.to_openpose18(np.stack([kp, kp]))
    assert batched.shape == (2, 18, 3) and np.allclose(batched[1], op)


def test_part_mean_scores() -> None:
    kp = np.zeros((133, 3))
    kp[sk.BODY, 2] = 1.0
    kp[sk.LEFT_HAND, 2] = 0.5
    kp[sk.FACE.start, 2] = np.nan
    scores = sk.part_mean_scores(kp)
    assert scores == {"body": 1.0, "feet": 0.0, "face": 0.0, "left_hand": 0.5, "right_hand": 0.0}
