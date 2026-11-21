"""The primary subject: curator's area x score rule, or the body that best matches a primary-face hint."""

from __future__ import annotations

import pytest
from conftest import person

from argus_puppet.primary import assign_primary, match_score, primary_index, prominence


def test_nobody_has_no_primary() -> None:
    assert primary_index([], []) is None
    assert assign_primary([]) == ([], None)


def test_area_weighted_by_score_as_curator_does() -> None:
    boxes = [(0, 0, 10, 10), (0, 0, 11, 11), (0, 0, 10, 10)]
    assert primary_index(boxes, [0.9, 0.3, 1.0]) == 2  # 100*1.0 > 121*0.65 > 100*0.95
    assert prominence((0, 0, 10, 10), 1.0) == 100.0
    assert prominence((0, 0, 10, 10), 0.0) == 50.0


def test_the_first_index_wins_a_tie() -> None:
    assert primary_index([(0, 0, 10, 10), (50, 50, 10, 10)], [0.8, 0.8]) == 0


def test_match_score_is_mean_of_iou_and_face_coverage() -> None:
    body, face = (0, 0, 100, 200), (40, 10, 20, 20)
    iou = 400 / (20000 + 400 - 400)
    assert match_score(body, face) == pytest.approx((iou + 1.0) / 2)
    assert match_score(body, (500, 500, 10, 10)) == 0.0


def test_a_face_hint_picks_the_body_that_holds_that_face() -> None:
    boxes = [(0, 0, 300, 600), (400, 0, 100, 200)]  # the big one would win on area
    scores = [0.9, 0.9]
    assert primary_index(boxes, scores) == 0
    assert primary_index(boxes, scores, face_hint=(430, 10, 30, 30)) == 1


def test_a_face_hint_that_matches_nobody_falls_back_to_area() -> None:
    boxes = [(0, 0, 300, 600), (400, 0, 100, 200)]
    assert primary_index(boxes, [0.9, 0.9], face_hint=(900, 900, 10, 10)) == 0


def test_assign_flags_exactly_one_without_touching_the_input() -> None:
    people = [person(0, 0, 10, 10), person(0, 0, 50, 50), person(5, 5, 20, 20)]
    flagged, idx = assign_primary(people)
    assert idx == 1
    assert [p.primary for p in flagged] == [False, True, False]
    assert not any(p.primary for p in people)
