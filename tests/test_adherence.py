"""Pose adherence: does a generated image hold the pose it was asked for? (library, POST /adherence, the CLI)

Every pose is a synthetic figure (``posegen``); no image is involved.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from conftest import FakeEstimator, draw_image
from fastapi.testclient import TestClient
from posegen import make, result
from typer.testing import CliRunner

from argus_puppet import estimator as estimator_mod
from argus_puppet.adherence import pose_adherence
from argus_puppet.cli import app
from argus_puppet.embedding import embed, oriented_distances, pose_distance
from argus_puppet.extract import write_sidecar
from argus_puppet.models import (
    DEFAULT_ADHERENCE_THRESHOLD,
    DEFAULT_CLUSTER_THRESHOLD,
    BarePose,
    PersonPose,
    PoseAdherence,
    PoseResult,
    sidecar_path,
)
from argus_puppet.server.app import create_app

runner = CliRunner()


def two(image: str, *figures: np.ndarray) -> PoseResult:
    """A PoseResult holding several people (the first is primary)."""
    base = result(image, figures[0])
    people = [
        PersonPose.from_array(kp, bbox=(0.0, 0.0, 10.0, 10.0), det_score=0.9, primary=i == 0)
        for i, kp in enumerate(figures)
    ]
    return base.model_copy(update={"people": people, "primary": 0})


def bare(*figures: np.ndarray) -> BarePose:
    return BarePose(keypoints=[kp.tolist() for kp in figures], width=1000, height=1000)


# ── the distance: left and right matter ──────────────────────────────────────────────────────────────────────────


def test_oriented_distances_split_what_the_mirror_invariant_distance_merges() -> None:
    wave, flipped = embed(make("wave")), embed(make("wave", mirror=True))
    posed, mirrored = oriented_distances(wave, flipped)
    assert mirrored == pytest.approx(0.0, abs=1e-9) and posed > 0.3  # one raised arm, on the other side
    assert pose_distance(wave, flipped) == pytest.approx(min(posed, mirrored))
    same = oriented_distances(wave, wave)
    assert same[0] == pytest.approx(0.0, abs=1e-9) and same[1] == pytest.approx(posed)


def test_oriented_distances_are_nan_when_too_few_joints_are_shared() -> None:
    legs = embed(make("standing", dropout=(7, 8, 9, 10)), min_usable_joints=1)  # no elbows or wrists
    arms = embed(make("standing", dropout=(13, 14, 15, 16)), min_usable_joints=1)  # no knees or ankles
    posed, mirrored = oriented_distances(legs, arms, min_shared_limb_joints=2)
    assert np.isnan(posed) and np.isnan(mirrored)


# ── the verdict ──────────────────────────────────────────────────────────────────────────────────────────────────


def test_the_same_pose_at_another_size_and_place_adheres() -> None:
    target = result("target.png", make("star_jump"))
    output = result("out.png", make("star_jump", scale=40.0, tx=200.0, ty=300.0, joint_noise=0.02))
    got = pose_adherence(target, output)
    assert got.adheres and got.threshold == DEFAULT_ADHERENCE_THRESHOLD == DEFAULT_CLUSTER_THRESHOLD == 0.25
    (m,) = got.matches
    assert (m.target, m.output, m.close, m.mirrored) == (0, 0, True, False)
    assert m.distance < 0.05 and got.distance == m.distance == got.mean_distance
    assert (got.target_people, got.output_people, got.comparable, got.matched, got.close) == (1, 1, 1, 1, 1)
    assert got.estimator is not None  # both are PoseResults naming one estimator


def test_another_pose_does_not_adhere() -> None:
    got = pose_adherence(result("t.png", make("arms_up")), result("o.png", make("seated")))
    assert not got.adheres and got.close == 0 and got.matched == 1
    assert got.matches[0].distance > 0.25 and not got.matches[0].mirrored


def test_a_mirror_image_does_not_adhere_and_is_named() -> None:
    got = pose_adherence(result("t.png", make("wave")), result("o.png", make("wave", mirror=True)))
    assert not got.adheres
    (m,) = got.matches
    assert not m.close and m.mirrored and m.mirrored_distance < 0.01 < 0.25 < m.distance


def test_a_symmetric_pose_is_close_either_way_and_never_called_mirrored() -> None:
    got = pose_adherence(result("t.png", make("t_pose")), result("o.png", make("t_pose", mirror=True)))
    assert got.adheres and not got.matches[0].mirrored


def test_the_threshold_decides() -> None:
    target, output = result("t.png", make("standing")), result("o.png", make("hands_on_hips"))
    d = pose_adherence(target, output).matches[0].distance
    assert pose_adherence(target, output, threshold=d + 0.01).adheres
    assert not pose_adherence(target, output, threshold=d).adheres  # strictly below
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="threshold"):
            pose_adherence(target, output, threshold=bad)


# ── several people ───────────────────────────────────────────────────────────────────────────────────────────────


def test_each_target_person_is_matched_to_their_own_output_person() -> None:
    target = two("t.png", make("arms_up"), make("seated"))
    output = two("o.png", make("seated", tx=700.0), make("arms_up", tx=200.0), make("lying"))
    got = pose_adherence(target, output)
    assert got.adheres and [(m.target, m.output) for m in got.matches] == [(0, 1), (1, 0)]
    assert (got.target_people, got.output_people, got.comparable, got.close) == (2, 3, 2, 2)


def test_one_output_person_cannot_stand_in_for_two_target_people() -> None:
    target = two("t.png", make("arms_up"), make("arms_up", tx=700.0))
    got = pose_adherence(target, result("o.png", make("arms_up")))
    assert not got.adheres and got.matched == 1 and got.close == 1
    assert [m.output for m in got.matches] == [0, None]


def test_everyone_must_hold_their_pose() -> None:
    target = two("t.png", make("arms_up"), make("seated"))
    output = two("o.png", make("arms_up"), make("lying"))
    got = pose_adherence(target, output)
    assert not got.adheres and got.close == 1 and got.distance == max(m.distance for m in got.matches)


def test_primary_only_compares_the_two_primary_subjects() -> None:
    target = two("t.png", make("arms_up"), make("seated"))
    output = two("o.png", make("arms_up"), make("lying"))
    got = pose_adherence(target, output, primary_only=True)
    assert got.adheres and got.primary_only and (got.target_people, got.output_people) == (1, 1)


# ── people who cannot be compared ───────────────────────────────────────────────────────────────────────────────


def test_an_unusable_target_person_is_listed_and_does_not_count() -> None:
    hidden = make("standing", dropout=tuple(range(17)))
    got = pose_adherence(two("t.png", make("arms_up"), hidden), result("o.png", make("arms_up")))
    assert got.adheres and got.comparable == 1
    assert got.matches[1].unusable_reason == "no_confident_torso" and got.matches[1].output is None


def test_no_comparable_target_never_adheres() -> None:
    hidden = make("standing", dropout=tuple(range(17)))
    assert not pose_adherence(result("t.png", hidden), result("o.png", make("standing"))).adheres
    nobody = pose_adherence(result("t.png", None), result("o.png", make("standing")))
    assert not nobody.adheres and nobody.matches == [] and nobody.distance is None


def test_an_output_with_nobody_or_nobody_usable_does_not_adhere() -> None:
    target = result("t.png", make("arms_up"))
    for output in (result("o.png", None), result("o.png", make("standing", dropout=tuple(range(17))))):
        got = pose_adherence(target, output)
        assert not got.adheres and got.matched == 0 and got.matches[0].output is None
        assert got.matches[0].unusable_reason is None  # the target is fine: nothing in the output could be compared


def test_bare_poses_and_arrays_are_taken_and_carry_no_estimator() -> None:
    target, output = make("wave"), make("wave", joint_noise=0.01)
    for t, o in ((bare(target), bare(output)), ([target], [output]), (bare(target), result("o.png", output))):
        got = pose_adherence(t, o)
        assert got.adheres and got.estimator is None


def test_the_result_does_not_depend_on_the_order_people_are_listed_in() -> None:
    a, b, c = make("arms_up"), make("seated"), make("wave")
    forward = pose_adherence(bare(a, b), bare(a, b, c))
    shuffled = pose_adherence(bare(a, b), bare(c, b, a))
    assert forward.adheres and shuffled.adheres
    assert [m.output for m in shuffled.matches] == [2, 1] and forward.distance == shuffled.distance


# ── POST /adherence ──────────────────────────────────────────────────────────────────────────────────────────────


def client(settings, **changes) -> TestClient:
    return TestClient(create_app(dataclasses.replace(settings, **changes), estimator=FakeEstimator()))


def dump(pose) -> dict:
    return pose.model_dump(mode="json")


def test_the_route_matches_the_library(settings) -> None:
    target, output = result("t.png", make("wave")), result("o.png", make("wave", mirror=True))
    reply = client(settings).post("/adherence", json={"target": dump(target), "output": dump(output)})
    assert reply.status_code == 200
    assert PoseAdherence.model_validate(reply.json()) == pose_adherence(target, output)
    assert reply.json()["adheres"] is False and reply.json()["matches"][0]["mirrored"] is True


def test_the_route_takes_bare_poses_a_threshold_and_primary_only(settings) -> None:
    body = {
        "target": dump(bare(make("standing"), make("seated"))),
        "output": dump(bare(make("hands_on_hips"), make("lying"))),
        "threshold": 5.0,
        "primary_only": True,
        "params": {"min_joint_score": 0.5},
    }
    got = client(settings).post("/adherence", json=body).json()
    assert got["adheres"] is True and got["threshold"] == 5.0 and got["target_people"] == 1
    assert got["params"]["min_joint_score"] == 0.5


def test_the_route_refuses_bad_bodies_without_echoing_them(settings) -> None:
    c = client(settings)
    target = dump(result("t.png", make("wave")))
    for body in ({"target": target}, {"target": target, "output": target, "threshold": -1}, {"output": "secret-text"}):
        reply = c.post("/adherence", json=body)
        assert reply.status_code == 422 and "secret-text" not in reply.text
    assert c.post("/adherence", content=b"{}", headers={"content-type": "text/plain"}).status_code == 415


def test_the_route_caps_people_on_either_side(settings) -> None:
    c = client(settings, max_people=2)
    crowd, one = dump(bare(*[make("standing")] * 3)), dump(bare(make("standing")))
    for body, side in (({"target": crowd, "output": one}, "target"), ({"target": one, "output": crowd}, "output")):
        reply = c.post("/adherence", json=body)
        assert reply.status_code == 413 and reply.json()["detail"].startswith(f"{side}: 3 people")


# ── the CLI ──────────────────────────────────────────────────────────────────────────────────────────────────────


def _sidecar(tmp_path, name: str, keypoints) -> str:
    path = sidecar_path(tmp_path / f"{name}.png")
    write_sidecar(result(f"{name}.png", keypoints), path)
    return str(path)


def test_cli_adherence_says_yes_and_exits_zero(tmp_path) -> None:
    out = tmp_path / "report" / "adherence.json"
    args = ["adherence", _sidecar(tmp_path, "t", make("wave")), _sidecar(tmp_path, "o", make("wave")), "-o", str(out)]
    reply = runner.invoke(app, args)
    assert reply.exit_code == 0, reply.output
    assert "adheres: 1 of 1" in reply.output and "target 0 -> output 0: 0.000 (close)" in reply.output
    assert PoseAdherence.model_validate_json(out.read_text()).adheres


def test_cli_adherence_names_a_mirror_image_and_exits_one(tmp_path) -> None:
    target, output = _sidecar(tmp_path, "t", make("wave")), _sidecar(tmp_path, "o", make("wave", mirror=True))
    reply = runner.invoke(app, ["adherence", target, output])
    assert reply.exit_code == 1 and "does not adhere" in reply.output and "(mirrored)" in reply.output
    as_json = runner.invoke(app, ["adherence", target, output, "--json", "--threshold", "9"])
    assert as_json.exit_code == 0 and json.loads(as_json.output)["adheres"] is True
    assert runner.invoke(app, ["adherence", target, output, "--threshold", "-1"]).exit_code == 1


def test_cli_adherence_poses_an_image_in_memory_and_writes_nothing(monkeypatch, tmp_path) -> None:
    fake = FakeEstimator()
    monkeypatch.setattr(estimator_mod, "get_estimator", lambda settings, **kw: fake)
    image = tmp_path / "out.png"
    image.write_bytes(draw_image())
    before = sorted(p.name for p in tmp_path.iterdir())
    reply = runner.invoke(app, ["adherence", str(image), str(image), "--device", "cpu"])
    assert fake.calls == 2 and "target" in reply.output  # both sides were posed
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # no sidecar was left behind


def test_cli_adherence_reports_a_source_it_cannot_read(tmp_path) -> None:
    bad = tmp_path / "bad.pose.json"
    bad.write_text("{}")
    reply = runner.invoke(app, ["adherence", str(bad), str(bad)])
    assert reply.exit_code == 1 and "cannot read a pose from" in reply.output
