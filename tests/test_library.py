from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from synthetic_poses import STANDING_BODY, figure, result

from argus_puppet.age import AgeUnavailable
from argus_puppet.compare import DEFAULT_CLOSE_THRESHOLD
from argus_puppet.library import (
    DEFAULT_DEDUPE_THRESHOLD,
    AgeGate,
    build_library,
    default_distance_matrix,
    entry_from_result,
    entry_id,
    load_library,
    render_entries,
    sample,
    save_library,
)
from argus_puppet.models import PoseLabels, PoseLibrary, RenderOptions
from argus_puppet.skeleton import flip_horizontal

ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def squat() -> np.ndarray:
    b = STANDING_BODY.copy()
    b[11:13, 1] = 130  # hips drop
    b[13:15, 1] = 125  # knees out and up
    b[13, 0], b[14, 0] = 75, 25
    b[:11, 1] += 25
    return b


def lean() -> np.ndarray:
    b = STANDING_BODY.copy()
    b[:11, 0] += np.linspace(30, 10, 11)  # upper body leans right
    return b


POSES = {
    "stand": lambda **kw: figure(**kw),
    "arms_up": lambda **kw: figure(arms_up=True, **kw),
    "squat": lambda **kw: figure(body=squat(), **kw),
    "lean": lambda **kw: figure(body=lean(), **kw),
}


def results_with_duplicates() -> list:
    """Each pose four times: shifted, scaled and jittered copies (near-duplicates), in separate images."""
    rng = np.random.default_rng(0)
    out = []
    for name, make in POSES.items():
        for k in range(4):
            kp = make(scale=1 + 0.1 * k, dx=3 * k, dy=-2 * k)
            kp[:, :2] += rng.normal(0, 0.2, (133, 2))
            out.append(result([kp], image=f"{name}/{k}.png", width=200, height=300))
    return out


# ── entries ──────────────────────────────────────────────────────────────────────────────────────────────────────


def test_entry_ids_are_stable_content_derived_and_valid() -> None:
    r = result([figure()], image="x.png")
    e1, e2 = entry_from_result(r), entry_from_result(r)
    assert e1 is not None and e2 is not None and e1.id == e2.id
    assert ID_PATTERN.match(e1.id) and e1.id.startswith(r.image_sha256[:16])
    moved = result([figure(dx=1)], image="x.png")
    assert entry_from_result(moved).id != e1.id  # type: ignore[union-attr]
    kp = figure()
    assert entry_id("ab" * 32, kp) == entry_id("ab" * 32, kp + 1e-6)  # float noise does not change it


def test_entry_records_provenance_and_uses_the_primary_person() -> None:
    a, b = figure(), figure(arms_up=True)
    r = result([a, b], image="dir/p.jpg", width=100, height=200, primary=1)
    e = entry_from_result(r, labeler=lambda kp: PoseLabels(arms="up"))
    assert e is not None
    assert (e.source_image, e.source_sha256, e.width, e.height) == ("dir/p.jpg", r.image_sha256, 100, 200)
    assert np.allclose(e.keypoints_array(), r.people[1].keypoints_array())
    assert e.labels.arms == "up"


def test_unusable_results_are_skipped() -> None:
    assert entry_from_result(result([])) is None
    assert entry_from_result(result([figure(score=0.1)])) is None
    assert entry_from_result(result([figure(score=0.1)]), min_body_score=0.05) is not None


# ── the default distance ─────────────────────────────────────────────────────────────────────────────────────────


def test_default_distance_is_the_real_metric_translation_scale_and_mirror_invariant() -> None:
    kp = figure(body=lean())
    mirrored = flip_horizontal(kp, 100)
    d = default_distance_matrix([kp, figure(body=lean(), scale=3, dx=40), mirrored, figure(arms_up=True)])
    assert d[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert d[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert d[0, 3] > 0.3
    assert np.allclose(d, d.T, equal_nan=True) and np.all(np.diag(d) == 0)


def test_default_distance_is_nan_when_a_pose_cannot_be_embedded() -> None:
    no_hips = figure()
    no_hips[11:13, 2] = 0.0
    few = figure()
    few[:17, 2] = 0.0
    few[[5, 11, 12], 2] = 0.9  # anchored but only three joints
    d = default_distance_matrix([figure(), no_hips, few])
    assert np.isnan(d[0, 1]) and np.isnan(d[1, 1]) and np.isnan(d[0, 2])
    assert default_distance_matrix([]).shape == (0, 0)


# ── building ─────────────────────────────────────────────────────────────────────────────────────────────────────


def test_build_skips_unusable_and_repeated_images() -> None:
    good = result([figure()], image="a.png")
    build = build_library(
        [good, good, result([], image="empty.png"), result([figure(score=0.05)], image="faint.png")],
        created_from="synthetic test figures",
    )
    lib = build.library
    assert [e.source_image for e in lib.entries] == ["a.png"]
    assert (build.considered, build.no_person, build.low_score, build.duplicate_ids) == (4, 1, 1, 1)
    assert lib.created_from == "synthetic test figures"
    assert lib.estimator == good.estimator


def test_build_dedupes_to_one_per_pose_by_default() -> None:
    build = build_library(results_with_duplicates())
    names = sorted(e.source_image.split("/")[0] for e in build.library.entries)
    assert names == sorted(POSES)
    assert build.near_duplicates == 12 and len(build.dropped) == 12
    assert DEFAULT_DEDUPE_THRESHOLD == DEFAULT_CLOSE_THRESHOLD == 0.04


def _posegen_result(name: str, kp: np.ndarray):
    import hashlib

    import posegen

    r = posegen.result(name, kp)
    return r.model_copy(update={"image_sha256": hashlib.sha256(name.encode()).hexdigest()})


def test_unembeddable_poses_are_rejected_and_counted() -> None:
    # Regression: 30 head-and-shoulders frames (mean body score 0.52) were admitted, never deduplicated (NaN), each
    # became its own cluster and took 5 of 20 samples.
    import posegen
    from test_metrics import NAMES, sample

    rng = np.random.default_rng(60)
    good = [
        _posegen_result(r.image, r.people[0].keypoints_array()) for r in sample(rng, NAMES, 60, angle=10, prefix="g")
    ]
    portraits = []
    for k in range(30):
        kp = posegen.make("standing", rng=rng, angle_jitter=2, joint_noise=0.01)
        kp[:17, 2] = 0.1
        kp[:9, 2] = 0.9  # head, shoulders and elbows: no hip to anchor on
        portraits.append(_posegen_result(f"p{k:03d}.png", kp))
    build = build_library(good + portraits)
    assert build.unembeddable == 30 and build.low_score == 0
    assert build.unembeddable_reasons == {"no_confident_hip": 30}  # embedding's reason, not a second guess
    assert not any(e.source_image.startswith("p") for e in build.library.entries)
    capped = build_library(good + portraits, max_entries=25)
    assert len(capped.library.entries) == 25 and capped.unembeddable == 30


def test_dedupe_needs_the_whole_limb_skeleton() -> None:
    import posegen

    full = posegen.make("standing")
    waist_up = posegen.make("standing", dropout=(13, 14, 15, 16))  # distance 0 to full, over 8 shared joints
    no_arms = posegen.make("standing", dropout=(7, 8, 9, 10))
    rs = [_posegen_result("a.png", waist_up), _posegen_result("b.png", full), _posegen_result("c.png", no_arms)]
    build = build_library(rs)
    assert build.near_duplicates == 0 and len(build.library.entries) == 3
    # a and c share no limb joint, so they were never compared; b (kept, best scored) cannot absorb both
    copies = [_posegen_result("d.png", full + [[1.0, 0.0, 0.0]]), _posegen_result("e.png", full)]
    assert build_library(copies).near_duplicates == 1


def test_dedupe_keeps_distinct_people_in_one_stance_and_drops_re_estimated_copies() -> None:
    from test_metrics import sample

    from argus_puppet.embedding import embed

    rng = np.random.default_rng(60)
    people = sample(rng, ["standing"], 60, angle=8, noise=0.03, prefix="g")
    rs = [_posegen_result(r.image, r.people[0].keypoints_array()) for r in people]
    for k, r in enumerate(people[:20]):
        kp = r.people[0].keypoints_array()
        kp[:23, :2] += rng.normal(0, 0.01 * embed(kp).scale_px, size=(23, 2))  # the same frame, re-estimated
        rs.append(_posegen_result(f"z-copy{k:02d}.png", kp))
    build = build_library(rs, labeler=None)
    survivors = {e.source_image for e in build.library.entries}
    assert sum(1 for s in survivors if s.startswith("z-copy")) <= 2
    lost = [r.image for r in people if r.image not in survivors and f"z-copy{people.index(r):02d}.png" not in survivors]
    assert len(lost) <= 2  # was 52 of 60 at the old 0.08


def test_build_does_not_depend_on_the_input_order() -> None:
    rs = results_with_duplicates()
    rng = np.random.default_rng(4)
    a = build_library(rs, max_entries=3)
    for _ in range(4):
        shuffled = [rs[i] for i in rng.permutation(len(rs))]
        b = build_library(shuffled, max_entries=3)
        key = lambda lib: sorted((e.id, e.cluster) for e in lib.entries)  # noqa: E731
        assert key(b.library) == key(a.library)
        assert [e.id for e in b.library.entries] == [r for r in (x.id for x in b.library.entries)]


def test_build_labels_and_clusters_with_the_real_rules_by_default() -> None:
    lib = build_library(results_with_duplicates(), dedupe_threshold=None).library
    by_pose = {e.source_image.split("/")[0]: e for e in lib.entries}
    assert by_pose["stand"].labels.stance == "standing" and by_pose["arms_up"].labels.arms == "up"
    assert all(e.cluster is not None for e in lib.entries)
    # four copies of each pose share one cluster; arms-up (>= 0.35 from every other pose) is a cluster of its own,
    # while this mild squat sits 0.18 from standing, under the 0.25 cut, and joins it
    clusters = {name: {e.cluster for e in lib.entries if e.source_image.startswith(name + "/")} for name in POSES}
    assert all(len(c) == 1 for c in clusters.values())
    others = set().union(*(c for name, c in clusters.items() if name != "arms_up"))
    assert not clusters["arms_up"] & others and len(others) == 2
    unlabelled = build_library(results_with_duplicates()[:1], labeler=None, clusterer=None).library.entries[0]
    assert unlabelled.labels == PoseLabels() and unlabelled.cluster is None


def test_build_dedupe_keeps_the_best_scored_copy() -> None:
    faint, strong = figure(score=0.5), figure(score=0.95, dx=1)
    build = build_library(
        [result([faint], image="faint.png"), result([strong], image="strong.png")],
    )
    assert [e.source_image for e in build.library.entries] == ["strong.png"]


def test_build_clusters_and_samples_with_injected_callables() -> None:
    seen: dict[str, int] = {}

    def distance(poses: list[np.ndarray]) -> np.ndarray:
        seen["n"] = len(poses)
        assert all(p.shape == (133, 3) for p in poses)
        return default_distance_matrix(poses)

    def clusterer(d: np.ndarray) -> list[int | None]:
        return [i // 4 for i in range(len(d))]  # inputs come pose by pose, four copies each

    rs = results_with_duplicates()
    build = build_library(rs, distance_matrix=distance, clusterer=clusterer, max_entries=4, dedupe_threshold=None)
    lib = build.library
    assert seen["n"] == 16
    assert len(lib.entries) == 4 and build.sampled_out == 12
    assert sorted({e.source_image.split("/")[0] for e in lib.entries}) == sorted(POSES)
    assert all(e.cluster is not None for e in lib.entries)
    # entries keep input order
    order = [e.source_image for e in lib.entries]
    assert order == [r.image for r in rs if r.image in order]


def test_build_rejects_a_misshapen_distance_matrix() -> None:
    with pytest.raises(ValueError):
        build_library(results_with_duplicates(), distance_matrix=lambda ps: np.zeros((2, 2)), dedupe_threshold=0.1)
    with pytest.raises(ValueError):
        build_library(results_with_duplicates(), clusterer=lambda d: [0])


def test_build_without_distance_steps_never_computes_one() -> None:
    def boom(poses: list[np.ndarray]) -> np.ndarray:
        raise AssertionError("not needed")

    rs = results_with_duplicates()
    built = build_library(rs, distance_matrix=boom, dedupe_threshold=None, clusterer=None)
    assert len(built.library.entries) == 16


# ── persistence ──────────────────────────────────────────────────────────────────────────────────────────────────


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    lib = build_library(results_with_duplicates()[:3], created_from="test").library
    assert lib.entries and lib.age_checked is False
    path = save_library(lib, tmp_path / "lib.json")
    assert load_library(path) == lib
    text = path.read_text()
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":")) + "\n"
    assert [p.name for p in tmp_path.iterdir()] == ["lib.json"]  # no temp file left behind
    save_library(lib, path)  # overwrite in place
    assert path.read_text() == text


def test_failed_save_leaves_the_old_file_and_no_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lib = PoseLibrary()
    path = save_library(lib, tmp_path / "lib.json")
    before = path.read_text()

    def fail(*a: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("argus_puppet.library.os.replace", fail)
    with pytest.raises(OSError):
        save_library(build_library(results_with_duplicates()[:1]).library, path)
    assert path.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["lib.json"]


# ── sampling and rendering ───────────────────────────────────────────────────────────────────────────────────────


def library_with_labels() -> PoseLibrary:
    def labeler(kp: np.ndarray) -> PoseLabels:
        return PoseLabels(arms="up" if kp[9, 1] < kp[0, 1] else "down", stance="standing")

    return build_library(results_with_duplicates(), labeler=labeler, dedupe_threshold=None, clusterer=None).library


def test_sample_returns_diverse_poses() -> None:
    lib = library_with_labels()
    picked = sample(lib, 4)
    assert sorted({e.source_image.split("/")[0] for e in picked}) == sorted(POSES)
    assert sample(lib, 4) == picked  # unseeded is deterministic
    assert sample(lib, 4, seed=3) == sample(lib, 4, seed=3)


def test_sample_edge_cases_and_filters() -> None:
    lib = library_with_labels()
    assert len(sample(lib, 100)) == 16
    assert sample(lib, 0) == [] and sample(PoseLibrary(), 3) == []
    up = sample(lib, 10, arms="up")
    assert up and all(e.labels.arms == "up" for e in up) and len(up) == 4
    assert sample(lib, 3, stance="lying") == []
    one = PoseLibrary(entries=lib.entries[:1])
    assert sample(one, 5) == lib.entries[:1]


def test_sample_stratifies_by_cluster() -> None:
    lib = library_with_labels()
    # Plant clusters: two copies of the squat are cluster 1, everything else (all four poses) cluster 0.
    entries = [
        e.model_copy(update={"cluster": 1 if e.source_image in ("squat/0.png", "squat/1.png") else 0})
        for e in lib.entries
    ]
    lib = PoseLibrary(entries=entries)
    picked = sample(lib, 2)
    assert sorted(e.cluster for e in picked) == [0, 1]


def test_render_entries_refits_each_skeleton() -> None:
    lib = library_with_labels()
    picked = sample(lib, 3)
    images = render_entries(picked, (256, 384), RenderOptions(width=10, height=10), margin=0.1)
    assert len(images) == 3 and all(im.size == (256, 384) for im in images)
    for im in images:
        arr = np.asarray(im)
        ys, xs = np.nonzero(arr.any(axis=-1))
        assert ys.min() >= 38 - 5 and ys.max() <= 346 + 5  # inside the 10 % margin (dots overhang by a radius)
        assert xs.min() >= 25 - 5 and xs.max() <= 231 + 5
        assert (ys.max() - ys.min() > 280) or (xs.max() - xs.min() > 190)  # fills one dimension


# ── the age fail-safe ────────────────────────────────────────────────────────────────────────────────────────────


class FakeAges:
    """Answers per image bytes: a number (the youngest estimate), None (nobody), or an exception."""

    def __init__(self, answers: dict[bytes, object]) -> None:
        self.answers = answers
        self.calls: list[bytes] = []

    def min_age(self, data: bytes) -> float | None:
        self.calls.append(data)
        answer = self.answers[data]
        if isinstance(answer, Exception):
            raise answer
        return answer  # type: ignore[return-value]


def _aged(name: str, pose: str, data: bytes):
    r = result([POSES[pose]()], image=name, width=200, height=300)
    return r.model_copy(update={"image_sha256": __import__("hashlib").sha256(data).hexdigest()})


def test_the_age_check_keeps_out_minors_and_fails_closed() -> None:
    images = {
        "adult.png": (b"adult", 34.0),
        "teen.png": (b"teen", 16.5),
        "down.png": (b"down", AgeUnavailable("the age check could not be reached")),
        "nobody.png": (b"nobody", None),
        "edge.png": (b"edge", 18.0),
    }
    poses = dict(zip(images, ["stand", "arms_up", "squat", "lean", "arms_up"], strict=True))
    results = [_aged(name, poses[name], data) for name, (data, _) in images.items()]
    results.append(_aged("moved.png", "squat", b"original"))  # the file on disk changed since posing
    store = {name: data for name, (data, _) in images.items()} | {"moved.png": b"edited"}
    ages = FakeAges({data: age for data, age in images.values()})

    def load(r) -> bytes:
        return store[r.image]

    build = build_library(results, age_gate=AgeGate(ages, load), dedupe_threshold=None)
    kept = sorted(e.source_image for e in build.library.entries)
    assert kept == ["adult.png", "edge.png"]
    assert (build.age_passed, build.under_age, build.age_unchecked, build.age_checked) == (2, 1, 3, 3)
    why = dict(build.age_excluded)
    assert why["teen.png"] == "under 18"
    assert why["down.png"].startswith("unchecked") and "reached" in why["down.png"]
    assert why["nobody.png"] == "unchecked: no face or body to estimate"
    assert "SHA-256" in why["moved.png"]
    assert b"edited" not in ages.calls  # a mismatched image is never sent
    assert build.library.age_checked is True


def test_no_library_entry_comes_from_a_frame_that_failed_the_age_check() -> None:
    """CLAUDE.md "Must never happen": with the check on, a frame that read under 18 or could not be checked never
    becomes an entry, whatever the later steps (dedupe, clustering, the cap) do."""
    rng = np.random.default_rng(5)
    results, store, answers = [], {}, {}
    for k in range(24):
        data = f"img{k}".encode()
        name = f"f{k:02d}.png"
        results.append(_aged(name, list(POSES)[k % 4], data))
        store[name] = data
        answers[data] = [40.0, 12.0, None, AgeUnavailable("x")][int(rng.integers(4))]
    bad = {r.image for r in results if not isinstance(answers[store[r.image]], float) or answers[store[r.image]] < 18}

    def load(r) -> bytes:
        if r.image == "f00.png":
            raise FileNotFoundError(r.image)
        return store[r.image]

    for kwargs in ({}, {"dedupe_threshold": None}, {"max_entries": 2}):
        build = build_library(results, age_gate=AgeGate(FakeAges(answers), load), **kwargs)
        sources = {e.source_image for e in build.library.entries}
        assert not sources & (bad | {"f00.png"})


def test_without_the_age_check_nothing_is_sent() -> None:
    ages = FakeAges({})
    build = build_library([_aged("a.png", "stand", b"a")])
    assert ages.calls == [] and build.library.age_checked is False and build.age_checked == 0
