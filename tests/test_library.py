from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from synthetic_poses import STANDING_BODY, figure, result

from argus_puppet.library import (
    FALLBACK_DEDUPE_THRESHOLD,
    build_library,
    entry_from_result,
    entry_id,
    fallback_distance_matrix,
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


# ── the stand-in distance ────────────────────────────────────────────────────────────────────────────────────────


def test_fallback_distance_is_translation_scale_and_mirror_invariant() -> None:
    kp = figure(body=lean())
    mirrored = flip_horizontal(kp, 100)
    d = fallback_distance_matrix([kp, figure(body=lean(), scale=3, dx=40), mirrored, figure(arms_up=True)])
    assert d[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert d[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert d[0, 3] > 0.3
    assert np.allclose(d, d.T, equal_nan=True) and np.all(np.diag(d) == 0)


def test_fallback_distance_is_nan_when_a_pose_cannot_be_normalised() -> None:
    no_hips = figure()
    no_hips[11:13, 2] = 0.0
    few = figure()
    few[:17, 2] = 0.0
    few[[5, 11, 12], 2] = 0.9  # normalisable but only three joints
    d = fallback_distance_matrix([figure(), no_hips, few])
    assert np.isnan(d[0, 1]) and np.isnan(d[1, 1]) and np.isnan(d[0, 2])
    assert fallback_distance_matrix([]).shape == (0, 0)


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


def test_build_dedupes_to_one_per_pose() -> None:
    build = build_library(results_with_duplicates(), dedupe_threshold=FALLBACK_DEDUPE_THRESHOLD)
    names = sorted(e.source_image.split("/")[0] for e in build.library.entries)
    assert names == sorted(POSES)
    assert build.near_duplicates == 12 and len(build.dropped) == 12


def test_build_dedupe_keeps_the_best_scored_copy() -> None:
    faint, strong = figure(score=0.5), figure(score=0.95, dx=1)
    build = build_library(
        [result([faint], image="faint.png"), result([strong], image="strong.png")],
        dedupe_threshold=FALLBACK_DEDUPE_THRESHOLD,
    )
    assert [e.source_image for e in build.library.entries] == ["strong.png"]


def test_build_clusters_and_samples_with_injected_callables() -> None:
    seen: dict[str, int] = {}

    def distance(poses: list[np.ndarray]) -> np.ndarray:
        seen["n"] = len(poses)
        assert all(p.shape == (133, 3) for p in poses)
        return fallback_distance_matrix(poses)

    def clusterer(d: np.ndarray) -> list[int | None]:
        return [i // 4 for i in range(len(d))]  # inputs come pose by pose, four copies each

    rs = results_with_duplicates()
    build = build_library(rs, distance_matrix=distance, clusterer=clusterer, max_entries=4)
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

    assert len(build_library(results_with_duplicates(), distance_matrix=boom).library.entries) == 16


# ── persistence ──────────────────────────────────────────────────────────────────────────────────────────────────


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    lib = build_library(results_with_duplicates()[:3], created_from="test").library
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

    return build_library(results_with_duplicates(), labeler=labeler).library


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
