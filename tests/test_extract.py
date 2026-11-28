"""pose_image / pose_folder with a fake estimator: results, sidecars, skips, collisions, leases."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from argus_cortex.limits import BadInput, TooLarge
from conftest import FakeEstimator, RecordingCoordinator, draw_image
from PIL import Image

from argus_puppet import estimator as estimator_mod
from argus_puppet import extract, runtime
from argus_puppet.models import PUPPET_VERSION, PoseResult, sidecar_path


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ── one image ────────────────────────────────────────────────────────────────────────────────────────────────────


def test_pose_image_from_bytes(settings, fake) -> None:
    data = draw_image(64, 48)
    r = extract.pose_image(data, settings=settings, estimator=fake)
    assert isinstance(r, PoseResult)
    assert (r.image, r.width, r.height) == ("upload", 64, 48)
    assert r.image_sha256 == hashlib.sha256(data).hexdigest()
    assert [p.bbox[2] * p.bbox[3] for p in r.people] == [1600, 600]  # largest first
    assert r.primary == 0 and r.people[0].primary and not r.people[1].primary
    assert r.estimator.name == "fake"


def test_pose_image_from_a_path_names_it_by_file(settings, fake, tmp_path) -> None:
    p = _write(tmp_path / "pic.jpg", draw_image(fmt="JPEG"))
    assert extract.pose_image(p, settings=settings, estimator=fake).image == "pic.jpg"
    assert extract.pose_image(str(p), name="other.jpg", settings=settings, estimator=fake).image == "other.jpg"


def test_a_face_hint_moves_the_primary(settings, fake) -> None:
    r = extract.pose_image(draw_image(), settings=settings, estimator=fake, face_hint=(12, 6, 6, 6))
    assert r.primary == 1 and r.people[1].bbox == (10, 5, 20, 30)


def test_nobody_found(settings) -> None:
    r = extract.pose_image(draw_image(), settings=settings, estimator=FakeEstimator(people=[]))
    assert r.people == [] and r.primary is None


def test_exif_orientation_is_applied(settings, fake) -> None:
    img = Image.new("RGB", (60, 40), (10, 20, 30))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 CW on display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    r = extract.pose_image(buf.getvalue(), settings=settings, estimator=fake)
    assert (r.width, r.height) == (40, 60)
    assert fake.sizes == [(40, 60)]


@pytest.mark.parametrize(("value", "expected"), [(0, 0), (1000, 4), (40000, 156), (65535, 255)])
def test_16_bit_greyscale_is_scaled_not_clipped(settings, value, expected) -> None:
    buf = io.BytesIO()
    Image.new("I;16", (8, 6), value).save(buf, format="PNG")  # a 16-bit greyscale PNG
    rgb = np.asarray(extract.decode(buf.getvalue(), settings))
    assert rgb.shape == (6, 8, 3) and (rgb == expected).all(), (value, rgb.max())


@pytest.mark.parametrize("mode", ["I;16", "I;16B", "I", "F"])
def test_every_wide_mode_is_read_on_the_16_bit_scale(mode) -> None:
    img = Image.fromarray(np.array([[0, 1000, 40000, 65535]], np.uint16)).convert("I").convert(mode)
    assert np.asarray(extract.to_rgb(img))[0, :, 0].tolist() == [0, 4, 156, 255]


def test_bad_and_oversized_images(settings, fake, tmp_path) -> None:
    with pytest.raises(BadInput):
        extract.pose_image(b"not an image", settings=settings, estimator=fake)
    import dataclasses

    small = dataclasses.replace(settings, max_pixels=64 * 64)
    with pytest.raises(TooLarge):
        extract.pose_image(draw_image(100, 100), settings=small, estimator=fake)
    assert fake.calls == 0


def test_pose_image_uses_the_module_getter(settings, fake, monkeypatch) -> None:
    monkeypatch.setattr(estimator_mod, "get_estimator", lambda s, **kw: fake)
    extract.pose_image(draw_image(), settings=settings)
    assert fake.calls == 1


def test_pose_image_leases_the_gpu_on_cuda(settings, fake, cuda) -> None:
    rec = RecordingCoordinator()
    extract.pose_image(draw_image(), settings=settings, estimator=fake, coordinator=rec, priority="background")
    assert len(rec.leases) == 1 and rec.leases[0]["priority"] == "background" and rec.leases[0]["kind"] == "pose"


def test_pose_image_takes_no_lease_on_cpu(settings, fake) -> None:
    rec = RecordingCoordinator()
    extract.pose_image(draw_image(), settings=settings, estimator=fake, coordinator=rec)
    assert rec.leases == []


# ── a folder ─────────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def folder(tmp_path) -> Path:
    root = tmp_path / "set"
    _write(root / "b.png", draw_image())
    _write(root / "a.jpg", draw_image(fmt="JPEG"))
    _write(root / "notes.txt", b"not an image")
    _write(root / ".hidden.png", draw_image())
    _write(root / "sub" / "c.webp", draw_image(fmt="WEBP"))
    return root


def test_pose_folder_writes_one_sidecar_per_image(settings, fake, folder) -> None:
    outcomes = list(extract.pose_folder(folder, settings=settings, estimator=fake))
    assert [(o.image, o.status, o.people) for o in outcomes] == [("a.jpg", "written", 2), ("b.png", "written", 2)]
    assert outcomes[0].sidecar == "a.pose.json"
    side = folder / "a.pose.json"
    text = side.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert list(payload) == sorted(payload)  # sorted keys
    r = PoseResult.model_validate(payload)
    assert r.image == "a.jpg" and r.image_sha256 == hashlib.sha256((folder / "a.jpg").read_bytes()).hexdigest()
    assert not list(folder.glob("*.tmp")) and not list(folder.glob(".*.tmp"))
    assert not (folder / "sub" / "c.pose.json").exists()


def test_recursive_names_are_relative_to_the_root(settings, fake, folder) -> None:
    outcomes = list(extract.pose_folder(folder, recursive=True, settings=settings, estimator=fake))
    assert [o.image for o in outcomes] == ["a.jpg", "b.png", "sub/c.webp"]
    assert PoseResult.model_validate_json((folder / "sub" / "c.pose.json").read_text()).image == "sub/c.webp"
    assert outcomes[2].sidecar == "sub/c.pose.json"


def test_existing_sidecars_are_kept_unless_overwrite(settings, fake, folder) -> None:
    (folder / "a.pose.json").write_text("{}", encoding="utf-8")
    outcomes = list(extract.pose_folder(folder, settings=settings, estimator=fake))
    assert [(o.image, o.status, o.reason) for o in outcomes] == [
        ("a.jpg", "skipped", "exists"),
        ("b.png", "written", None),
    ]
    assert (folder / "a.pose.json").read_text() == "{}"
    list(extract.pose_folder(folder, overwrite=True, settings=settings, estimator=fake))
    assert PoseResult.model_validate_json((folder / "a.pose.json").read_text()).image == "a.jpg"


def test_images_sharing_a_stem_never_overwrite_each_other(settings, fake, tmp_path) -> None:
    root = tmp_path / "dupes"
    _write(root / "photo.jpg", draw_image(fmt="JPEG"))
    _write(root / "photo.png", draw_image())
    outcomes = list(extract.pose_folder(root, settings=settings, estimator=fake, overwrite=True))
    assert [(o.image, o.status, o.reason) for o in outcomes] == [
        ("photo.jpg", "written", None),
        ("photo.png", "skipped", "sidecar_collision"),
    ]
    assert PoseResult.model_validate_json((root / "photo.pose.json").read_text()).image == "photo.jpg"
    assert fake.calls == 1


def test_an_unreadable_image_is_reported_not_fatal(settings, fake, tmp_path) -> None:
    root = tmp_path / "mixed"
    _write(root / "a.png", b"\x89PNG broken")
    _write(root / "b.png", draw_image())
    outcomes = list(extract.pose_folder(root, settings=settings, estimator=fake))
    assert [(o.image, o.status, o.reason) for o in outcomes] == [
        ("a.png", "failed", "unreadable"),
        ("b.png", "written", None),
    ]
    assert outcomes[0].detail
    assert not (root / "a.pose.json").exists()


@pytest.fixture
def read_only_folder(tmp_path):
    root = tmp_path / "ro"
    _write(root / "a.png", draw_image())
    _write(root / "b.png", draw_image())
    root.chmod(0o555)
    if os.access(root, os.W_OK):  # root ignores the mode bits
        root.chmod(0o755)
        pytest.skip("running as a user who can write to a 0555 folder")
    yield root
    root.chmod(0o755)


def test_a_sidecar_that_cannot_be_written_is_a_failed_outcome_not_an_abort(settings, fake, read_only_folder) -> None:
    outcomes = list(extract.pose_folder(read_only_folder, settings=settings, estimator=fake))
    assert [(o.image, o.status, o.reason) for o in outcomes] == [
        ("a.png", "failed", "unwritable"),
        ("b.png", "failed", "unwritable"),
    ]
    assert all(o.detail for o in outcomes)
    assert fake.calls == 0  # a folder that cannot take a sidecar is not posed into


def test_one_unwritable_sidecar_does_not_stop_the_rest(settings, fake, tmp_path) -> None:
    root = tmp_path / "clash"
    _write(root / "a.png", draw_image())
    _write(root / "b.png", draw_image())
    (root / "a.pose.json").mkdir()  # a directory where a's sidecar would go
    summary = extract.scan(root, settings=settings, estimator=fake, overwrite=True)
    assert [(o.image, o.status, o.reason) for o in summary.outcomes] == [
        ("a.png", "failed", "unwritable"),
        ("b.png", "written", None),
    ]
    assert summary.failed == {"unwritable": 1} and summary.written == 1
    assert sidecar_path(root / "b.png").exists()
    assert not [p for p in root.iterdir() if p.name.endswith(".tmp")]  # no temp file left behind


def test_a_model_that_cannot_run_stops_the_run(settings, folder) -> None:
    broken = FakeEstimator(error=runtime.ModelUnavailable("Model x not found."))
    with pytest.raises(runtime.ModelUnavailable):
        list(extract.pose_folder(folder, settings=settings, estimator=broken))


def test_a_single_file_root(settings, fake, folder) -> None:
    outcomes = list(extract.pose_folder(folder / "b.png", settings=settings, estimator=fake))
    assert [(o.image, o.status, o.sidecar) for o in outcomes] == [("b.png", "written", "b.pose.json")]
    assert sidecar_path(folder / "b.png").exists()


def test_a_missing_root(settings, fake, tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        list(extract.pose_folder(tmp_path / "nope", settings=settings, estimator=fake))


def test_one_lease_is_held_across_a_folder(settings, fake, folder, cuda) -> None:
    rec = RecordingCoordinator()
    outcomes = list(extract.pose_folder(folder, recursive=True, settings=settings, estimator=fake, coordinator=rec))
    assert len(outcomes) == 3 and fake.calls == 3
    assert len(rec.leases) == 1 and rec.released == 1
    assert rec.leases[0]["priority"] == "batch" and rec.leases[0]["kind"] == "pose"
    assert rec.handles[0].progress == 1.0


def test_the_folder_lease_yields_between_images(settings, fake, folder, cuda) -> None:
    rec = RecordingCoordinator(yield_after=1)
    list(extract.pose_folder(folder, settings=settings, estimator=fake, coordinator=rec))
    assert len(rec.leases) == 2 and rec.released == 2


def test_no_lease_when_there_is_nothing_to_do(settings, fake, folder, cuda) -> None:
    for name in ("a", "b"):
        (folder / f"{name}.pose.json").write_text("{}")
    rec = RecordingCoordinator()
    assert {o.reason for o in extract.pose_folder(folder, settings=settings, estimator=fake, coordinator=rec)} == {
        "exists"
    }
    assert rec.leases == []


def test_max_images_caps_the_estimator_runs(settings, fake, folder) -> None:
    summary = extract.scan(folder, recursive=True, settings=settings, estimator=fake, max_images=2)
    assert summary.written == 2 and summary.skipped == {"max_images": 1} and summary.truncated
    assert fake.calls == 2


def test_a_symlink_out_of_the_fence_is_skipped(settings, fake, tmp_path) -> None:
    outside = _write(tmp_path / "outside" / "secret.png", draw_image())
    root = tmp_path / "fenced"
    _write(root / "ok.png", draw_image())
    os.symlink(outside, root / "link.png")
    summary = extract.scan(root, settings=settings, estimator=fake, fence=root)
    assert {o.image: o.reason for o in summary.outcomes} == {"link.png": "outside_root", "ok.png": None}


def test_a_fenced_out_image_claims_no_sidecar_from_its_in_fence_sibling(settings, fake, tmp_path) -> None:
    """``a.jpg`` (a symlink out of the fence) sorts before ``a.png``: it is never posed, so ``a.png`` gets the sidecar."""
    outside = _write(tmp_path / "outside" / "secret.png", draw_image())
    root = tmp_path / "fenced"
    _write(root / "a.png", draw_image())
    os.symlink(outside, root / "a.jpg")
    summary = extract.scan(root, settings=settings, estimator=fake, fence=root)
    assert {o.image: o.reason for o in summary.outcomes} == {"a.jpg": "outside_root", "a.png": None}
    assert summary.written == 1 and (root / "a.pose.json").is_file()


def test_scan_summarises(settings, fake, folder) -> None:
    (folder / "a.pose.json").write_text("{}")
    _write(folder / "z.png", b"junk")
    s = extract.scan(folder, settings=settings, estimator=fake, label="set")
    assert (s.root, s.images, s.written, s.people) == ("set", 3, 1, 2)
    assert s.skipped == {"exists": 1} and s.failed == {"unreadable": 1} and not s.truncated


# ── labels (wire 1.1) ────────────────────────────────────────────────────────────────────────────────────────────


def _labelled_fake() -> FakeEstimator:
    from posegen import make

    from argus_puppet.models import PersonPose

    standing = PersonPose.from_array(make("standing", tx=40, ty=30, scale=8), bbox=(30, 5, 20, 50), det_score=0.9)
    arms_up = PersonPose.from_array(make("arms_up", tx=15, ty=30, scale=5), bbox=(8, 10, 14, 30), det_score=0.8)
    return FakeEstimator(people=[standing, arms_up])


def test_every_person_is_labelled_from_their_keypoints(settings) -> None:
    from argus_puppet.labels import pose_labels

    r = extract.pose_image(draw_image(64, 64), settings=settings, estimator=_labelled_fake())
    assert len(r.people) == 2
    for p in r.people:
        assert p.labels is not None and p.labels == pose_labels(p.keypoints_array())
    assert r.people[0].labels.stance == "standing" and r.people[1].labels.arms == "up"


def test_labels_are_recomputed_even_when_the_estimator_supplies_some(settings) -> None:
    from argus_puppet.labels import pose_labels
    from argus_puppet.models import PoseLabels

    fake = _labelled_fake()
    fake.people = [p.model_copy(update={"labels": PoseLabels(stance="lying", arms="out")}) for p in fake.people]
    r = extract.pose_image(draw_image(64, 64), settings=settings, estimator=fake)
    assert [p.labels for p in r.people] == [pose_labels(p.keypoints_array()) for p in r.people]


def test_sidecars_carry_labels_and_a_1_0_sidecar_is_kept_unless_overwrite(settings, tmp_path) -> None:
    root = tmp_path / "imgs"
    _write(root / "a.png", draw_image(64, 64))
    _write(root / "b.png", draw_image(64, 64))
    old = extract.pose_image(root / "a.png", settings=settings, estimator=_labelled_fake())
    payload = old.model_dump(mode="json")
    payload["puppet_version"] = "1.0"
    for p in payload["people"]:
        del p["labels"]
    old_text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    (root / "a.pose.json").write_text(old_text, encoding="utf-8")

    summary = extract.scan(root, settings=settings, estimator=_labelled_fake())
    assert summary.skipped == {"exists": 1} and summary.written == 1
    assert (root / "a.pose.json").read_text(encoding="utf-8") == old_text  # not rewritten just to add labels
    kept = PoseResult.model_validate_json(old_text)
    assert kept.puppet_version == "1.0" and all(p.labels is None for p in kept.people)
    fresh = json.loads((root / "b.pose.json").read_text(encoding="utf-8"))
    assert all(set(p["labels"]) == {"stance", "arms", "facing"} for p in fresh["people"])

    extract.scan(root, settings=settings, estimator=_labelled_fake(), overwrite=True)
    redone = PoseResult.model_validate_json((root / "a.pose.json").read_text(encoding="utf-8"))
    assert redone.puppet_version == PUPPET_VERSION and all(p.labels is not None for p in redone.people)
