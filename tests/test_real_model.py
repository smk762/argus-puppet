"""Opt-in: the real pipeline on CPU against real photos.

Every test is skipped unless the four pinned model files are in the model directory (``ARGUS_PUPPET_TEST_MODEL_DIR``,
else the default ``~/.cache/argus_puppet/models``; ``argus-puppet models fetch`` puts them there). Then:

* ``ARGUS_PUPPET_TEST_IMAGE`` names a photo with one clearly visible person (the keypoint test).
* ``ARGUS_PUPPET_TEST_FIXTURES`` names a directory holding the two usability fixtures (the unusable-reason test):
  ``woman_yoga.jpg``, one adult standing, whole body in frame (480x720), and ``woman_walker_1280.jpg``, a crop of
  legs and one arm with the shoulders out of frame. Any photos of those two kinds work under those names.

No photo is kept in the repo and nothing is downloaded.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from argus_puppet import registry
from argus_puppet.config import DEFAULT_MODEL_DIR, Settings

# Read at import, before conftest's autouse fixture clears ARGUS_PUPPET_* for each test.
IMAGE = os.environ.get("ARGUS_PUPPET_TEST_IMAGE", "").strip()
FIXTURES = Path(os.environ.get("ARGUS_PUPPET_TEST_FIXTURES", "").strip() or "/nonexistent").expanduser()
FULL_BODY = "woman_yoga.jpg"
NO_SHOULDERS = "woman_walker_1280.jpg"
MODEL_DIR = Path(os.environ.get("ARGUS_PUPPET_TEST_MODEL_DIR", "").strip() or DEFAULT_MODEL_DIR).expanduser()


def _image_ready() -> str | None:
    if not IMAGE or not Path(IMAGE).expanduser().is_file():
        return "set ARGUS_PUPPET_TEST_IMAGE to a photo of one person"
    return None


def _fixtures_ready() -> str | None:
    missing = [n for n in (FULL_BODY, NO_SHOULDERS) if not (FIXTURES / n).is_file()]
    if missing:
        return f"set ARGUS_PUPPET_TEST_FIXTURES to a directory holding {', '.join(missing)}"
    return None


def _ready() -> str | None:
    missing = [m.filename for m in registry.MODELS.values() if not (MODEL_DIR / m.filename).is_file()]
    if missing:
        return f"model files missing from {MODEL_DIR}: {', '.join(missing)} (argus-puppet models fetch)"
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "onnxruntime is not installed"
    return None


pytestmark = pytest.mark.skipif(_ready() is not None, reason=_ready() or "")


@pytest.mark.skipif(_image_ready() is not None, reason=_image_ready() or "")
@pytest.mark.parametrize(
    ("pose", "detector"),
    [(registry.DEFAULT_POSE, registry.DEFAULT_DETECTOR), (registry.LIGHT_POSE, registry.LIGHT_DETECTOR)],
)
def test_one_person_with_confident_body_keypoints(pose: str, detector: str) -> None:
    from argus_puppet import extract

    settings = Settings(model_dir=MODEL_DIR, pose_model=pose, detector_model=detector, device="cpu")
    result = extract.pose_image(Path(IMAGE).expanduser(), settings=settings)
    assert len(result.people) == 1, [p.det_score for p in result.people]
    subject = result.primary_person
    assert subject is not None and subject.primary and subject.det_score > 0.5
    body = subject.keypoints_array()[:17]
    assert (body[:, 2] >= 0.5).sum() >= 12, body[:, 2]
    assert subject.part_scores.body >= 0.5
    x, y, w, h = subject.bbox
    inside = (
        (body[:, 0] >= x - 0.1 * w)
        & (body[:, 0] <= x + 1.1 * w)
        & (body[:, 1] >= y - 0.1 * h)
        & (body[:, 1] <= y + 1.1 * h)
    )
    assert inside[body[:, 2] >= 0.5].all()
    assert result.estimator.provider == "CPUExecutionProvider"
    assert (result.estimator.pose.name, result.estimator.detector.name) == (pose, detector)


# ── usability on real photos ─────────────────────────────────────────────────────────────────────────────────────

# Longest side of each rescaled copy (None: as shot). The yoga photo is 480x720, so this is an upscale and two
# downscales; the sizes the investigation behind the unusable reasons swept.
SCALES = (None, 1024, 512, 384)


def _variants(name: str, scales: tuple[int | None, ...] = SCALES) -> list[tuple[str, bytes]]:
    """(name, PNG bytes) of the photo plain and mirrored, at each of *scales*."""
    from PIL import Image, ImageOps

    with Image.open(FIXTURES / name) as raw:
        im = ImageOps.exif_transpose(raw).convert("RGB")
    out = []
    for mirrored in (False, True):
        for side in scales:
            x = ImageOps.mirror(im) if mirrored else im
            if side is not None:
                r = side / max(x.size)
                x = x.resize((round(x.width * r), round(x.height * r)), Image.Resampling.BICUBIC)
            buf = io.BytesIO()
            x.save(buf, "PNG")
            out.append((f"{'m' if mirrored else 'p'}_{side or 'native'}_{name}.png", buf.getvalue()))
    return out


@pytest.mark.skipif(_fixtures_ready() is not None, reason=_fixtures_ready() or "")
def test_a_full_body_photo_is_usable_and_a_crop_without_shoulders_is_not() -> None:
    from argus_puppet import extract
    from argus_puppet.metrics import pose_set_stats

    settings = Settings(model_dir=MODEL_DIR, device="cpu")
    full = [extract.pose_image(b, name=n, settings=settings) for n, b in _variants(FULL_BODY)]
    crop = [extract.pose_image(b, name=n, settings=settings) for n, b in _variants(NO_SHOULDERS, (None,))]
    stats = pose_set_stats(full + crop)
    by = {i.image: i for i in stats.items}
    for r in full:
        assert by[r.image].usable and by[r.image].unusable_reason is None, r.image
    for r in crop:
        assert not by[r.image].usable and by[r.image].unusable_reason == "no_confident_shoulder", r.image
    assert stats.unusable_reasons == {"no_confident_shoulder": len(crop)}
