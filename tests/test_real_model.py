"""Opt-in: the real pipeline on CPU against a real photo of one person.

Skipped unless the four pinned model files are in the model directory (``ARGUS_PUPPET_TEST_MODEL_DIR``, else the
default ``~/.cache/argus_puppet/models``; ``argus-puppet models fetch`` puts them there) **and**
``ARGUS_PUPPET_TEST_IMAGE`` names a photo with one clearly visible person. No photo is kept in the repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from argus_puppet import registry
from argus_puppet.config import DEFAULT_MODEL_DIR, Settings

# Read at import, before conftest's autouse fixture clears ARGUS_PUPPET_* for each test.
IMAGE = os.environ.get("ARGUS_PUPPET_TEST_IMAGE", "").strip()
MODEL_DIR = Path(os.environ.get("ARGUS_PUPPET_TEST_MODEL_DIR", "").strip() or DEFAULT_MODEL_DIR).expanduser()


def _ready() -> str | None:
    if not IMAGE or not Path(IMAGE).expanduser().is_file():
        return "set ARGUS_PUPPET_TEST_IMAGE to a photo of one person"
    missing = [m.filename for m in registry.MODELS.values() if not (MODEL_DIR / m.filename).is_file()]
    if missing:
        return f"model files missing from {MODEL_DIR}: {', '.join(missing)} (argus-puppet models fetch)"
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "onnxruntime is not installed"
    return None


pytestmark = pytest.mark.skipif(_ready() is not None, reason=_ready() or "")


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
