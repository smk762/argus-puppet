"""The estimator's numpy pipeline, against fake ONNX sessions (no model files)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from argus_puppet import estimator as est
from argus_puppet import registry, runtime
from argus_puppet.config import Settings
from argus_puppet.skeleton import NUM_KEYPOINTS


class FakeSession:
    """An ONNX session stand-in: a fixed input shape and a function from the input to the outputs."""

    def __init__(self, name: str, shape: list, fn, provider: str = "CPUExecutionProvider") -> None:
        self._input = SimpleNamespace(name=name, shape=shape)
        self.fn = fn
        self.fed: list[np.ndarray] = []
        self._provider = provider

    def get_inputs(self):
        return [self._input]

    def get_providers(self):
        return [self._provider]

    def run(self, _outputs, feed):
        x = next(iter(feed.values()))
        self.fed.append(x)
        return self.fn(x)


# ── resampling and letterbox ─────────────────────────────────────────────────────────────────────────────────────


def test_resample_identity_and_zero_border() -> None:
    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    assert np.array_equal(est.resample(img, 1, 0, 1, 0, 5, 4, clamp=True), img)
    shifted = est.resample(img, 1, -2, 1, 0, 5, 4, clamp=False)  # src x = dst x - 2: the left two columns are off
    assert (shifted[:, :2] == 0).all() and np.array_equal(shifted[:, 2:], img[:, :3])


def _resample_before(img, sx, tx, sy, ty, out_w, out_h, *, clamp):
    """resample as it was: the whole image to float32, then rows and columns gathered (the reference)."""
    h, w = img.shape[:2]
    y0, y1, wy0, wy1 = est._axis(out_h, sy, ty, h, clamp)
    x0, x1, wx0, wx1 = est._axis(out_w, sx, tx, w, clamp)
    f = img.astype(np.float32)
    rows = f[y0] * wy0[:, None, None] + f[y1] * wy1[:, None, None]
    out = rows[:, x0] * wx0[None, :, None] + rows[:, x1] * wx1[None, :, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    ("shape", "geometry", "clamp"),
    [
        ((480, 640), (640 / 427, 0.5 * 640 / 427 - 0.5, 480 / 320, 0.5 * 480 / 320 - 0.5, 427, 320), True),  # letterbox
        ((300, 200), (200 / 400, 0.5 * 200 / 400 - 0.5, 300 / 600, 0.5 * 300 / 600 - 0.5, 400, 600), True),  # upscale
        ((480, 640), (0.83, -40.2, 0.83, 13.7, 192, 256), False),  # a person crop running off the left edge
        ((120, 90), (1.7, 60.0, 1.7, -30.0, 192, 256), False),  # mostly outside the image
    ],
)
def test_resample_is_unchanged_by_gathering_before_converting(shape, geometry, clamp) -> None:
    img = np.random.default_rng(7).integers(0, 256, (*shape, 3), dtype=np.uint8)
    assert np.array_equal(est.resample(img, *geometry, clamp=clamp), _resample_before(img, *geometry, clamp=clamp))


def test_resample_does_not_convert_the_whole_image() -> None:
    """A 192x256 crop of a 12 MP frame works on the crop's pixels: no float copy of the frame (144 MB here)."""
    import tracemalloc

    img = np.zeros((3000, 4000, 3), np.uint8)
    tracemalloc.start()
    try:
        est.resample(img, 1.3, 900.0, 1.3, 700.0, 192, 256, clamp=False)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024, f"resample allocated {peak / 2**20:.0f} MB for a 192x256 crop"


def test_letterbox_pastes_top_left_on_grey() -> None:
    rgb = np.full((100, 200, 3), 10, np.uint8)
    canvas, r = est.letterbox(rgb, 64, 64)
    assert r == pytest.approx(0.32)
    assert canvas.shape == (64, 64, 3)
    assert (canvas[:32, :64] == 10).all()  # 200x100 -> 64x32, top-left
    assert (canvas[32:] == est.LETTERBOX_PAD).all()


# ── detector ─────────────────────────────────────────────────────────────────────────────────────────────────────


def test_nms_keeps_the_best_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.6, 0.9, 0.5], np.float32)
    assert est.nms(boxes, scores, 0.45).tolist() == [1, 2]
    assert est.nms(boxes, scores, 0.99).tolist() == [1, 0, 2]


def _yolox_head(in_hw: tuple[int, int], cells: dict[int, tuple[float, float]]) -> np.ndarray:
    """A raw head with every anchor scoring 0 except *cells* (anchor index -> (objectness, person score))."""
    n = sum((in_hw[0] // s) * (in_hw[1] // s) for s in est.YOLOX_STRIDES)
    out = np.zeros((n, 85), np.float32)
    out[:, 2:4] = np.log(4.0)  # w = h = 4 strides
    for i, (obj, cls) in cells.items():
        out[i, 4], out[i, 5] = obj, cls
    return out


def test_decode_yolox_places_boxes_on_the_grid() -> None:
    out = _yolox_head((64, 64), {9: (1.0, 0.8)})  # stride 8, an 8x8 grid: anchor 9 is cell (x=1, y=1)
    boxes, scores = est.decode_yolox(out, 64, 64)
    assert scores[9] == pytest.approx(0.8) and scores.sum() == pytest.approx(0.8)
    assert boxes[9].tolist() == pytest.approx([8 - 16, 8 - 16, 8 + 16, 8 + 16])
    with pytest.raises(ValueError, match="anchors"):
        est.decode_yolox(out[:-1], 64, 64)


def test_detect_people_feeds_bgr_and_maps_boxes_back() -> None:
    head = _yolox_head((64, 64), {9: (1.0, 0.9), 10: (1.0, 0.2)})  # the second is under the threshold
    session = FakeSession("images", [1, 3, 64, 64], lambda x: [head[None]])
    rgb = np.zeros((32, 128, 3), np.uint8)
    rgb[..., 0] = 255  # pure red
    boxes = est.detect_people(session, rgb, score_thr=0.3, nms_iou=0.45, input_hw=(0, 0))
    fed = session.fed[0]
    assert fed.shape == (1, 3, 64, 64) and fed.dtype == np.float32
    assert fed[0, 2, 0, 0] == 255 and fed[0, 0, 0, 0] == 0  # BGR: red is the last channel, 0-255
    assert fed[0, 0, 63, 63] == est.LETTERBOX_PAD
    assert boxes.shape == (1, 5)
    r = 0.5  # 128 wide -> 64
    assert boxes[0, :4].tolist() == pytest.approx([0, 0, min((8 + 16) / r, 128), min((8 + 16) / r, 32)])
    assert boxes[0, 4] == pytest.approx(0.9)


def test_a_box_that_clips_to_nothing_is_not_a_person() -> None:
    """The letterbox pads below a wide image: a confident box out there clips to zero height in the image."""
    head = _yolox_head((64, 64), {9: (1.0, 0.9), 57: (1.0, 0.95)})  # 57: cell (x=1, y=7), in the grey padding
    session = FakeSession("images", [1, 3, 64, 64], lambda x: [head[None]])
    boxes = est.detect_people(session, np.zeros((32, 128, 3), np.uint8), score_thr=0.3, nms_iou=0.45, input_hw=(0, 0))
    assert boxes.shape == (1, 5) and boxes[0, 4] == pytest.approx(0.9)
    assert ((boxes[:, 2] - boxes[:, 0] >= est.MIN_BOX_PX) & (boxes[:, 3] - boxes[:, 1] >= est.MIN_BOX_PX)).all()


# ── pose ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_center_scale_pads_and_fixes_the_aspect() -> None:
    c, s = est.bbox_center_scale(np.array([0, 0, 100, 100]), 192, 256)
    assert c.tolist() == [50, 50]
    assert s.tolist() == pytest.approx([125, 125 / 0.75])  # x1.25, then heightened to 192:256
    _, s = est.bbox_center_scale(np.array([0, 0, 30, 100]), 192, 256)
    assert s.tolist() == pytest.approx([125 * 0.75, 125])  # a tall box is widened instead


def _simcc_at(points: np.ndarray, in_w: int, in_h: int, peak: float = 0.9):
    """A pose session whose SimCC peaks put every keypoint of person i at network pixel ``points[i]``."""

    def fn(x: np.ndarray):
        b = x.shape[0]
        sx = np.zeros((b, NUM_KEYPOINTS, in_w * 2), np.float32)
        sy = np.zeros((b, NUM_KEYPOINTS, in_h * 2), np.float32)
        for i in range(b):
            sx[i, :, int(points[i][0] * 2)] = peak
            sy[i, :, int(points[i][1] * 2)] = peak
        return [sx, sy]

    return fn


def test_estimate_keypoints_maps_simcc_back_to_the_image() -> None:
    in_w, in_h = 48, 64
    box = np.array([10, 20, 58, 84, 0.9])  # 48x64 box, centre (34, 52): already the network's aspect
    centre = np.array([[in_w / 2, in_h / 2]])
    session = FakeSession("input", ["batch", 3, in_h, in_w], _simcc_at(centre, in_w, in_h))
    rgb = np.zeros((100, 100, 3), np.uint8)
    kp = est.estimate_keypoints(session, rgb, box[None], input_hw=(in_h, in_w))
    assert kp.shape == (1, NUM_KEYPOINTS, 3)
    assert kp[0, :, :2] == pytest.approx(np.tile([34, 52], (NUM_KEYPOINTS, 1)))
    assert kp[0, :, 2] == pytest.approx(0.9)
    x = session.fed[0]
    assert x.shape == (1, 3, in_h, in_w)
    assert x[0, 0, 0, 0] == pytest.approx((0 - est.POSE_MEAN[0]) / est.POSE_STD[0])  # RGB normalised


def test_estimate_keypoints_batches_every_person_and_nans_unscored_points() -> None:
    def fn(x):
        b = x.shape[0]
        return [np.zeros((b, NUM_KEYPOINTS, 96), np.float32), np.zeros((b, NUM_KEYPOINTS, 128), np.float32)]

    session = FakeSession("input", ["batch", 3, 64, 48], fn)
    boxes = np.array([[0, 0, 10, 10, 0.9], [5, 5, 20, 30, 0.8], [1, 1, 2, 2, 0.5]], np.float32)
    kp = est.estimate_keypoints(session, np.zeros((40, 40, 3), np.uint8), boxes, input_hw=(64, 48))
    assert len(session.fed) == 1 and session.fed[0].shape[0] == 3  # one batch
    assert np.isnan(kp[..., :2]).all()  # score 0 -> unplaced
    assert est.estimate_keypoints(
        session, np.zeros((4, 4, 3), np.uint8), np.zeros((0, 5)), input_hw=(64, 48)
    ).shape == (
        0,
        NUM_KEYPOINTS,
        3,
    )


# ── OnnxEstimator ────────────────────────────────────────────────────────────────────────────────────────────────


def _fake_sessions(monkeypatch, n_people: int, in_hw=(64, 64), pose_hw=(64, 48)):
    cells = {i: (1.0, 0.9 - 0.01 * i) for i in range(0, 8 * n_people, 8)}  # spread out: NMS keeps them all
    det = FakeSession("images", [1, 3, *in_hw], lambda x: [_yolox_head(in_hw, cells)[None]])
    pose = FakeSession("input", ["batch", 3, *pose_hw], _simcc_at(np.full((64, 2), 10.0), pose_hw[1], pose_hw[0]))
    loaded = []

    def load(settings, name, device=None):
        loaded.append(name)
        return det if registry.get(name).role == "detector" else pose

    monkeypatch.setattr(runtime, "load_session", load)
    return det, pose, loaded


def test_onnx_estimator_end_to_end_with_fake_sessions(monkeypatch, tmp_path) -> None:
    _, pose, loaded = _fake_sessions(monkeypatch, 3)
    s = Settings(model_dir=tmp_path, device="cpu", max_people=2)
    e = est.OnnxEstimator(s)
    assert e.provider is None
    people = e.estimate(Image.new("RGB", (64, 64)))
    assert sorted(loaded) == sorted([s.detector_model, s.pose_model])
    assert len(people) == 2  # capped, highest score first
    assert people[0].det_score >= people[1].det_score
    assert pose.fed[0].shape[0] == 2
    assert all(len(p.keypoints) == NUM_KEYPOINTS and not p.primary for p in people)
    prov = e.provenance()
    assert prov.provider == "CPUExecutionProvider" and prov.name == est.PIPELINE
    assert prov.pose.name == s.pose_model and prov.detector.sha256 == registry.get(s.detector_model).sha256
    assert (prov.det_threshold, prov.nms_iou) == (s.det_threshold, s.nms_iou)


def test_onnx_estimator_with_nobody(monkeypatch, tmp_path) -> None:
    _fake_sessions(monkeypatch, 0)
    assert est.OnnxEstimator(Settings(model_dir=tmp_path, device="cpu")).estimate(Image.new("RGB", (30, 20))) == []


def test_get_estimator_is_shared_per_configuration(tmp_path) -> None:
    s = Settings(model_dir=tmp_path, device="cpu")
    a = est.get_estimator(s)
    assert est.get_estimator(s) is a
    assert est.get_estimator(s, pose_model=registry.LIGHT_POSE) is not a
    assert isinstance(a, est.Estimator)


def test_onnx_estimator_rejects_unknown_models(tmp_path) -> None:
    with pytest.raises(KeyError):
        est.OnnxEstimator(Settings(model_dir=tmp_path), pose_model="yolox_m")
