"""The top-down whole-body estimator: a YOLOX person detector, then a DWPose (RTMPose SimCC) pose model.

numpy + onnxruntime + Pillow only: no OpenCV, rtmlib, mmpose or ultralytics. The pre- and post-processing
reproduces rtmlib / mmpose's conventions exactly (it matches rtmlib to 0.000 px given the same boxes):

* **Detector** (``format == "yolox-raw"``, Megvii's release ONNX): BGR, 0-255, no normalisation; letterboxed
  top-left onto a 114-grey canvas with a cv2-style (half-pixel) bilinear resize; the raw head decoded over strides
  8/16/32; score = objectness x person-class; threshold, then greedy NMS. Batch 1 only.
* **Pose** (``format == "rtmpose-simcc"``): RGB normalised by ImageNet mean/std (0-255 scale); each box scaled
  x1.25 and widened or heightened to the network's aspect; cropped with a cv2.warpAffine-convention bilinear warp
  (no half-pixel offset, zero border); every person of an image in one batch; SimCC argmax / ``split_ratio`` 2.0;
  score = min(max_x, max_y). The input size is read from the ONNX graph, not the registry.

Tests never load a model: they monkeypatch :func:`get_estimator` with a fake :class:`Estimator`.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from argus_puppet import registry, runtime
from argus_puppet.config import Settings
from argus_puppet.models import EstimatorProvenance, PersonPose
from argus_puppet.skeleton import NUM_KEYPOINTS

PIPELINE = "topdown"
POSE_MEAN = np.array([123.675, 116.28, 103.53], np.float32)  # RGB, 0-255 scale
POSE_STD = np.array([58.395, 57.12, 57.375], np.float32)
LETTERBOX_PAD = 114
YOLOX_STRIDES = (8, 16, 32)
PERSON_CLASS = 0
BBOX_PADDING = 1.25
SIMCC_SPLIT_RATIO = 2.0
# A detector box narrower or shorter than this, once clipped to the image, is not a person: a confident box over the
# letterbox padding clips to nothing, and the pose model would still hand back 133 confident points for it.
MIN_BOX_PX = 2.0


@runtime_checkable
class Estimator(Protocol):
    """Finds the people in an image and their 133 keypoints."""

    def estimate(self, image: Image.Image) -> list[PersonPose]:
        """Every person in *image* (already EXIF-oriented), in pixel coordinates, none flagged primary."""
        ...

    def provenance(self) -> EstimatorProvenance:
        """What produced the keypoints."""
        ...


# ── resampling ───────────────────────────────────────────────────────────────────────────────────────────────────


def _axis(n_out: int, s: float, t: float, n_in: int, clamp: bool) -> tuple[np.ndarray, ...]:
    """1-D bilinear taps for ``src = s * dst + t``. ``clamp`` replicates the edge (cv2.resize); otherwise a tap
    outside the image weighs 0 (cv2.warpAffine with BORDER_CONSTANT 0)."""
    src = s * np.arange(n_out, dtype=np.float64) + t
    if clamp:
        src = np.clip(src, 0, n_in - 1)
    i0 = np.floor(src).astype(np.int64)
    f = (src - i0).astype(np.float32)
    i1 = i0 + 1
    if clamp:
        i1 = np.minimum(i1, n_in - 1)
    w0 = (1 - f) * ((i0 >= 0) & (i0 < n_in))
    w1 = f * ((i1 >= 0) & (i1 < n_in))
    return np.clip(i0, 0, n_in - 1), np.clip(i1, 0, n_in - 1), w0, w1


def resample(
    img: np.ndarray, sx: float, tx: float, sy: float, ty: float, out_w: int, out_h: int, *, clamp: bool
) -> np.ndarray:
    """Axis-aligned bilinear warp of an ``HxWx3`` uint8 image: ``out[y, x] = img(sy*y + ty, sx*x + tx)``.

    The four taps are gathered from the uint8 image first and only those output-sized arrays become float32 (never
    the whole frame); the arithmetic is the same, term for term, as weighting rows and then columns."""
    h, w = img.shape[:2]
    y0, y1, wy0, wy1 = _axis(out_h, sy, ty, h, clamp)
    x0, x1, wx0, wx1 = _axis(out_w, sx, tx, w, clamp)
    ry0, ry1 = y0[:, None], y1[:, None]
    wy0, wy1 = wy0[:, None, None], wy1[:, None, None]
    left = img[ry0, x0].astype(np.float32) * wy0 + img[ry1, x0].astype(np.float32) * wy1
    right = img[ry0, x1].astype(np.float32) * wy0 + img[ry1, x1].astype(np.float32) * wy1
    out = left * wx0[None, :, None] + right * wx1[None, :, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def to_rgb(image: Image.Image) -> Image.Image:
    """*image* as 8-bit RGB. A 16-bit greyscale PNG (``I;16``) is scaled down to 8 bits first: ``convert("RGB")``
    clips it, turning anything brighter than 255 of 65535 white. ``I`` and ``F`` (not produced by the formats decoded
    here, but possible from older Pillows or callers) are read on the same 16-bit scale."""
    if image.mode.startswith("I;16") or image.mode in ("I", "F"):
        wide = np.asarray(image, dtype=np.float64)
        image = Image.fromarray(np.clip(np.rint(wide / 257.0), 0, 255).astype(np.uint8), "L")
    return image.convert("RGB")


def to_rgb_array(image: Image.Image) -> np.ndarray:
    """``HxWx3`` uint8 RGB of an image that is already EXIF-oriented (:func:`to_rgb`)."""
    return np.asarray(to_rgb(image))


# ── detector ─────────────────────────────────────────────────────────────────────────────────────────────────────


def letterbox(rgb: np.ndarray, in_h: int, in_w: int) -> tuple[np.ndarray, float]:
    """Keep-ratio resize pasted top-left on a grey canvas. Returns ``(canvas, ratio)``."""
    h, w = rgb.shape[:2]
    r = min(in_h / h, in_w / w)
    nw, nh = max(1, int(w * r)), max(1, int(h * r))  # truncation, as rtmlib
    sx, sy = w / nw, h / nh  # cv2.resize INTER_LINEAR: src = (dst + 0.5) * scale - 0.5
    resized = resample(rgb, sx, 0.5 * sx - 0.5, sy, 0.5 * sy - 0.5, nw, nh, clamp=True)
    canvas = np.full((in_h, in_w, 3), LETTERBOX_PAD, np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, r


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS over ``xyxy`` boxes, highest score first; the indices kept, in score order."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # +1 as in the YOLOX / rtmlib demo NMS
    order = scores.argsort(kind="stable")[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thr]
    return np.array(keep, np.int64)


def decode_yolox(out: np.ndarray, in_h: int, in_w: int, cls: int = PERSON_CLASS) -> tuple[np.ndarray, np.ndarray]:
    """Megvii YOLOX head ``[A, 5 + C]`` (cx, cy, w, h offsets, objectness, classes) -> ``xyxy`` boxes and scores."""
    grids, strides = [], []
    for s in YOLOX_STRIDES:
        gy, gx = np.mgrid[0 : in_h // s, 0 : in_w // s]
        grids.append(np.stack([gx, gy], -1).reshape(-1, 2))
        strides.append(np.full((grids[-1].shape[0], 1), s))
    g, st = np.concatenate(grids), np.concatenate(strides)
    if g.shape[0] != out.shape[0]:
        raise ValueError(f"detector output has {out.shape[0]} anchors, expected {g.shape[0]} for {in_w}x{in_h}")
    xy = (out[:, :2] + g) * st
    wh = np.exp(out[:, 2:4]) * st
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
    return boxes, out[:, 4] * out[:, 5 + cls]


def _input_hw(session: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape
    h, w = shape[2], shape[3]
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        return h, w
    return fallback


def detect_people(
    session: Any, rgb: np.ndarray, *, score_thr: float, nms_iou: float, input_hw: tuple[int, int]
) -> np.ndarray:
    """Person boxes ``[N, 5]`` (x1, y1, x2, y2, score) in image pixels, highest score first; a box under
    :data:`MIN_BOX_PX` on either side after clipping is dropped."""
    in_h, in_w = _input_hw(session, input_hw)
    canvas, r = letterbox(rgb, in_h, in_w)
    x = canvas[..., ::-1].astype(np.float32)  # the detector was trained on BGR, 0-255
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    out = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    boxes, scores = decode_yolox(np.asarray(out, np.float32), in_h, in_w)
    keep = scores > score_thr
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        k = nms(boxes, scores, nms_iou)
        boxes, scores = boxes[k], scores[k]
    boxes = boxes / r
    h, w = rgb.shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    real = (boxes[:, 2] - boxes[:, 0] >= MIN_BOX_PX) & (boxes[:, 3] - boxes[:, 1] >= MIN_BOX_PX)
    boxes, scores = boxes[real], scores[real]
    return np.concatenate([boxes, scores[:, None]], 1).astype(np.float32).reshape(-1, 5)


# ── pose ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def bbox_center_scale(box: np.ndarray, in_w: int, in_h: int, padding: float = BBOX_PADDING) -> tuple[np.ndarray, ...]:
    """A box's centre and its padded size, widened or heightened to the network's aspect ratio."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float64)
    bw, bh = (x2 - x1) * padding, (y2 - y1) * padding
    aspect = in_w / in_h
    if bw > bh * aspect:
        bh = bw / aspect
    else:
        bw = bh * aspect
    return c, np.array([bw, bh], np.float64)


def crop(rgb: np.ndarray, box: np.ndarray, in_w: int, in_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The network-sized patch for *box* (cv2.warpAffine convention: ``src = k * dst + (c - s / 2)``)."""
    c, s = bbox_center_scale(box, in_w, in_h)
    k = s[0] / in_w  # source pixels per network pixel (== s[1] / in_h)
    patch = resample(rgb, k, c[0] - s[0] / 2, k, c[1] - s[1] / 2, in_w, in_h, clamp=False)
    return patch, c, s


def estimate_keypoints(session: Any, rgb: np.ndarray, boxes: np.ndarray, *, input_hw: tuple[int, int]) -> np.ndarray:
    """``[N, 133, 3]`` keypoints (x, y in image pixels, raw SimCC score) for ``xyxy`` *boxes*, in one batch.

    A point whose score is <= 0 gets NaN coordinates (mmpose marks those -1 before rescaling)."""
    if len(boxes) == 0:
        return np.zeros((0, NUM_KEYPOINTS, 3), np.float64)
    in_h, in_w = _input_hw(session, input_hw)
    patches, centers, scales = zip(*(crop(rgb, b, in_w, in_h) for b in boxes), strict=True)
    x = (np.stack(patches).astype(np.float32) - POSE_MEAN) / POSE_STD
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    simcc_x, simcc_y = session.run(None, {session.get_inputs()[0].name: x})[:2]
    locs = np.stack([simcc_x.argmax(-1), simcc_y.argmax(-1)], -1).astype(np.float64) / SIMCC_SPLIT_RATIO
    scores = np.minimum(simcc_x.max(-1), simcc_y.max(-1)).astype(np.float64)
    c, s = np.stack(centers)[:, None], np.stack(scales)[:, None]
    xy = locs / np.array([in_w, in_h]) * s + c - s / 2
    xy[scores <= 0] = np.nan
    return np.concatenate([xy, scores[..., None]], -1)


def _xywh(box: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


class OnnxEstimator:
    """The real estimator: pinned ONNX models on ONNX Runtime. Sessions load (and verify) on first use."""

    def __init__(
        self,
        settings: Settings,
        *,
        device: str | None = None,
        pose_model: str | None = None,
        detector_model: str | None = None,
    ) -> None:
        self.settings = settings
        self.device = device or settings.device
        self.pose = registry.get(pose_model or settings.pose_model, "pose")
        self.detector = registry.get(detector_model or settings.detector_model, "detector")
        if self.pose.format != "rtmpose-simcc":
            raise ValueError(f"pose model {self.pose.name}: unsupported format {self.pose.format}")
        if self.detector.format != "yolox-raw":
            raise ValueError(f"detector {self.detector.name}: unsupported format {self.detector.format}")
        self._det_session: Any = None
        self._pose_session: Any = None
        self._lock = threading.Lock()

    def _sessions(self) -> tuple[Any, Any]:
        with self._lock:
            if self._det_session is None:
                self._det_session = runtime.load_session(self.settings, self.detector.name, self.device)
            if self._pose_session is None:
                self._pose_session = runtime.load_session(self.settings, self.pose.name, self.device)
            return self._det_session, self._pose_session

    def load(self) -> None:
        """Load (and verify) both models now rather than on the first image."""
        self._sessions()

    @property
    def provider(self) -> str | None:
        """The execution provider the pose model runs on, once loaded; ``None`` before."""
        if self._pose_session is None:
            return None
        return str(self._pose_session.get_providers()[0])

    def estimate(self, image: Image.Image) -> list[PersonPose]:
        det, pose = self._sessions()
        rgb = to_rgb_array(image)
        boxes = detect_people(
            det,
            rgb,
            score_thr=self.settings.det_threshold,
            nms_iou=self.settings.nms_iou,
            input_hw=self.detector.input_hw,
        )[: self.settings.max_people]
        keypoints = estimate_keypoints(pose, rgb, boxes, input_hw=self.pose.input_hw)
        return [
            PersonPose.from_array(kp, bbox=_xywh(box), det_score=float(box[4]))
            for box, kp in zip(boxes, keypoints, strict=True)
        ]

    def provenance(self) -> EstimatorProvenance:
        from argus_puppet import __version__  # noqa: PLC0415

        provider = self.provider or runtime.providers(self.device)[0]
        return EstimatorProvenance(
            name=PIPELINE,
            detector=registry.model_ref(self.detector.name),
            pose=registry.model_ref(self.pose.name),
            provider=provider,
            package_version=__version__,
            det_threshold=self.settings.det_threshold,
            nms_iou=self.settings.nms_iou,
        )


_estimators: dict[tuple[Any, ...], Estimator] = {}
_estimators_lock = threading.Lock()


def get_estimator(
    settings: Settings,
    *,
    device: str | None = None,
    pose_model: str | None = None,
    detector_model: str | None = None,
) -> Estimator:
    """The shared estimator for these settings, built once (its sessions load lazily). Tests monkeypatch this."""
    key = (
        settings.model_dir,
        device or settings.device,
        pose_model or settings.pose_model,
        detector_model or settings.detector_model,
        settings.det_threshold,
        settings.nms_iou,
        settings.max_people,
        settings.gpu_mem_limit_mb,
    )
    with _estimators_lock:
        hit = _estimators.get(key)
        if hit is None:
            hit = OnnxEstimator(settings, device=device, pose_model=pose_model, detector_model=detector_model)
            _estimators[key] = hit
        return hit
