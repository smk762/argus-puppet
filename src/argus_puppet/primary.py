"""Which person is an image's primary subject: the same rule argus-curator uses for its primary face.

* No hint: the person maximising ``bbox_w * bbox_h * (0.5 + 0.5 * det_score)``; the first index wins a tie
  (curator's ``faces._assign_primary``).
* A primary-face hint (``[x, y, w, h]``, e.g. curator's primary face): the person whose body best matches that
  face, scored as curator matches bodies to faces (``bodies.match_score``: the mean of their IoU and the share of
  the face inside the body). A score at or under :data:`MIN_FACE_MATCH` is no match; with no match the area x score
  rule decides.

Exactly one person is primary whenever there is anyone.
"""

from __future__ import annotations

from collections.abc import Sequence

from argus_puppet.models import PersonPose

BoxXYWH = Sequence[float]
# curator's FaceConfig.body_face_match default: a pair scoring at or under this is no match.
MIN_FACE_MATCH = 0.0001


def _xyxy(box: BoxXYWH) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box)
    return (x, y, x + w, y + h)


def _area(b: tuple[float, float, float, float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def match_score(body: BoxXYWH, face: BoxXYWH) -> float:
    """Mean of the IoU of *body* and *face* and the share of *face* inside *body* (both ``xywh``); 0 if apart."""
    b, f = _xyxy(body), _xyxy(face)
    w = min(b[2], f[2]) - max(b[0], f[0])
    h = min(b[3], f[3]) - max(b[1], f[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    face_area = _area(f)
    union = _area(b) + face_area - inter
    if union <= 0 or face_area <= 0:
        return 0.0
    return (inter / union + inter / face_area) / 2


def prominence(box: BoxXYWH, det_score: float) -> float:
    """curator's primary score: box area weighted by detector confidence."""
    return float(box[2]) * float(box[3]) * (0.5 + 0.5 * float(det_score))


def primary_index(
    boxes: Sequence[BoxXYWH], det_scores: Sequence[float], *, face_hint: BoxXYWH | None = None
) -> int | None:
    """Index of the primary person among ``xywh`` *boxes*, or ``None`` when there are none."""
    if not boxes:
        return None
    if face_hint is not None:
        scores = [match_score(b, face_hint) for b in boxes]
        best = max(range(len(boxes)), key=lambda i: scores[i])  # max() keeps the first of equals
        if scores[best] > MIN_FACE_MATCH:
            return best
    return max(range(len(boxes)), key=lambda i: prominence(boxes[i], det_scores[i]))


def assign_primary(
    people: list[PersonPose], *, face_hint: BoxXYWH | None = None
) -> tuple[list[PersonPose], int | None]:
    """*people* with exactly the primary one flagged (copies; the input is not changed), and its index."""
    idx = primary_index([p.bbox for p in people], [p.det_score for p in people], face_hint=face_hint)
    return [p.model_copy(update={"primary": i == idx}) for i, p in enumerate(people)], idx
