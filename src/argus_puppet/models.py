"""The wire contract: what a pose sidecar, the server and the CLI's ``--json`` carry.

This module is the API surface other packages code against (argus-proof, argus-curator, argus-darkroom, the demo
UI). Top-level payloads subclass the versioned base and carry ``puppet_version``; nested types do not.
``schema/puppet-wire.schema.json`` is rendered from here (``argus-puppet schema``) and CI fails if it drifts.

Conventions everywhere:

* **Keypoints** are COCO-WholeBody 133 (:mod:`argus_puppet.skeleton`): ``(x, y, score)`` in the image's pixel
  coordinates (after EXIF orientation), exactly 133 per person. Coordinates are always finite; a point the
  estimator could not place has score 0. Scores are clipped into ``[0, 1]``.
* **Boxes** are ``[x, y, w, h]`` in pixels, the top-left corner then the size (argus-curator's convention).
* **Images** are named by a path relative to the scanned root (with ``/`` separators), or by the name a caller
  supplied for an upload; never an absolute path.
* Unknown fields are ignored on read, so a 1.x reader accepts a later 1.y payload.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from argus_cortex.wire import make_versioned_base, render_schema
from pydantic import AfterValidator, BaseModel, Field, StringConstraints, field_validator, model_validator

from argus_puppet.skeleton import LAYOUT, NUM_KEYPOINTS, part_mean_scores

# 1.0: initial contract.
PUPPET_VERSION = "1.0"
SUPPORTED_PUPPET_MAJORS: tuple[str, ...] = ("1",)
WIRE_TITLE = "argus-puppet wire contract"
# The committed schema, relative to the repo root (``argus-puppet schema`` writes it, ``--check`` compares).
SCHEMA_PATH = Path("schema/puppet-wire.schema.json")

# A pose sidecar sits beside its image as ``<stem>.pose.json`` (``<stem>.json`` is another package's sidecar).
SIDECAR_SUFFIX = ".pose.json"

_Versioned = make_versioned_base(
    "puppet_version", PUPPET_VERSION, SUPPORTED_PUPPET_MAJORS, label="puppet_version", class_name="_Versioned"
)


def sidecar_path(image: Path) -> Path:
    """Where *image*'s pose sidecar goes: ``photo.jpg`` -> ``photo.pose.json`` in the same directory."""
    return image.with_name(image.stem + SIDECAR_SUFFIX)


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Unit = Annotated[float, Field(ge=0.0, le=1.0)]
Count = Annotated[int, Field(ge=0)]
Layout = Literal["coco_wholebody_133"]


def _check_keypoints(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    for i, (x, y, s) in enumerate(points):
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"keypoint {i}: coordinates must be finite (score an unplaced point 0 instead)")
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"keypoint {i}: score {s} is outside [0, 1]")
    return points


def _check_bbox(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not all(math.isfinite(v) for v in box):
        raise ValueError("bbox values must be finite")
    if box[2] < 0 or box[3] < 0:
        raise ValueError("bbox width and height must be >= 0")
    return box


Keypoint = tuple[float, float, float]
Keypoints = Annotated[
    list[Keypoint],
    Field(
        min_length=NUM_KEYPOINTS,
        max_length=NUM_KEYPOINTS,
        description="Exactly 133 COCO-WholeBody keypoints, each (x, y, score): pixel coordinates and a score in "
        "[0, 1]. A point the estimator could not place has score 0.",
    ),
    AfterValidator(_check_keypoints),
]
BBox = Annotated[
    tuple[float, float, float, float],
    Field(description="[x, y, w, h] in pixels: the top-left corner, then width and height."),
    AfterValidator(_check_bbox),
]


# ── provenance ───────────────────────────────────────────────────────────────────────────────────────────────────


class ModelRef(BaseModel):
    """One pinned model file, as a result records it."""

    name: str = Field(..., description="Registry name, e.g. 'dwpose-l-384x288' or 'yolox_m'.")
    sha256: Sha256 = Field(..., description="SHA-256 of the model file that ran (lower-case hex).")
    license: str = Field(..., description="SPDX licence of the weights, e.g. 'Apache-2.0'.")
    bytes: int = Field(..., ge=0, description="Size of the model file in bytes.")


class EstimatorProvenance(BaseModel):
    """What produced a set of keypoints: the detector and pose model, where they ran, and the detection settings."""

    name: str = Field(..., description="The estimator pipeline, e.g. 'topdown' (a person detector, then a pose model).")
    layout: Layout = Field(LAYOUT, description="Keypoint layout of the output.")
    detector: ModelRef = Field(..., description="The person detector.")
    pose: ModelRef = Field(..., description="The pose model.")
    provider: str = Field(..., description="ONNX Runtime execution provider used, e.g. 'CPUExecutionProvider'.")
    package_version: str = Field(..., description="argus-puppet version that ran the estimator.")
    det_threshold: Unit = Field(..., description="Minimum detector score for a person box to be kept.")
    nms_iou: Unit = Field(..., description="IoU above which overlapping person boxes are suppressed.")


# ── per image ────────────────────────────────────────────────────────────────────────────────────────────────────


class PartScores(BaseModel):
    """Mean keypoint score of each part of one person (keys match ``skeleton.PARTS``)."""

    body: Unit = Field(..., description="Mean score of the 17 body joints (0-16).")
    feet: Unit = Field(..., description="Mean score of the 6 foot points (17-22).")
    face: Unit = Field(..., description="Mean score of the 68 face landmarks (23-90).")
    left_hand: Unit = Field(..., description="Mean score of the 21 left-hand points (91-111).")
    right_hand: Unit = Field(..., description="Mean score of the 21 right-hand points (112-132).")


class PersonPose(BaseModel):
    """One detected person and their whole-body keypoints."""

    bbox: BBox
    det_score: Unit = Field(..., description="The detector's score for this person's box.")
    keypoints: Keypoints
    primary: bool = Field(False, description="Whether this is the image's primary subject (PoseResult.primary).")
    part_scores: PartScores = Field(..., description="Mean keypoint score per part.")

    @classmethod
    def from_array(
        cls, keypoints: np.ndarray, *, bbox: tuple[float, float, float, float], det_score: float, primary: bool = False
    ) -> PersonPose:
        """Build from a ``(133, 3)`` array, computing the part scores. NaN coordinates become 0 with score 0, and
        scores are clipped into [0, 1]."""
        kp = np.array(keypoints, dtype=np.float64)
        if kp.shape != (NUM_KEYPOINTS, 3):
            raise ValueError(f"expected keypoints shaped ({NUM_KEYPOINTS}, 3), got {kp.shape}")
        unplaced = ~np.isfinite(kp[:, :2]).all(axis=1)
        kp[unplaced] = 0.0
        kp[:, 2] = np.clip(np.nan_to_num(kp[:, 2], nan=0.0), 0.0, 1.0)
        return cls(
            bbox=tuple(float(v) for v in bbox),
            det_score=float(min(max(det_score, 0.0), 1.0)),
            keypoints=[(float(x), float(y), float(s)) for x, y, s in kp],
            primary=primary,
            part_scores=PartScores(**part_mean_scores(kp)),
        )

    def keypoints_array(self) -> np.ndarray:
        """The keypoints as a ``(133, 3)`` float64 array."""
        return np.asarray(self.keypoints, dtype=np.float64)


class PoseResult(_Versioned):
    """One image's poses: the sidecar ``<stem>.pose.json`` and the body of ``POST /pose``."""

    image: str = Field(..., description="Path relative to the scanned root ('/' separators), or the upload's name.")
    image_sha256: Sha256 = Field(..., description="SHA-256 of the image file's bytes.")
    width: int = Field(..., ge=1, description="Image width in pixels, after EXIF orientation.")
    height: int = Field(..., ge=1, description="Image height in pixels, after EXIF orientation.")
    layout: Layout = Field(LAYOUT, description="Keypoint layout of every person.")
    people: list[PersonPose] = Field(default_factory=list, description="Every person found, largest first.")
    primary: int | None = Field(
        None, ge=0, description="Index into people of the primary subject; null when there is nobody."
    )
    estimator: EstimatorProvenance

    @model_validator(mode="after")
    def _primary_agrees(self) -> PoseResult:
        flagged = [i for i, p in enumerate(self.people) if p.primary]
        if self.primary is None:
            if flagged:
                raise ValueError("people are flagged primary but PoseResult.primary is null")
        else:
            if self.primary >= len(self.people):
                raise ValueError(f"primary {self.primary} is out of range for {len(self.people)} people")
            if flagged != [self.primary]:
                raise ValueError(f"primary is {self.primary} but the people flagged primary are {flagged}")
        return self

    @property
    def primary_person(self) -> PersonPose | None:
        """The primary subject, or ``None``."""
        return None if self.primary is None else self.people[self.primary]


# ── labels and set metrics ───────────────────────────────────────────────────────────────────────────────────────

Stance = Literal["standing", "jumping", "crouching", "seated", "lying", "unknown"]
Arms = Literal["up", "out", "down", "mixed", "unknown"]
Facing = Literal["front", "back", "left", "right", "unknown"]


class PoseLabels(BaseModel):
    """Coarse labels derived from keypoints (geometry rules, not a classifier). 'unknown' when the joints a rule
    needs are not confident enough. Facing is the subject's, as seen by the camera: 'left' means they face the
    image's left."""

    stance: Stance = "unknown"
    arms: Arms = "unknown"
    facing: Facing = "unknown"


class Coverage(BaseModel):
    """How many usable poses carry each label. Producers list every label of each kind, zero or not."""

    stance: dict[Stance, Count] = Field(default_factory=dict, description="Count per stance label.")
    arms: dict[Arms, Count] = Field(default_factory=dict, description="Count per arms label.")
    facing: dict[Facing, Count] = Field(default_factory=dict, description="Count per facing label.")


class PoseSetItem(BaseModel):
    """One image of a set, as the set metrics saw it."""

    image: str = Field(..., description="The image, named as in its PoseResult.")
    usable: bool = Field(
        ..., description="Whether it had a primary person with enough confident joints to embed (see params)."
    )
    cluster: int | None = Field(None, ge=0, description="The PoseCluster.id it fell in; null when not usable.")
    labels: PoseLabels = Field(default_factory=PoseLabels)
    xy: tuple[float, float] | None = Field(
        None, description="2-D projection of its pose vector for a scatter plot; null when not usable."
    )


class PoseCluster(BaseModel):
    """A group of similar poses."""

    id: int = Field(..., ge=0, description="Cluster id, 0 for the largest.")
    size: int = Field(..., ge=1, description="Number of usable poses in it.")
    share: Unit = Field(..., description="size / the set's usable count.")
    medoid: str = Field(..., description="The image whose pose is most central to the cluster.")


class MetricParams(BaseModel):
    """The thresholds a set's metrics were computed with."""

    min_joint_score: Unit = Field(..., description="A joint counts as confident at or above this score.")
    min_usable_joints: int = Field(
        ..., ge=1, description="Confident body joints a primary person needs for the image to be usable."
    )
    cluster_threshold: float = Field(
        ..., ge=0.0, description="Pose distance below which two poses join the same cluster."
    )


class PoseSetStats(_Versioned):
    """Pose diversity and coverage of one image set (argus-proof's pose-diversity gate reads ``diversity``)."""

    count: int = Field(..., ge=0, description="Images in the set.")
    usable: int = Field(..., ge=0, description="Images with a usable primary pose.")
    diversity: Unit = Field(
        ..., description="How varied the usable poses are, 0 (all one pose) to 1 (highly varied); 0 under 2 usable."
    )
    mean_pairwise_distance: float = Field(
        ..., ge=0.0, description="Mean pose distance over all usable pairs; 0 under 2 usable."
    )
    cluster_count: int = Field(..., ge=0)
    largest_cluster_share: Unit = Field(..., description="Share of usable poses in the largest cluster; 0 if none.")
    clusters: list[PoseCluster] = Field(default_factory=list, description="Largest first.")
    coverage: Coverage = Field(default_factory=Coverage)
    items: list[PoseSetItem] = Field(default_factory=list)
    params: MetricParams
    estimator: EstimatorProvenance | None = Field(None, description="The estimator, when the stats extracted poses.")

    @model_validator(mode="after")
    def _counts_agree(self) -> PoseSetStats:
        if self.usable > self.count:
            raise ValueError(f"usable {self.usable} exceeds count {self.count}")
        if self.cluster_count != len(self.clusters):
            raise ValueError(f"cluster_count {self.cluster_count} != {len(self.clusters)} clusters")
        return self


class PoseMatch(BaseModel):
    """An output image's nearest reference pose."""

    image: str = Field(..., description="The output image.")
    nearest: str = Field(..., description="The reference image with the closest pose.")
    distance: float = Field(..., ge=0.0, description="Pose distance between the two.")
    memorised: bool = Field(..., description="Whether distance is below the comparison's threshold.")


class PoseSetComparison(_Versioned):
    """An output set's poses against a reference (training) set: copied poses and coverage drift."""

    output_count: int = Field(..., ge=0, description="Usable output poses compared.")
    reference_count: int = Field(..., ge=0, description="Usable reference poses compared against.")
    matches: list[PoseMatch] = Field(default_factory=list, description="One per usable output image.")
    memorised_share: Unit = Field(..., description="Share of matches flagged memorised.")
    threshold: float = Field(..., ge=0.0, description="Pose distance below which an output counts as memorised.")
    coverage_divergence: Unit = Field(
        ...,
        description="Jensen-Shannon divergence (base 2) between the output and reference coverage histograms, "
        "averaged over stance, arms and facing: 0 = same mix of poses, 1 = disjoint.",
    )
    output_coverage: Coverage = Field(default_factory=Coverage)
    reference_coverage: Coverage = Field(default_factory=Coverage)
    params: MetricParams
    estimator: EstimatorProvenance | None = None


# ── conditioning ─────────────────────────────────────────────────────────────────────────────────────────────────


class RenderOptions(BaseModel):
    """How to draw an OpenPose-convention skeleton control image."""

    body: bool = Field(True, description="Draw the 18-joint body.")
    hands: bool = Field(True, description="Draw both hands.")
    face: bool = Field(True, description="Draw the face landmarks.")
    width: int | None = Field(None, ge=1, le=8192, description="Output width; default the source image's.")
    height: int | None = Field(None, ge=1, le=8192, description="Output height; default the source image's.")
    min_score: Unit = Field(0.3, description="Points below this score are not drawn.")


class PoseLibraryEntry(BaseModel):
    """One pose in a library: its keypoints and where they came from."""

    id: str = Field(..., pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$", description="Unique within the library.")
    source_image: str = Field(..., description="The image the pose was extracted from, relative to its root.")
    source_sha256: Sha256 = Field(..., description="SHA-256 of the source image.")
    width: int = Field(..., ge=1, description="Source width in pixels; the keypoints' coordinate frame.")
    height: int = Field(..., ge=1, description="Source height in pixels.")
    keypoints: Keypoints
    labels: PoseLabels = Field(default_factory=PoseLabels)
    cluster: int | None = Field(None, ge=0, description="Cluster id within the library, if clustered.")

    def keypoints_array(self) -> np.ndarray:
        """The keypoints as a ``(133, 3)`` float64 array."""
        return np.asarray(self.keypoints, dtype=np.float64)


class PoseLibrary(_Versioned):
    """A set of poses to sample from for pose-conditioned generation."""

    layout: Layout = LAYOUT
    created_from: str = Field("", description="Where the poses came from, e.g. 'CC0 photos of adults, folder poses/'.")
    estimator: EstimatorProvenance | None = Field(None, description="The estimator that extracted the poses.")
    entries: list[PoseLibraryEntry] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def _unique_ids(cls, entries: list[PoseLibraryEntry]) -> list[PoseLibraryEntry]:
        ids = [e.id for e in entries]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate entry ids: {', '.join(dupes)}")
        return entries


# ── service ──────────────────────────────────────────────────────────────────────────────────────────────────────


class Health(_Versioned):
    """``GET /health``."""

    version: str = Field(..., description="argus-puppet package version.")
    read_only: bool = Field(..., description="Whether state-changing routes are refused.")
    models: dict[str, bool] = Field(
        default_factory=dict, description="Registry name -> whether its verified file is present."
    )
    provider: str | None = Field(None, description="ONNX Runtime provider in use, once a model is loaded.")
    device: str = Field("cpu", description="'cpu' or 'cuda'.")
    free_vram_mb: int | None = Field(None, ge=0, description="Free GPU memory, when on CUDA and measurable.")


WIRE_MODELS: tuple[type[BaseModel], ...] = (
    PoseResult,
    PoseSetStats,
    PoseSetComparison,
    PoseLibrary,
    RenderOptions,
    Health,
)


def render_wire_schema() -> str:
    """The canonical committed-schema string (``schema`` writes it, ``schema --check`` compares)."""
    return render_schema(WIRE_MODELS, title=WIRE_TITLE)
