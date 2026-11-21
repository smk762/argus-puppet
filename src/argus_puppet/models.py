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
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from argus_cortex.wire import make_versioned_base, render_schema
from pydantic import AfterValidator, BaseModel, Field, StringConstraints, field_validator, model_validator

from argus_puppet.embedding import (
    DEFAULT_MIN_JOINT_SCORE,
    DEFAULT_MIN_SHARED_JOINTS,
    DEFAULT_MIN_SHARED_LIMB_JOINTS,
    DEFAULT_MIN_USABLE_JOINTS,
    NUM_CORE_JOINTS,
)
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


# Average-linkage cut, in torso units. Two renders of one pose with estimator-level noise and a few degrees of limb
# variation sit 0.05-0.2 apart; the nearest distinct presets of the synthetic library (standing vs hands-on-hips,
# T-pose vs reach) sit 0.35 and more apart, and different stances 0.6-1.5. See docs/metrics.md.
DEFAULT_CLUSTER_THRESHOLD = 0.25
# Diversity is the mean, over comparable pairs, of g(d) = 1 - exp(-(d / scale)^2): a pair at distance 0 adds nothing,
# a pair *scale* apart adds 0.63, and a pair of different stances (d >= 0.6) adds 0.9 or more. The quadratic exponent
# keeps the slope at d = 0 flat, so estimator jitter on one repeated pose scores near zero. Calibrated on the
# synthetic generators: see docs/metrics.md.
DEFAULT_DIVERSITY_SCALE = 0.3
# A close pose match (or a library duplicate) needs the comparison to cover the whole limb skeleton: two poses that
# share only their upper bodies can match at 0.05 whatever their legs do (a waist-up standing output against a
# full-body seated frame).
DEFAULT_MIN_CLOSE_JOINTS = NUM_CORE_JOINTS


class MetricParams(BaseModel):
    """The thresholds a set's metrics were computed with. Every field has the suite default, so ``{}`` (or leaving
    ``params`` out of a request) means "the defaults"."""

    min_joint_score: Unit = Field(DEFAULT_MIN_JOINT_SCORE, description="A joint counts as confident at or above this.")
    min_usable_joints: int = Field(
        DEFAULT_MIN_USABLE_JOINTS,
        ge=1,
        le=17,
        description="Confident body joints a primary person needs for the image to be usable.",
    )
    cluster_threshold: float = Field(
        DEFAULT_CLUSTER_THRESHOLD, ge=0.0, description="Pose distance below which two poses join the same cluster."
    )
    min_shared_joints: int = Field(
        DEFAULT_MIN_SHARED_JOINTS,
        ge=1,
        le=17,
        description="Body joints that must be confident in both poses for their distance to be defined.",
    )
    min_shared_limb_joints: int = Field(
        DEFAULT_MIN_SHARED_LIMB_JOINTS,
        ge=0,
        le=8,
        description="Of those, how many must be limb joints (elbows, wrists, knees, ankles).",
    )
    diversity_scale: float = Field(
        DEFAULT_DIVERSITY_SCALE,
        gt=0.0,
        description="Distance scale of the diversity score: a pair this far apart adds 1 - 1/e (0.63) to the mean.",
    )
    min_close_joints: int = Field(
        DEFAULT_MIN_CLOSE_JOINTS,
        ge=0,
        le=NUM_CORE_JOINTS,
        description="Of the 12 non-head body joints (shoulders, elbows, wrists, hips, knees, ankles), how many two "
        "poses must share before their distance can call them a close pose match (compare) or duplicates (library "
        "dedupe). 12 means the whole limb skeleton: a waist-up pose is never a close match to a full-body frame.",
    )


class PoseSetStats(_Versioned):
    """Pose diversity and coverage of one image set (argus-proof's pose-diversity gate reads ``diversity``)."""

    count: int = Field(..., ge=0, description="Images in the set.")
    usable: int = Field(..., ge=0, description="Images with a usable primary pose.")
    comparable_pairs: int = Field(
        ...,
        ge=0,
        description="Pairs of usable poses whose distance is defined (enough shared joints): the number of pairs "
        "diversity and mean_pairwise_distance average over.",
    )
    comparable_share: Unit = Field(
        ...,
        description="comparable_pairs / (usable * (usable - 1) / 2); 0 under 2 usable. A low share means diversity "
        "describes only a subset of the set.",
    )
    diversity: Unit = Field(
        ...,
        description="How varied the usable poses are, 0 (all one pose) to 1 (highly varied): the mean over the "
        "comparable pairs of 1 - exp(-(d / diversity_scale)^2); 0 under 2 usable or with no comparable pair. Trust it "
        "only with enough usable poses and a high comparable_share (docs/metrics.md).",
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
    estimator: EstimatorProvenance | None = Field(
        None, description="The estimator that produced every pose in the set, when they all share one."
    )

    @model_validator(mode="after")
    def _counts_agree(self) -> PoseSetStats:
        if self.usable > self.count:
            raise ValueError(f"usable {self.usable} exceeds count {self.count}")
        if self.comparable_pairs > self.usable * (self.usable - 1) // 2:
            raise ValueError(f"comparable_pairs {self.comparable_pairs} exceeds the {self.usable} usable poses' pairs")
        if self.cluster_count != len(self.clusters):
            raise ValueError(f"cluster_count {self.cluster_count} != {len(self.clusters)} clusters")
        return self


class DistanceQuantiles(BaseModel):
    """Quantiles of a set of nearest-neighbour distances (torso units)."""

    n: int = Field(..., ge=1, description="How many distances the quantiles are over.")
    p01: float = Field(..., ge=0.0, description="1st percentile.")
    p10: float = Field(..., ge=0.0, description="10th percentile.")
    p50: float = Field(..., ge=0.0, description="Median.")
    p90: float = Field(..., ge=0.0, description="90th percentile.")


class PoseMatch(BaseModel):
    """An output image's nearest reference pose."""

    image: str = Field(..., description="The output image.")
    nearest: str | None = Field(
        None,
        description="The reference image with the closest pose; null when the output is unusable or shares too few "
        "joints with every reference pose (no comparable reference pose). When the output is close, the nearest "
        "reference that shares the whole limb skeleton with it (which may differ from the nearest partial match).",
    )
    distance: float | None = Field(
        None, ge=0.0, description="Pose distance between the two; null exactly when nearest is null."
    )
    shared_joints: int | None = Field(
        None,
        ge=0,
        le=NUM_CORE_JOINTS,
        description="Non-head body joints (of 12: shoulders, elbows, wrists, hips, knees, ankles) confident in both "
        "poses in the orientation compared; null exactly when nearest is null.",
    )
    close: bool = Field(
        ...,
        description="A close pose match: distance below the comparison's threshold with at least "
        "params.min_close_joints shared joints (the whole limb skeleton by default). It means 'the same pose', not "
        "'a copied frame': most same-stance poses match at the default threshold. Use it as a pointer to the training "
        "frame an image-level check should compare against; the run-level signal is close_excess.",
    )

    @model_validator(mode="after")
    def _match_agrees(self) -> PoseMatch:
        if not ((self.nearest is None) == (self.distance is None) == (self.shared_joints is None)):
            raise ValueError("nearest, distance and shared_joints are all set or all null")
        if self.nearest is None and self.close:
            raise ValueError("an output with no comparable reference pose cannot be close")
        return self


class PoseSetComparison(_Versioned):
    """An output set's poses against a reference (training) set: close pose matches against the reference set's own
    baseline, and coverage drift."""

    output_count: int = Field(..., ge=0, description="Output images compared, usable or not (one match entry each).")
    matched_count: int = Field(
        ...,
        ge=0,
        description="Outputs with a nearest reference pose: usable, and comparable with at least one usable "
        "reference pose.",
    )
    reference_count: int = Field(..., ge=0, description="Usable reference poses compared against.")
    matches: list[PoseMatch] = Field(
        default_factory=list,
        description="One per output image, sorted by image name; nearest/distance are null for an unmatched output.",
    )
    close_eligible: int = Field(
        ...,
        ge=0,
        description="Outputs that could be close: they share at least params.min_close_joints joints with at least "
        "one reference pose (in practice: the output and some reference frame both show the whole limb skeleton). "
        "close_share is over these.",
    )
    close_share: Unit = Field(
        ...,
        description="Close outputs / close_eligible (0 when none is eligible). On its own it is not evidence of "
        "copying: compare it with baseline_close_share.",
    )
    baseline_eligible: int = Field(
        ...,
        ge=0,
        description="Reference poses that could be close to another reference pose (leave-one-out, same rules).",
    )
    baseline_close_share: Unit | None = Field(
        None,
        description="The same statistic inside the reference set, leave-one-out: the share of eligible reference "
        "poses whose nearest *other* reference pose (sharing params.min_close_joints joints) is closer than the "
        "threshold. How often training frames are close to each other, i.e. what close_share looks like without "
        "any copying. Null with fewer than 2 usable reference poses or none eligible.",
    )
    close_excess: Unit | None = Field(
        None,
        description="max(0, close_share - baseline_close_share): how much more often outputs sit within the "
        "threshold of a training frame than training frames sit to each other. The run-level copying signal; null "
        "when baseline_close_share is null.",
    )
    output_nn_distance: DistanceQuantiles | None = Field(
        None,
        description="Quantiles, over the eligible outputs, of the distance to the nearest reference pose sharing "
        "params.min_close_joints joints; null when no output is eligible.",
    )
    reference_nn_distance: DistanceQuantiles | None = Field(
        None,
        description="Quantiles, over the eligible reference poses, of the distance to the nearest other reference "
        "pose sharing params.min_close_joints joints (leave-one-out); null when baseline_close_share is null. "
        "Outputs whose quantiles sit well below these are closer to the training frames than the frames are to "
        "each other.",
    )
    threshold: float = Field(..., ge=0.0, description="Pose distance below which a match can be close.")
    coverage_divergence: Unit = Field(
        ...,
        description="Jensen-Shannon divergence (base 2) between the output and reference coverage histograms, "
        "averaged over stance, arms and facing: 0 = same mix of poses, 1 = disjoint.",
    )
    output_coverage: Coverage = Field(default_factory=Coverage)
    reference_coverage: Coverage = Field(default_factory=Coverage)
    params: MetricParams
    estimator: EstimatorProvenance | None = Field(
        None, description="The estimator that produced every pose on both sides, when they all share one."
    )

    @model_validator(mode="after")
    def _counts_agree(self) -> PoseSetComparison:
        if self.matched_count > self.output_count:
            raise ValueError(f"matched_count {self.matched_count} exceeds output_count {self.output_count}")
        if len(self.matches) != self.output_count:
            raise ValueError(f"{len(self.matches)} matches for output_count {self.output_count}")
        if sum(m.nearest is not None for m in self.matches) != self.matched_count:
            raise ValueError("matched_count disagrees with the matches that have a nearest reference")
        if self.close_eligible > self.matched_count:
            raise ValueError(f"close_eligible {self.close_eligible} exceeds matched_count {self.matched_count}")
        if self.baseline_eligible > self.reference_count:
            raise ValueError(f"baseline_eligible {self.baseline_eligible} exceeds reference_count")
        if (self.baseline_close_share is None) != (self.close_excess is None):
            raise ValueError("close_excess is set exactly when baseline_close_share is")
        return self


# ── conditioning ─────────────────────────────────────────────────────────────────────────────────────────────────


Fit = Literal["contain", "cover", "stretch", "skeleton"]


class RenderOptions(BaseModel):
    """How to draw an OpenPose-convention skeleton control image (see docs/conditioning.md)."""

    body: bool = Field(True, description="Draw the 18-joint body.")
    hands: bool = Field(True, description="Draw both hands.")
    face: bool = Field(True, description="Draw the face landmarks.")
    width: int | None = Field(None, ge=1, le=8192, description="Output width; default the source image's.")
    height: int | None = Field(None, ge=1, le=8192, description="Output height; default the source image's.")
    min_score: Unit = Field(
        0.3,
        description="Visibility threshold, as the DWPose annotator applies it: a body joint is drawn when its score is "
        "above this (the synthesised neck only when both shoulders are), a hand or face point at or above it.",
    )
    fit: Fit = Field(
        "contain",
        description="How the source frame maps onto a canvas of another shape: 'contain' (letterbox, the default), "
        "'cover' (fill, cropping), 'stretch' (distorts), or 'skeleton' (fit the drawn points into the canvas less "
        "margin, ignoring the source frame).",
    )
    primary_only: bool = Field(False, description="Draw only the primary person (default: everyone).")
    stroke_scale: Annotated[float, Field(gt=0.0, le=64.0)] | Literal["auto"] = Field(
        "auto",
        description="Multiplies every line width and radius. 'auto' (the default) is min(width, height) / 512: the "
        "annotators draw at a 512 px working resolution and resize to the output, so their strokes scale with the "
        "canvas (below 1 under 512 px). 1 draws the annotator's pixel sizes at any canvas size.",
    )
    margin: float = Field(
        0.05, ge=0.0, lt=0.5, description="fit='skeleton' only: the fraction of each side kept clear on every edge."
    )


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
    age_checked: bool = Field(
        False,
        description="Whether the suite's age fail-safe ran on every source image at build time (argus-curator's "
        "POST /faces?body=true): when true, no entry comes from a frame where any face or body read under 18, or "
        "that could not be checked.",
    )
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
    pose_model: str = Field(..., description="Registry name of the pose model requests use.")
    detector_model: str = Field(..., description="Registry name of the person detector requests use.")
    provider: str | None = Field(None, description="ONNX Runtime provider in use, once a model is loaded.")
    device: str = Field("cpu", description="'cpu' or 'cuda'.")
    gpu_coordinator: str = Field(
        "none", description="GPU lease coordinator mode (ARGUS_GPU_COORDINATOR): 'none', 'lease' or 'broker'."
    )
    free_vram_mb: int | None = Field(None, ge=0, description="Free GPU memory, when on CUDA and measurable.")


# ── folder runs ──────────────────────────────────────────────────────────────────────────────────────────────────

ScanStatus = Literal["written", "skipped", "failed"]
# ImageOutcome.reason of an image left out because the per-run image cap was reached (ScanSummary.truncated).
SKIP_MAX_IMAGES = "max_images"


class ImageOutcome(BaseModel):
    """What happened to one image of a folder run."""

    image: str = Field(..., description="Path relative to the scanned root ('/' separators).")
    status: ScanStatus
    reason: str | None = Field(
        None,
        description="Why it was skipped ('exists', 'sidecar_collision', 'max_images', 'outside_root') or failed "
        "('unreadable', 'too_large').",
    )
    detail: str | None = Field(None, description="The error, for a failed image.")
    people: int = Field(0, ge=0, description="People found (0 unless written).")
    sidecar: str | None = Field(None, description="The sidecar written, relative to the scanned root.")


class ScanSummary(_Versioned):
    """A folder run's totals and per-image outcomes (``argus-puppet pose --json`` and ``POST /pose/scan``)."""

    root: str = Field(..., description="The folder scanned, as the caller named it.")
    images: int = Field(0, ge=0, description="Images found.")
    written: int = Field(0, ge=0, description="Sidecars written.")
    people: int = Field(0, ge=0, description="People found across the written images.")
    skipped: dict[str, int] = Field(default_factory=dict, description="Images skipped, by reason.")
    failed: dict[str, int] = Field(default_factory=dict, description="Images that failed, by reason.")
    truncated: bool = Field(False, description="Whether the per-run image cap stopped the run early.")
    outcomes: list[ImageOutcome] = Field(default_factory=list)

    @classmethod
    def of(cls, root: str, outcomes: Iterable[ImageOutcome]) -> ScanSummary:
        items = list(outcomes)
        skipped = Counter(o.reason or "" for o in items if o.status == "skipped")
        failed = Counter(o.reason or "" for o in items if o.status == "failed")
        return cls(
            root=root,
            images=len(items),
            written=sum(o.status == "written" for o in items),
            people=sum(o.people for o in items),
            skipped=dict(sorted(skipped.items())),
            failed=dict(sorted(failed.items())),
            truncated=skipped.get(SKIP_MAX_IMAGES, 0) > 0,
            outcomes=items,
        )


# ── requests ─────────────────────────────────────────────────────────────────────────────────────────────────────


class ScanRequest(BaseModel):
    """``POST /pose/scan``: a folder under the data root to pose into sidecars."""

    path: str = Field(..., description="Folder (or one image) relative to ARGUS_PUPPET_DATA_ROOT.")
    recursive: bool = Field(False, description="Include subfolders.")
    overwrite: bool = Field(False, description="Re-pose images that already have a sidecar.")


class PoseSetSource(BaseModel):
    """A set of poses, given inline or as a folder of sidecars under the data root: exactly one of the two."""

    results: list[PoseResult] | None = Field(None, description="The poses themselves (e.g. sidecars a client read).")
    path: str | None = Field(
        None, description="A folder relative to ARGUS_PUPPET_DATA_ROOT whose <stem>.pose.json sidecars are the set."
    )
    recursive: bool = Field(True, description="With path: include sidecars in subfolders.")

    @model_validator(mode="after")
    def _one_source(self) -> PoseSetSource:
        if (self.results is None) == (self.path is None):
            raise ValueError("give exactly one of results or path")
        return self


class StatsRequest(PoseSetSource):
    """``POST /stats``: one set's pose diversity and coverage."""

    params: MetricParams = Field(default_factory=MetricParams, description="Metric thresholds (default: suite's).")


class CompareRequest(BaseModel):
    """``POST /compare``: an output set against a reference (training) set."""

    outputs: PoseSetSource
    reference: PoseSetSource
    threshold: float | None = Field(
        None, ge=0.0, description="Close pose match distance (default: compare.DEFAULT_CLOSE_THRESHOLD, 0.08)."
    )
    params: MetricParams = Field(default_factory=MetricParams, description="Metric thresholds (default: suite's).")


class BarePose(BaseModel):
    """People's keypoints without the rest of a PoseResult: one person (133 points) or a list of people."""

    keypoints: Keypoints | list[Keypoints] = Field(
        ..., description="One person's 133 (x, y, score) points, or a list of such people."
    )
    width: int = Field(..., ge=1, le=65535, description="The keypoints' frame width in pixels.")
    height: int = Field(..., ge=1, le=65535, description="The keypoints' frame height in pixels.")

    def people(self) -> list[np.ndarray]:
        """Each person as a ``(133, 3)`` array."""
        arr = np.asarray(self.keypoints, dtype=np.float64)
        return [arr] if arr.ndim == 2 else list(arr)


class RenderRequest(BaseModel):
    """``POST /render``: one pose to an OpenPose-convention control PNG."""

    pose: PoseResult | BarePose = Field(..., description="A PoseResult (e.g. a sidecar), or bare keypoints + frame.")
    options: RenderOptions = Field(default_factory=RenderOptions)


WIRE_MODELS: tuple[type[BaseModel], ...] = (
    PoseResult,
    PoseSetStats,
    PoseSetComparison,
    PoseLibrary,
    RenderOptions,
    Health,
    ScanSummary,
    ScanRequest,
    StatsRequest,
    CompareRequest,
    RenderRequest,
)


def render_wire_schema() -> str:
    """The canonical committed-schema string (``schema`` writes it, ``schema --check`` compares)."""
    return render_schema(WIRE_MODELS, title=WIRE_TITLE)
