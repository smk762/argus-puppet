from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from argus_cortex.wire import VersionError
from pydantic import ValidationError

from argus_puppet import registry
from argus_puppet.models import (
    PUPPET_VERSION,
    SIDECAR_SUFFIX,
    Coverage,
    EstimatorProvenance,
    Health,
    MetricParams,
    PersonPose,
    PoseCluster,
    PoseLibrary,
    PoseLibraryEntry,
    PoseMatch,
    PoseResult,
    PoseSetComparison,
    PoseSetItem,
    PoseSetStats,
    RenderOptions,
    render_wire_schema,
    sidecar_path,
)

SHA = "ab" * 32


def provenance() -> EstimatorProvenance:
    return EstimatorProvenance(
        name="topdown",
        detector=registry.model_ref(registry.DEFAULT_DETECTOR),
        pose=registry.model_ref(registry.DEFAULT_POSE),
        provider="CPUExecutionProvider",
        package_version="0.1.0",
        det_threshold=0.3,
        nms_iou=0.45,
    )


def keypoints(score: float = 0.9) -> np.ndarray:
    kp = np.zeros((133, 3))
    kp[:, 0] = np.linspace(10, 90, 133)
    kp[:, 1] = np.linspace(5, 195, 133)
    kp[:, 2] = score
    return kp


def person(primary: bool = False) -> PersonPose:
    return PersonPose.from_array(keypoints(), bbox=(1, 2, 80, 190), det_score=0.95, primary=primary)


def result(**kw) -> PoseResult:
    base = {
        "image": "a/b.jpg",
        "image_sha256": SHA,
        "width": 100,
        "height": 200,
        "people": [person(True), person()],
        "primary": 0,
        "estimator": provenance(),
    }
    return PoseResult(**{**base, **kw})


def test_pose_result_round_trips_through_json() -> None:
    r = result()
    assert r.puppet_version == PUPPET_VERSION and r.layout == "coco_wholebody_133"
    back = PoseResult.model_validate_json(r.model_dump_json())
    assert back == r
    assert back.primary_person == r.people[0]
    assert back.people[0].keypoints_array().shape == (133, 3)


def test_a_future_major_is_refused_and_a_future_minor_read() -> None:
    data = json.loads(result().model_dump_json())
    data["puppet_version"] = "2.0"
    with pytest.raises((VersionError, ValidationError)):
        PoseResult.model_validate(data)
    data["puppet_version"] = "1.7"
    data["a_field_from_1_7"] = True
    assert PoseResult.model_validate(data).puppet_version == "1.7"


def test_primary_must_agree_with_the_flags() -> None:
    with pytest.raises(ValidationError, match="flagged primary"):
        result(primary=None)
    with pytest.raises(ValidationError, match="out of range"):
        result(primary=5)
    with pytest.raises(ValidationError, match="flagged primary are"):
        result(primary=1)
    assert result(people=[], primary=None).primary_person is None


def test_keypoints_are_exactly_133_finite_and_scored_in_unit_range() -> None:
    good = person().model_dump()
    for bad in (good["keypoints"][:17], [*good["keypoints"], (0.0, 0.0, 0.0)]):
        with pytest.raises(ValidationError):
            PersonPose.model_validate({**good, "keypoints": bad})
    for point in ((float("nan"), 0.0, 0.5), (0.0, 0.0, 1.5), (0.0, 0.0, -0.1)):
        with pytest.raises(ValidationError):
            PersonPose.model_validate({**good, "keypoints": [point, *good["keypoints"][1:]]})
    with pytest.raises(ValidationError, match="width and height"):
        PersonPose.model_validate({**good, "bbox": (0, 0, -1, 5)})


def test_from_array_cleans_and_scores_parts() -> None:
    kp = keypoints(0.5)
    kp[0, :2] = np.nan
    kp[1, 2] = 1.7
    kp[91:112, 2] = 0.0
    p = PersonPose.from_array(kp, bbox=(0, 0, 10, 10), det_score=1.2)
    assert p.keypoints[0] == (0.0, 0.0, 0.0)
    assert p.keypoints[1][2] == 1.0
    assert p.det_score == 1.0
    assert p.part_scores.left_hand == 0.0 and p.part_scores.right_hand == 0.5
    with pytest.raises(ValueError):
        PersonPose.from_array(np.zeros((17, 3)), bbox=(0, 0, 1, 1), det_score=0.5)


def test_hashes_must_be_lowercase_hex() -> None:
    with pytest.raises(ValidationError):
        result(image_sha256="AB" * 32)


def params() -> MetricParams:
    return MetricParams(min_joint_score=0.3, min_usable_joints=8, cluster_threshold=0.5)


def test_set_stats() -> None:
    stats = PoseSetStats(
        count=3,
        usable=2,
        comparable_pairs=1,
        comparable_share=1.0,
        diversity=0.4,
        mean_pairwise_distance=0.8,
        cluster_count=1,
        largest_cluster_share=1.0,
        clusters=[PoseCluster(id=0, size=2, share=1.0, medoid="a.jpg")],
        coverage=Coverage(stance={"standing": 2, "jumping": 0}, arms={"up": 2}, facing={"front": 2}),
        items=[
            PoseSetItem(image="a.jpg", usable=True, cluster=0, xy=(0.1, 0.2)),
            PoseSetItem(image="b.jpg", usable=True, cluster=0, xy=(0.3, 0.2)),
            PoseSetItem(image="c.jpg", usable=False),
        ],
        params=params(),
    )
    assert PoseSetStats.model_validate_json(stats.model_dump_json()) == stats
    assert stats.items[2].labels.stance == "unknown"
    with pytest.raises(ValidationError):
        stats.model_validate({**stats.model_dump(), "diversity": 1.5})
    with pytest.raises(ValidationError, match="usable"):
        stats.model_validate({**stats.model_dump(), "usable": 4})
    with pytest.raises(ValidationError, match="comparable_pairs"):
        stats.model_validate({**stats.model_dump(), "comparable_pairs": 2})
    with pytest.raises(ValidationError, match="cluster_count"):
        stats.model_validate({**stats.model_dump(), "cluster_count": 2})
    with pytest.raises(ValidationError):
        Coverage(stance={"flying": 1})


def test_comparison() -> None:
    cmp = PoseSetComparison(
        output_count=2,
        matched_count=1,
        reference_count=4,
        matches=[
            PoseMatch(image="o.png", nearest="t.jpg", distance=0.05, memorised=True),
            PoseMatch(image="u.png", memorised=False),
        ],
        memorised_share=1.0,
        threshold=0.1,
        coverage_divergence=0.2,
        params=params(),
    )
    assert PoseSetComparison.model_validate_json(cmp.model_dump_json()) == cmp
    with pytest.raises(ValidationError, match="matched_count"):
        cmp.model_validate({**cmp.model_dump(), "matched_count": 2})
    with pytest.raises(ValidationError, match="matches"):
        cmp.model_validate({**cmp.model_dump(), "output_count": 3})
    with pytest.raises(ValidationError, match="both"):
        PoseMatch(image="o.png", nearest="t.jpg", memorised=False)
    with pytest.raises(ValidationError, match="memorised"):
        PoseMatch(image="o.png", memorised=True)


def test_metric_params_default_to_the_suite_defaults() -> None:
    p = MetricParams()
    assert (p.min_joint_score, p.min_usable_joints, p.cluster_threshold) == (0.3, 8, 0.25)
    assert (p.min_shared_joints, p.min_shared_limb_joints, p.diversity_scale) == (6, 2, 0.3)
    assert MetricParams.model_validate({}) == p
    with pytest.raises(ValidationError):
        MetricParams(diversity_scale=0)


def test_library_ids_are_unique() -> None:
    entry = PoseLibraryEntry(
        id="p-001",
        source_image="poses/a.jpg",
        source_sha256=SHA,
        width=100,
        height=200,
        keypoints=person().keypoints,
    )
    lib = PoseLibrary(entries=[entry], estimator=provenance(), created_from="synthetic")
    assert PoseLibrary.model_validate_json(lib.model_dump_json()) == lib
    with pytest.raises(ValidationError, match="duplicate"):
        PoseLibrary(entries=[entry, entry])
    with pytest.raises(ValidationError):
        PoseLibraryEntry.model_validate({**entry.model_dump(), "id": "../etc"})


def test_render_options_and_health_defaults() -> None:
    opts = RenderOptions()
    assert (opts.body, opts.hands, opts.face, opts.min_score) == (True, True, True, 0.3)
    assert opts.width is None and opts.height is None
    assert (opts.fit, opts.primary_only, opts.stroke_scale, opts.margin) == ("contain", False, 1.0, 0.05)
    assert RenderOptions(stroke_scale="auto").stroke_scale == "auto"
    for bad in ({"stroke_scale": 0}, {"stroke_scale": "big"}, {"fit": "zoom"}, {"margin": 0.5}):
        with pytest.raises(ValidationError):
            RenderOptions.model_validate(bad)
    h = Health(
        version="0.1.0", read_only=False, models={"yolox_m": True}, pose_model="dwpose-l", detector_model="yolox_m"
    )
    assert h.puppet_version == PUPPET_VERSION and h.free_vram_mb is None and h.gpu_coordinator == "none"


def test_sidecar_name() -> None:
    assert SIDECAR_SUFFIX == ".pose.json"
    assert sidecar_path(Path("d/photo.final.jpg")) == Path("d/photo.final.pose.json")


def test_schema_covers_the_top_level_payloads() -> None:
    schema = json.loads(render_wire_schema())
    for name in (
        "PoseResult",
        "PoseSetStats",
        "PoseSetComparison",
        "PoseLibrary",
        "RenderOptions",
        "Health",
        "ScanSummary",
        "ScanRequest",
        "StatsRequest",
        "CompareRequest",
        "RenderRequest",
    ):
        assert name in schema["$defs"]
    assert "puppet_version" in schema["$defs"]["ScanSummary"]["properties"]
    assert "puppet_version" in schema["$defs"]["PoseResult"]["properties"]
    assert "puppet_version" not in schema["$defs"]["PersonPose"]["properties"]
