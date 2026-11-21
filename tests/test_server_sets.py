"""POST /stats, /compare and /render: sets inline or as fenced sidecar folders, the caps, the errors."""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import FakeEstimator, asgi_request
from fastapi.testclient import TestClient
from PIL import Image
from posegen import PRESETS, make, result

from argus_puppet import metrics
from argus_puppet.compare import compare_pose_sets
from argus_puppet.extract import write_sidecar
from argus_puppet.metrics import pose_set_stats
from argus_puppet.models import PoseSetComparison, PoseSetStats, sidecar_path
from argus_puppet.render import render_png
from argus_puppet.server.app import RENDER_JSON_BYTES, create_app

NAMES = list(PRESETS)


def poses(prefix: str = "img", n: int = 6) -> list:
    return [result(f"{prefix}/{k}.png", make(NAMES[k % len(NAMES)])) for k in range(n)]


def body(rs) -> list[dict]:
    return [r.model_dump(mode="json") for r in rs]


@pytest.fixture
def root(tmp_path) -> Path:
    """A data root holding a folder of sidecars (and a sub-folder), plus a folder outside it."""
    data = tmp_path / "data"
    for r in poses("set", 4):
        target = data / r.image
        target.parent.mkdir(parents=True, exist_ok=True)
        write_sidecar(r, sidecar_path(target))
    sub = data / "set" / "sub"
    sub.mkdir()
    extra = result("set/sub/x.png", make("t_pose"))
    write_sidecar(extra, sidecar_path(sub / "x.png"))
    (tmp_path / "outside").mkdir()
    write_sidecar(extra, tmp_path / "outside" / "y.pose.json")
    return data


def client(settings, root: Path | None = None, **changes) -> TestClient:
    s = dataclasses.replace(settings, data_root=root, **changes)
    return TestClient(create_app(s, estimator=FakeEstimator()))


# ── /stats ───────────────────────────────────────────────────────────────────────────────────────────────────────


def test_stats_inline_matches_the_library(settings) -> None:
    rs = poses()
    reply = client(settings).post("/stats", json={"results": body(rs)})
    assert reply.status_code == 200, reply.text
    got = PoseSetStats.model_validate(reply.json())
    assert got == pose_set_stats(rs)
    assert got.params == metrics.DEFAULT_PARAMS and got.estimator is not None  # every pose shares one estimator


def test_stats_params_are_honoured(settings) -> None:
    rs = poses()
    params = {"cluster_threshold": 50, "diversity_scale": 3.0}
    got = client(settings).post("/stats", json={"results": body(rs), "params": params}).json()
    assert got["cluster_count"] == 1 and got["params"]["diversity_scale"] == 3.0
    assert got["diversity"] < pose_set_stats(rs).diversity


def test_stats_from_a_sidecar_folder_under_the_data_root(settings, root) -> None:
    c = client(settings, root)
    flat = c.post("/stats", json={"path": "set", "recursive": False}).json()
    deep = c.post("/stats", json={"path": "set"}).json()
    assert (flat["count"], deep["count"]) == (4, 5)
    assert c.post("/stats", json={"path": "."}).json()["count"] == 5  # the root itself


def test_stats_path_refusals(settings, root) -> None:
    (root / "escape").symlink_to(root.parent / "outside")
    c = client(settings, root)
    for path in ("../outside", "/etc", "set/../../outside", "escape", "a\\b", ""):
        assert c.post("/stats", json={"path": path}).status_code == 400, path
    assert c.post("/stats", json={"path": "nope"}).status_code == 404
    assert c.post("/stats", json={"path": "set/0.pose.json"}).status_code == 400  # not a folder
    reply = client(settings, None).post("/stats", json={"path": "set"})
    assert reply.status_code == 409 and "ARGUS_PUPPET_DATA_ROOT" in reply.json()["detail"]


def test_a_symlinked_sidecar_pointing_outside_is_skipped(settings, root) -> None:
    (root / "set" / "leak.pose.json").symlink_to(root.parent / "outside" / "y.pose.json")
    assert client(settings, root).post("/stats", json={"path": "set", "recursive": False}).json()["count"] == 4


def test_an_invalid_sidecar_is_422_named_relative_to_the_root(settings, root) -> None:
    (root / "set" / "bad.pose.json").write_text("{}")
    reply = client(settings, root).post("/stats", json={"path": "set"})
    assert reply.status_code == 422 and reply.json()["detail"] == "set/bad.pose.json is not a valid pose sidecar"


def test_stats_body_errors(settings) -> None:
    c = client(settings)
    assert c.post("/stats", json={}).status_code == 422  # neither results nor path
    assert c.post("/stats", json={"results": [], "path": "x"}).status_code == 422  # both
    assert c.post("/stats", content=b"{not json", headers={"content-type": "application/json"}).status_code == 422
    assert c.post("/stats", content=b"x", headers={"content-type": "text/plain"}).status_code == 415
    errors = c.post("/stats", json={"results": [{"image": "x"}]}).json()["detail"]
    assert all("input" not in e for e in errors)  # the body is never echoed back


def test_set_size_and_body_caps(settings, root) -> None:
    rs = poses(n=3)
    small = client(settings, root, max_set_results=2)
    reply = small.post("/stats", json={"results": body(rs)})
    assert reply.status_code == 413 and "ARGUS_PUPPET_MAX_SET_RESULTS" in reply.json()["detail"]
    assert small.post("/stats", json={"path": "set"}).status_code == 413
    tiny = client(settings, max_json_bytes=2048)
    assert tiny.post("/stats", json={"results": body(rs)}).status_code == 413


def test_the_set_cap_bites_before_the_body_is_validated(settings) -> None:
    """A body with more results than the cap is refused on a count of its bytes, before any result is validated
    (a 127 MB body of tiny results once cost ~1 GB and seconds of validation before the 413)."""
    rs = body(poses(n=3))
    poisoned = [*rs, {"not": "a pose"}]  # validating it would answer 422, so a 413 means nothing was validated
    small = client(settings, max_set_results=2)
    reply = small.post("/stats", json={"results": poisoned})
    assert reply.status_code == 413 and "ARGUS_PUPPET_MAX_SET_RESULTS" in reply.json()["detail"]
    escaped = json.dumps({"results": poisoned}).replace('"image_sha256"', '"image\\u005fsha256"')
    reply = small.post("/stats", content=escaped, headers={"content-type": "application/json"})
    assert reply.status_code == 413, "an escaped key must not slip past the count"
    pair = {"outputs": {"results": poisoned}, "reference": {"results": rs[:2]}}
    assert small.post("/compare", json=pair).status_code == 413  # 5 results in all: over 2 sets of 2
    fits = {"outputs": {"results": rs[:2]}, "reference": {"results": rs[:2]}}
    assert small.post("/compare", json=fits).status_code == 200  # each set counts against the cap on its own
    named = [dict(r, image=f"caf\u00e9-{k}.png") for k, r in enumerate(rs[:2])]
    ascii_body = json.dumps({"results": named}, ensure_ascii=True)  # \u escapes in values cost nothing
    assert "\\u00e9" in ascii_body
    reply = small.post("/stats", content=ascii_body, headers={"content-type": "application/json"})
    assert reply.status_code == 200, reply.text


def test_people_are_counted_before_the_body_is_validated(settings) -> None:
    """Zero-filled people cost ~1 KB of JSON and ~18 KB validated: a set body may carry no more of them than its size
    allows real ones (ARGUS_PUPPET_MAX_JSON_BYTES / PERSON_JSON_FLOOR); a render body is capped at 4 MiB."""
    r = poses(n=1)[0]
    zero = dict(r.people[0].model_dump(mode="json"), keypoints=[[0, 0, 0]] * 133, primary=False)
    crowd = dict(r.model_dump(mode="json"), people=[zero] * 40, primary=None, height=0)  # invalid: 422 if read
    c = client(settings, max_json_bytes=64 * 1024)  # room for 32 people
    reply = c.post(
        "/stats",
        content=json.dumps({"results": [crowd]}, separators=(",", ":")).encode(),
        headers={"content-type": "application/json"},
    )
    assert reply.status_code == 413 and "people" in reply.json()["detail"]
    kp = [[0, 0, 0]] * 133
    big = {"pose": {"keypoints": [kp] * 5000, "width": 9, "height": 9}}  # ~5 MB of bare people
    reply = client(settings).post("/render", json=big)
    assert reply.status_code == 413 and str(RENDER_JSON_BYTES) in reply.json()["detail"]


# ── /compare ─────────────────────────────────────────────────────────────────────────────────────────────────────


def test_compare_inline_and_by_path(settings, root) -> None:
    rs = poses("set", 4)
    c = client(settings, root)
    reply = c.post("/compare", json={"outputs": {"results": body(rs)}, "reference": {"path": "set"}})
    assert reply.status_code == 200, reply.text
    cmp = PoseSetComparison.model_validate(reply.json())
    assert cmp.output_count == 4 and cmp.close_share == 1.0 and cmp.reference_count == 5
    assert all(m.distance == 0.0 and m.nearest == m.image for m in cmp.matches)
    loose = c.post(
        "/compare", json={"outputs": {"results": body(rs[:1])}, "reference": {"results": body(rs[1:])}, "threshold": 9}
    ).json()
    assert loose["threshold"] == 9 and loose["close_share"] == 1.0
    assert loose == json.loads(compare_pose_sets(rs[:1], rs[1:], threshold=9).model_dump_json())


def test_compare_fences_both_paths(settings, root) -> None:
    c = client(settings, root)
    reply = c.post("/compare", json={"outputs": {"path": "set"}, "reference": {"path": "../outside"}})
    assert reply.status_code == 400 and "reference.path" in reply.json()["detail"]
    assert c.post("/compare", json={"outputs": {"path": "set"}}).status_code == 422
    assert (
        c.post("/compare", json={"outputs": {"path": "set"}, "reference": {"path": "set"}, "threshold": -1}).status_code
        == 422
    )


# ── /render ──────────────────────────────────────────────────────────────────────────────────────────────────────


def _png(reply) -> np.ndarray:
    assert reply.status_code == 200, reply.text
    assert reply.headers["content-type"] == "image/png"
    return np.asarray(Image.open(io.BytesIO(reply.content)))


def test_render_a_pose_result_matches_the_library(settings) -> None:
    r = poses(n=1)[0]
    opts = {"width": 256, "height": 256, "face": False, "stroke_scale": "auto"}
    reply = client(settings).post("/render", json={"pose": r.model_dump(mode="json"), "options": opts})
    assert _png(reply).shape == (256, 256, 3)
    from argus_puppet.models import RenderOptions

    assert reply.content == render_png(r, RenderOptions(**opts))


def test_render_bare_keypoints_one_or_many(settings) -> None:
    kp = make("t_pose", scale=40.0, tx=100, ty=100).tolist()
    c = client(settings)
    one = _png(c.post("/render", json={"pose": {"keypoints": kp, "width": 200, "height": 220}}))
    assert one.shape == (220, 200, 3) and one.any()
    two = _png(c.post("/render", json={"pose": {"keypoints": [kp, kp], "width": 200, "height": 220}}))
    assert np.array_equal(one, two)  # the same person twice draws the same pixels
    assert c.post("/render", json={"pose": {"keypoints": kp[:10], "width": 9, "height": 9}}).status_code == 422


def test_render_caps_pixels_and_people(settings) -> None:
    kp = make("t_pose").tolist()
    c = client(settings, max_pixels=100 * 100, max_people=2)
    big = {"pose": {"keypoints": kp, "width": 50, "height": 50}, "options": {"width": 200, "height": 200}}
    assert c.post("/render", json=big).status_code == 413
    src = {"pose": {"keypoints": kp, "width": 500, "height": 500}}  # default canvas: the source frame
    assert c.post("/render", json=src).status_code == 413
    crowd = {"pose": {"keypoints": [kp] * 3, "width": 50, "height": 50}}
    assert c.post("/render", json=crowd).status_code == 413


@pytest.mark.parametrize("path", ["/stats", "/compare", "/render"])
def test_a_stalled_json_body_is_408_and_frees_the_compute_slot(settings, path) -> None:
    app = create_app(dataclasses.replace(settings, body_timeout_s=0.2, pose_concurrency=1), estimator=FakeEstimator())
    reply = asgi_request(app, path, [b'{"results": [', 30.0], headers={"content-type": "application/json"})
    assert reply.status == 408 and b"ARGUS_PUPPET_BODY_TIMEOUT_S" in reply.body
    assert app.state.puppet.compute_admit.in_flight == 0
    assert TestClient(app).post("/stats", json={"results": body(poses())}).status_code == 200


def test_compute_runs_off_the_event_loop(settings, monkeypatch) -> None:
    seen = []
    real = metrics.pose_set_stats

    def probe(*a, **kw):
        try:
            asyncio.get_running_loop()
            seen.append("loop")
        except RuntimeError:
            seen.append("thread")
        return real(*a, **kw)

    monkeypatch.setattr(metrics, "pose_set_stats", probe)
    assert client(settings).post("/stats", json={"results": body(poses())}).status_code == 200
    assert seen == ["thread"]
