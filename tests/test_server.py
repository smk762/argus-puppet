"""The HTTP service with a fake estimator and a recording GPU coordinator."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
import time
from pathlib import Path

import pytest
from conftest import FakeEstimator, RecordingCoordinator, asgi_request, draw_image, fake_ort, megabytes
from fastapi.testclient import TestClient

from argus_puppet import gpu, registry, runtime
from argus_puppet.config import Settings
from argus_puppet.models import PUPPET_VERSION, Health, PoseResult, ScanSummary
from argus_puppet.server.app import SCAN_JSON_BYTES, create_app, parse_face_hint


@pytest.fixture
def client(settings, fake) -> TestClient:
    return TestClient(create_app(settings, estimator=fake, coordinator=RecordingCoordinator()))


def _post(client: TestClient, data: bytes, **kw):
    return client.post("/pose", content=data, headers={"content-type": "image/png", **kw.pop("headers", {})}, **kw)


def test_health(client) -> None:
    reply = client.get("/health")
    assert reply.status_code == 200
    h = Health.model_validate(reply.json())
    assert h.read_only is False and h.device == "cpu" and h.free_vram_mb is None
    assert set(h.models) == set(registry.names()) and not any(h.models.values())
    assert h.provider == "FakeExecutionProvider"


def test_pose_raw_bytes(client, fake) -> None:
    reply = _post(client, draw_image(70, 50))
    assert reply.status_code == 200, reply.text
    r = PoseResult.model_validate(reply.json())
    assert (r.image, r.width, r.height, len(r.people), r.primary) == ("upload", 70, 50, 2, 0)
    assert fake.calls == 1
    assert r.puppet_version == PUPPET_VERSION
    assert all(set(p["labels"]) == {"stance", "arms", "facing"} for p in reply.json()["people"])


def test_pose_octet_stream_and_no_content_type(client) -> None:
    assert (
        client.post("/pose", content=draw_image(), headers={"content-type": "application/octet-stream"}).status_code
        == 200
    )


def test_pose_multipart_takes_the_filename(client) -> None:
    reply = client.post("/pose", files={"file": ("../../holiday.png", draw_image(), "image/png")})
    assert reply.status_code == 200, reply.text
    assert reply.json()["image"] == "holiday.png"


def test_pose_name_is_sanitised(client) -> None:
    assert _post(client, draw_image(), params={"name": "dir/../shot 1.png"}).json()["image"] == "shot 1.png"
    assert _post(client, draw_image(), params={"name": ".hidden"}).status_code == 400


def test_pose_face_hint(client) -> None:
    reply = _post(client, draw_image(), params={"face": "12,6,6,6"})
    assert reply.json()["primary"] == 1
    assert _post(client, draw_image(), params={"face": "1,2,3"}).status_code == 400
    assert _post(client, draw_image(), params={"face": "1,2,0,4"}).status_code == 400


def test_parse_face_hint() -> None:
    assert parse_face_hint(None) is None and parse_face_hint(" ") is None
    assert parse_face_hint("1, 2.5, 3, 4") == (1.0, 2.5, 3.0, 4.0)


def test_pose_errors_map_to_status_codes(settings, fake) -> None:
    small = dataclasses.replace(settings, max_upload_bytes=2048)
    client = TestClient(create_app(small, estimator=fake))
    big = b"\x89PNG" + b"0" * 4096
    assert _post(client, big).status_code == 413
    assert client.post("/pose", files={"f": ("a.png", big, "image/png")}).status_code == 413
    assert _post(client, b"not an image at all").status_code == 415
    assert client.post("/pose", content=b"x", headers={"content-type": "text/plain"}).status_code == 415
    assert client.post("/pose", content=b"", headers={"content-type": "image/png"}).status_code == 400
    assert client.post("/pose", data={"field": "value"}).status_code in (400, 415)
    tiny_pixels = dataclasses.replace(settings, max_pixels=64 * 64)
    assert _post(TestClient(create_app(tiny_pixels, estimator=fake)), draw_image(100, 100)).status_code == 413
    assert fake.calls == 0


MB = 1024 * 1024
FORM_HEAD = b'--B\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\nContent-Type: image/png\r\n\r\n'
FORM_TAIL = b"\r\n--B--\r\n"
MULTIPART = {"content-type": "multipart/form-data; boundary=B"}


def test_a_chunked_multipart_past_the_cap_is_refused_before_it_is_read_to_the_end(settings, fake) -> None:
    """No Content-Length: the stream is counted against the cap plus the multipart overhead, not spooled whole."""
    app = create_app(dataclasses.replace(settings, max_upload_bytes=MB), estimator=fake)
    reply = asgi_request(app, "/pose", megabytes(32, FORM_HEAD, FORM_TAIL), headers=MULTIPART)
    assert reply.status == 413 and str(MB).encode() in reply.body, reply.body
    assert reply.consumed <= 2 * MB + len(FORM_HEAD), f"the server read {reply.consumed // MB} MB past a 1 MB cap"
    assert fake.calls == 0
    ok = asgi_request(app, "/pose", [FORM_HEAD, draw_image(), FORM_TAIL], headers=MULTIPART)
    assert ok.status == 200, ok.body  # chunked and under the cap: posed, named by the part's filename
    assert PoseResult.model_validate_json(ok.body).image == "a.png"


@pytest.mark.parametrize("content_type", ["image/png", "multipart/form-data; boundary=B"])
def test_a_stalled_upload_is_408_and_frees_its_slot(settings, fake, content_type) -> None:
    """A body that stops arriving is cut off at ARGUS_PUPPET_BODY_TIMEOUT_S, not left holding an admission slot."""
    app = create_app(dataclasses.replace(settings, body_timeout_s=0.2, pose_concurrency=1), estimator=fake)
    started = time.monotonic()
    reply = asgi_request(app, "/pose", [b"0123456789", 30.0], headers={"content-type": content_type})
    assert reply.status == 408 and b"ARGUS_PUPPET_BODY_TIMEOUT_S" in reply.body
    assert time.monotonic() - started < 5
    assert app.state.puppet.admit.in_flight == 0
    assert _post(TestClient(app), draw_image()).status_code == 200  # the one slot is free again


def test_a_missing_model_is_503_with_the_hint(settings) -> None:
    broken = FakeEstimator(error=runtime.ModelUnavailable("Model yolox_m not found."))
    reply = _post(TestClient(create_app(settings, estimator=broken)), draw_image())
    assert reply.status_code == 503 and "models fetch" in reply.json()["detail"]


def test_the_real_estimator_without_models_is_503(settings) -> None:
    reply = _post(TestClient(create_app(settings)), draw_image())  # no model files in the tmp model dir
    assert reply.status_code == 503 and "models fetch" in reply.json()["detail"]


def test_priority_header_reaches_the_lease(settings, fake, cuda) -> None:
    rec = RecordingCoordinator()
    client = TestClient(create_app(settings, estimator=fake, coordinator=rec))
    assert _post(client, draw_image()).status_code == 200
    assert _post(client, draw_image(), headers={"X-Argus-Priority": "realtime"}).status_code == 200
    assert _post(client, draw_image(), headers={"X-Argus-Priority": "batch"}).status_code == 200
    assert [lease["priority"] for lease in rec.leases] == ["interactive", "interactive", "batch"]
    assert _post(client, draw_image(), headers={"X-Argus-Priority": "now!"}).status_code == 400


def test_a_refused_lease_is_503_retry_after(settings, fake, cuda) -> None:
    from argus_cortex.gpu import GpuLeaseTimeout

    client = TestClient(
        create_app(settings, estimator=fake, coordinator=RecordingCoordinator(error=GpuLeaseTimeout("x")))
    )
    reply = _post(client, draw_image())
    assert reply.status_code == 503 and reply.headers["retry-after"]


def test_an_invalid_gpu_coordinator_is_503_as_health_reports_it(settings, fake, cuda, monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_GPU_COORDINATOR", "bogus")
    monkeypatch.setattr(gpu, "_coordinator", None)
    client = TestClient(create_app(settings, estimator=fake), raise_server_exceptions=False)
    assert client.get("/health").json()["gpu_coordinator"] == "invalid"
    reply = _post(client, draw_image())
    assert reply.status_code == 503, reply.text
    assert "ARGUS_GPU_COORDINATOR" in reply.json()["detail"] and "retry-after" not in reply.headers
    assert fake.calls == 0


def test_calls_past_the_concurrency_cap_are_refused(settings) -> None:
    started, release = threading.Event(), threading.Event()

    class Slow(FakeEstimator):
        def estimate(self, image):
            started.set()
            release.wait(5)
            return super().estimate(image)

    app = create_app(dataclasses.replace(settings, pose_concurrency=1), estimator=Slow())
    results: list[int] = []
    with TestClient(app) as client:
        t = threading.Thread(target=lambda: results.append(_post(client, draw_image()).status_code))
        t.start()
        assert started.wait(5)
        busy = _post(client, draw_image())
        release.set()
        t.join(5)
    assert busy.status_code == 503 and busy.headers["retry-after"]
    assert results == [200]


def test_inference_runs_off_the_event_loop(settings) -> None:
    seen = []

    class Probe(FakeEstimator):
        def estimate(self, image):
            try:
                asyncio.get_running_loop()
                seen.append("loop")
            except RuntimeError:
                seen.append("thread")
            return super().estimate(image)

    assert _post(TestClient(create_app(settings, estimator=Probe())), draw_image()).status_code == 200
    assert seen == ["thread"]


# ── POST /pose/scan ──────────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def data_root(tmp_path) -> Path:
    root = tmp_path / "data"
    (root / "set" / "sub").mkdir(parents=True)
    (root / "set" / "a.png").write_bytes(draw_image())
    (root / "set" / "sub" / "b.png").write_bytes(draw_image())
    return root


def _scan_client(settings: Settings, root: Path | None, fake, **changes) -> TestClient:
    return TestClient(create_app(dataclasses.replace(settings, data_root=root, **changes), estimator=fake))


def test_scan_writes_sidecars_under_the_data_root(settings, fake, data_root) -> None:
    client = _scan_client(settings, data_root, fake)
    reply = client.post("/pose/scan", json={"path": "set", "recursive": True})
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert (body["root"], body["images"], body["written"], body["people"]) == ("set", 2, 2, 4)
    assert (data_root / "set" / "sub" / "b.pose.json").exists()
    again = client.post("/pose/scan", json={"path": "set", "recursive": True}).json()
    assert again["skipped"] == {"exists": 2}
    assert client.post("/pose/scan", json={"path": "set", "overwrite": True}).json()["written"] == 1


def test_scan_caps_images_per_request(settings, fake, data_root) -> None:
    body = (
        _scan_client(settings, data_root, fake, max_scan_images=1)
        .post("/pose/scan", json={"path": "set", "recursive": True})
        .json()
    )
    assert body["written"] == 1 and body["truncated"] and body["skipped"] == {"max_images": 1}


def test_scan_without_a_data_root_is_refused(settings, fake) -> None:
    reply = _scan_client(settings, None, fake).post("/pose/scan", json={"path": "set"})
    assert reply.status_code == 409 and "ARGUS_PUPPET_DATA_ROOT" in reply.json()["detail"]


@pytest.mark.parametrize("path", ["../outside", "/etc", "set/../../outside", "a\\b", ""])
def test_scan_refuses_paths_outside_the_root(settings, fake, data_root, path) -> None:
    (data_root.parent / "outside").mkdir(exist_ok=True)
    reply = _scan_client(settings, data_root, fake).post("/pose/scan", json={"path": path})
    assert reply.status_code == 400
    assert fake.calls == 0


def test_scan_of_a_missing_folder_is_404(settings, fake, data_root) -> None:
    assert _scan_client(settings, data_root, fake).post("/pose/scan", json={"path": "nope"}).status_code == 404


def test_the_scan_body_is_capped_checked_and_never_echoed(settings, fake, data_root) -> None:
    """/pose/scan reads its body like the other JSON routes: a small cap (counted as it streams), JSON only, and
    422 without the input."""
    app = create_app(dataclasses.replace(settings, data_root=data_root), estimator=fake)
    json_type = {"content-type": "application/json"}
    reply = asgi_request(app, "/pose/scan", megabytes(32, b'{"path": "set", "pad": "', b'"}'), headers=json_type)
    assert reply.status == 413, reply.body
    assert reply.consumed <= MB + 64, f"the server read {reply.consumed // MB} MB of a scan request"
    client = TestClient(app)
    padded = b'{"path": "set", "pad": "' + b"a" * (SCAN_JSON_BYTES + 1) + b'"}'
    assert client.post("/pose/scan", content=padded, headers=json_type).status_code == 413
    assert client.post("/pose/scan", content=b"path=set", headers={"content-type": "text/plain"}).status_code == 415
    reply = client.post("/pose/scan", json={"path": ["SECRET-INPUT"]})
    assert reply.status_code == 422 and "SECRET-INPUT" not in reply.text
    assert client.post("/pose/scan", content=b"[" * 8000 + b"]" * 8000, headers=json_type).status_code == 422
    assert fake.calls == 0


def test_a_scan_without_a_data_root_is_refused_before_its_body_is_read(settings, fake) -> None:
    app = create_app(settings, estimator=fake)  # no data root
    reply = asgi_request(app, "/pose/scan", [b'{"path": "set"}'], headers={"content-type": "application/json"})
    assert reply.status == 409 and reply.consumed == 0


def test_a_scan_that_can_write_nothing_is_409_with_the_summary(settings, fake, data_root) -> None:
    client = _scan_client(settings, data_root, fake)
    folder = data_root / "set"
    folder.chmod(0o555)
    try:
        if os.access(folder, os.W_OK):
            pytest.skip("running as a user who can write to a 0555 folder")
        reply = client.post("/pose/scan", json={"path": "set"})
    finally:
        folder.chmod(0o755)
    assert reply.status_code == 409, reply.text
    detail = reply.json()["detail"]
    assert "could not be written" in detail["message"]
    assert ScanSummary.model_validate(detail["summary"]).failed == {"unwritable": 1}
    (folder / "sub" / "b.pose.json").mkdir()  # one unwritable sidecar among writable ones: 200, reported
    reply = client.post("/pose/scan", json={"path": "set", "recursive": True, "overwrite": True})
    assert reply.status_code == 200 and reply.json()["failed"] == {"unwritable": 1} and reply.json()["written"] == 1


def test_scan_with_a_missing_model_is_503(settings, data_root) -> None:
    broken = FakeEstimator(error=runtime.ModelUnavailable("Model yolox_m not found."))
    reply = _scan_client(settings, data_root, broken).post("/pose/scan", json={"path": "set"})
    assert reply.status_code == 503


def test_the_module_level_app_is_built_from_the_environment(monkeypatch, tmp_path) -> None:
    import argus_puppet.server.app as mod

    monkeypatch.setenv("ARGUS_PUPPET_MODEL_DIR", str(tmp_path))
    monkeypatch.delitem(mod.__dict__, "app", raising=False)
    app = mod.app
    assert app.state.puppet.settings.model_dir == tmp_path
    monkeypatch.delitem(mod.__dict__, "app")


# ── device truth and broker refusals ────────────────────────────────────────────────────────────────────────────


def _no_gpu(monkeypatch) -> None:
    """A CUDA build in a container without --gpus: ORT has the provider compiled in, the driver is not there."""
    fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"))
    monkeypatch.setattr(runtime, "_probe_cuda_driver", lambda: "the CUDA driver (libcuda.so.1) cannot be loaded")


def test_health_never_claims_cuda_for_a_cuda_build_without_a_gpu(settings, fake, monkeypatch) -> None:
    _no_gpu(monkeypatch)
    rec = RecordingCoordinator()
    client = TestClient(create_app(dataclasses.replace(settings, device="auto"), estimator=fake, coordinator=rec))
    h = Health.model_validate(client.get("/health").json())
    assert h.device == "cpu" and "libcuda" in h.cuda_unavailable and h.free_vram_mb is None
    assert _post(client, draw_image()).status_code == 200  # auto: posed on the CPU
    assert rec.leases == []  # and the broker was not leased for it
    assert Health.model_validate(client.get("/health").json()).device == "cpu"


def test_health_reports_a_session_that_fell_back_to_the_cpu(settings, fake, monkeypatch) -> None:
    fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"))  # the driver probe passes (conftest)
    client = TestClient(create_app(dataclasses.replace(settings, device="auto"), estimator=fake))
    assert client.get("/health").json()["device"] == "cuda"
    runtime.note_cuda_unavailable("onnxruntime built the yolox_m session without CUDAExecutionProvider")
    h = Health.model_validate(client.get("/health").json())
    assert h.device == "cpu" and "without CUDAExecutionProvider" in h.cuda_unavailable


def test_explicit_cuda_without_a_gpu_is_a_loud_503(settings, monkeypatch) -> None:
    """ARGUS_PUPPET_DEVICE=cuda and no GPU: /pose is refused (503, no Retry-After: the operator's fix), nothing runs
    on the CPU, no lease is taken, and /health says why."""
    from argus_puppet.estimator import OnnxEstimator

    _no_gpu(monkeypatch)
    s = dataclasses.replace(settings, device="cuda")
    rec = RecordingCoordinator()
    client = TestClient(create_app(s, estimator=OnnxEstimator(s), coordinator=rec))
    reply = _post(client, draw_image())
    assert reply.status_code == 503 and "retry-after" not in reply.headers
    detail = reply.json()["detail"]
    assert "CUDA was asked for" in detail and "libcuda" in detail and "ARGUS_PUPPET_DEVICE=auto" in detail
    assert rec.leases == []
    h = Health.model_validate(client.get("/health").json())
    assert h.device == "cuda" and "libcuda" in h.cuda_unavailable and h.provider is None


@pytest.mark.parametrize("status", [401, 403, 422])
def test_a_broker_refusal_is_503_without_retry_after(settings, fake, cuda, monkeypatch, status) -> None:
    import httpx
    from argus_cortex.gpu import NullCoordinator
    from argus_cortex.gpu_broker import BrokerCoordinator

    monkeypatch.setenv("ARGUS_GPU_BROKER_TOKEN", "s3cret-token")
    broker = BrokerCoordinator(
        "http://iris.invalid:8120",
        tenant=gpu.TENANT,
        token="s3cret-token",
        fallback=NullCoordinator(),
        transport=httpx.MockTransport(lambda r: httpx.Response(status, text=r.headers.get("authorization", ""))),
    )
    reply = _post(TestClient(create_app(settings, estimator=fake, coordinator=broker)), draw_image())
    assert reply.status_code == 503 and "retry-after" not in reply.headers
    detail = reply.json()["detail"]
    assert "GPU busy" not in detail and "ARGUS_GPU_BROKER_TOKEN" in detail and "s3cret-token" not in detail
    assert fake.calls == 0


def test_health_says_how_the_last_lease_was_granted(settings, fake, cuda) -> None:
    rec = RecordingCoordinator()
    client = TestClient(create_app(settings, estimator=fake, coordinator=rec))
    assert client.get("/health").json()["last_lease_via"] is None
    assert _post(client, draw_image()).status_code == 200
    assert client.get("/health").json()["last_lease_via"] == "local"
