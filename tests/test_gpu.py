"""GPU leases: none on CPU, one of kind 'pose' on CUDA, the priority header, and the manifest's figures."""

from __future__ import annotations

from pathlib import Path

import pytest
from argus_cortex.gpu_wire import load_tenant_manifest
from conftest import RecordingCoordinator, fake_ort

from argus_puppet import gpu, runtime

MANIFEST = Path(__file__).resolve().parent.parent / "gpu-tenant.yaml"


def test_the_footprint_matches_the_tenant_manifest() -> None:
    manifest = load_tenant_manifest(MANIFEST)
    kind = manifest.kind_for(gpu.KIND)
    assert manifest.tenant == gpu.TENANT
    assert (kind.vram_mb, kind.model_mb) == (gpu.VRAM_MB, gpu.MODEL_MB)
    assert kind.priority == gpu.BATCH_PRIORITY


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        (None, "interactive"),
        ("", "interactive"),
        ("batch", "batch"),
        (" Background ", "background"),
        ("realtime", "interactive"),
        ("interactive", "interactive"),
    ],
)
def test_parse_priority(raw, want) -> None:
    assert gpu.parse_priority(raw) == want


def test_parse_priority_refuses_unknown_classes() -> None:
    with pytest.raises(ValueError, match=gpu.PRIORITY_HEADER):
        gpu.parse_priority("urgent")


def test_no_lease_on_cpu() -> None:
    rec = RecordingCoordinator()
    with gpu.lease("cpu", priority="interactive", coordinator=rec) as handle:
        assert handle is None
    assert rec.leases == []


def test_a_lease_is_requested_on_cuda(cuda) -> None:
    rec = RecordingCoordinator()
    with gpu.lease("cuda", priority="interactive", coordinator=rec, caller="image") as handle:
        assert handle is not None
    assert rec.leases == [
        {
            "caller": "argus-puppet:image",
            "min_vram_mb": gpu.VRAM_MB,
            "model_mb": gpu.MODEL_MB,
            "kind": "pose",
            "priority": "interactive",
        }
    ]
    assert rec.released == 1


def test_the_default_coordinator_comes_from_the_environment(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "_coordinator", None)
    assert gpu.default_coordinator().name == "none"
    assert gpu.default_coordinator() is gpu.default_coordinator()


def test_a_folder_lease_gives_the_gpu_back_when_asked(cuda) -> None:
    rec = RecordingCoordinator(yield_after=1)
    with gpu.folder_lease("cuda", coordinator=rec) as held:
        held.checkpoint()  # the first lease asks to yield: released and retaken
        assert rec.released == 1 and len(rec.leases) == 2
        held.checkpoint()  # the second does not
        assert len(rec.leases) == 2
    assert rec.released == 2
    assert {lease["priority"] for lease in rec.leases} == {"batch"}


def test_no_lease_for_work_that_runs_on_the_cpu_of_a_cuda_build(monkeypatch) -> None:
    """A CUDA build with no usable GPU (ORT has the provider compiled in; the driver sees nothing): its work runs on
    the CPU, so neither auto nor cuda leases the broker for it."""
    fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"))
    monkeypatch.setattr(runtime, "_probe_cuda_driver", lambda: "libcuda.so.1 cannot be loaded")
    rec = RecordingCoordinator()
    for device in ("auto", "cuda"):
        with gpu.lease(device, priority="interactive", coordinator=rec) as handle:
            assert handle is None
        with gpu.folder_lease(device, coordinator=rec) as held:
            assert held.handle is None
    assert rec.leases == []


def test_no_lease_once_a_session_came_back_without_cuda(monkeypatch, tmp_path) -> None:
    """The probe passed, but the first session fell back to the CPU: from then on no lease is taken."""
    fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"))
    rec = RecordingCoordinator()
    with gpu.lease("auto", priority="interactive", coordinator=rec) as handle:
        assert handle is not None
    runtime.note_cuda_unavailable("onnxruntime built the session without CUDAExecutionProvider")
    with gpu.lease("auto", priority="interactive", coordinator=rec) as handle:
        assert handle is None
    assert len(rec.leases) == 1


def test_the_way_the_last_lease_was_granted_is_kept(cuda) -> None:
    rec = RecordingCoordinator()
    assert gpu.last_via(rec) is None
    with gpu.lease("cuda", priority="interactive", coordinator=rec):
        pass
    assert gpu.last_via(rec) == "local"


def _broker(status: int, token: str = "s3cret-token"):
    """A real cortex BrokerCoordinator in front of a broker that answers every call with *status* (echoing the
    Authorization header it got, as a careless broker might)."""
    import httpx
    from argus_cortex.gpu import NullCoordinator
    from argus_cortex.gpu_broker import BrokerCoordinator

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=f"refused: {request.headers.get('authorization', '')}")

    return BrokerCoordinator(
        "http://iris.invalid:8120",
        tenant=gpu.TENANT,
        token=token,
        fallback=NullCoordinator(),
        transport=httpx.MockTransport(handler),
        timeout_s=1.0,
    )


@pytest.mark.parametrize("status", [401, 403, 422])
def test_a_broker_refusal_is_the_operators_not_a_busy_gpu(cuda, monkeypatch, status) -> None:
    monkeypatch.setenv("ARGUS_GPU_BROKER_TOKEN", "s3cret-token")
    with (
        pytest.raises(gpu.OPERATOR_ERRORS) as caught,
        gpu.lease("cuda", priority="interactive", coordinator=_broker(status)),
    ):
        pass
    assert not isinstance(caught.value, gpu.LEASE_ERRORS)
    message = gpu.operator_message(caught.value)
    assert "ARGUS_GPU_BROKER_TOKEN" in message and "not a busy GPU" in message and str(status) in message
    assert "s3cret-token" not in message


def test_a_broker_token_from_the_suite_file_is_not_echoed_either(cuda, monkeypatch, tmp_path) -> None:
    suite = tmp_path / "suite.env"
    suite.write_text("ARGUS_GPU_BROKER_TOKEN=s3cret-file-token\n")
    suite.chmod(0o600)
    monkeypatch.setenv("ARGUS_ENV_FILE", str(suite))
    with (
        pytest.raises(gpu.OPERATOR_ERRORS) as caught,
        gpu.lease("cuda", priority="interactive", coordinator=_broker(401)),
    ):
        pass
    exc = type(caught.value)(f"{caught.value} (token s3cret-file-token)")
    assert "s3cret-file-token" not in gpu.operator_message(exc)


def test_an_unreachable_broker_falls_back_rather_than_refusing(cuda) -> None:
    with gpu.lease("cuda", priority="interactive", coordinator=(coord := _broker(503))) as handle:
        assert handle.via == "fallback"
    assert gpu.last_via(coord) == "fallback"
