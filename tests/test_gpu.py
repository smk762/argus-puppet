"""GPU leases: none on CPU, one of kind 'pose' on CUDA, the priority header, and the manifest's figures."""

from __future__ import annotations

from pathlib import Path

import pytest
from argus_cortex.gpu_wire import load_tenant_manifest
from conftest import RecordingCoordinator

from argus_puppet import gpu

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
