"""The argus-iris tenant manifest validates against argus-cortex's TenantManifest."""

from __future__ import annotations

from pathlib import Path

from argus_cortex.gpu_wire import load_tenant_manifest

MANIFEST = Path(__file__).resolve().parent.parent / "gpu-tenant.yaml"


def test_the_manifest_validates() -> None:
    manifest = load_tenant_manifest(MANIFEST)
    assert manifest.tenant == "argus-puppet"
    assert manifest.repo == "smk762/argus-puppet"
    pose = manifest.kind_for("pose")
    assert pose is not None
    assert pose.priority == "batch" and pose.preemptible is False
    assert pose.model_mb is not None and pose.model_mb < pose.vram_mb
    assert manifest.residents == ()
