"""compose.yaml carries the discovery labels argus-iris's agents read (argus_cortex.service)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from argus_cortex.peers import PEER_NAME_PATTERN
from argus_cortex.service import HEALTH_LABEL, PEER_ENV, PEER_LABEL, PORT_LABEL, SERVICE_LABEL

from argus_puppet.config import DEFAULT_PORT
from argus_puppet.server.app import SERVICE

COMPOSE = Path(__file__).resolve().parent.parent / "compose.yaml"


def test_both_services_carry_the_discovery_labels() -> None:
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    # The GPU service keeps its argus-iris tenant's name: iris attributes its VRAM by it.
    assert set(services) == {"puppet", "argus-puppet"}
    for name, service in services.items():
        labels = service["labels"]
        assert labels == {
            PEER_LABEL: "puppet",
            SERVICE_LABEL: SERVICE,
            HEALTH_LABEL: "/health",
            PORT_LABEL: str(DEFAULT_PORT),
        }, name
        assert re.fullmatch(PEER_NAME_PATTERN, labels[PEER_LABEL])
        assert service["environment"][PEER_ENV] == labels[PEER_LABEL], name
        assert any(str(p).endswith(f":{labels[PORT_LABEL]}") for p in service["ports"]), name
