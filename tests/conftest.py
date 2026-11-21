"""Shared fakes: images drawn in code, an estimator that needs no model, a coordinator that records its leases."""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from argus_cortex.gpu import GpuLease
from PIL import Image, ImageDraw

from argus_puppet import registry
from argus_puppet.config import Settings
from argus_puppet.models import EstimatorProvenance, PersonPose
from argus_puppet.skeleton import NUM_KEYPOINTS


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    for key in list(os.environ):
        if key.startswith(("ARGUS_PUPPET_", "ARGUS_GPU_")):
            monkeypatch.delenv(key)


def draw_image(
    w: int = 64, h: int = 48, fmt: str = "PNG", colour: tuple[int, int, int] = (200, 30, 30), **save
) -> bytes:
    """A small picture (a filled rectangle on grey), encoded."""
    img = Image.new("RGB", (w, h), (120, 120, 120))
    ImageDraw.Draw(img).rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=colour)
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save)
    return buf.getvalue()


def person(x: float, y: float, w: float, h: float, score: float = 0.9) -> PersonPose:
    """A person whose keypoints spread over their box, all scored *score*."""
    kp = np.zeros((NUM_KEYPOINTS, 3))
    kp[:, 0] = x + w * np.linspace(0.1, 0.9, NUM_KEYPOINTS)
    kp[:, 1] = y + h * np.linspace(0.1, 0.9, NUM_KEYPOINTS)
    kp[:, 2] = score
    return PersonPose.from_array(kp, bbox=(x, y, w, h), det_score=score)


def provenance(provider: str = "FakeExecutionProvider") -> EstimatorProvenance:
    return EstimatorProvenance(
        name="fake",
        detector=registry.model_ref(registry.DEFAULT_DETECTOR),
        pose=registry.model_ref(registry.DEFAULT_POSE),
        provider=provider,
        package_version="0.0.0",
        det_threshold=0.3,
        nms_iou=0.45,
    )


@dataclass
class FakeEstimator:
    """Returns the same people for every image (scaled to nothing: coordinates as given) and counts calls."""

    people: list[PersonPose] = field(default_factory=lambda: [person(10, 5, 20, 30, 0.8), person(2, 2, 40, 40, 0.6)])
    calls: int = 0
    sizes: list[tuple[int, int]] = field(default_factory=list)
    provider: str | None = "FakeExecutionProvider"
    error: Exception | None = None

    def estimate(self, image: Image.Image) -> list[PersonPose]:
        if self.error is not None:
            raise self.error
        self.calls += 1
        self.sizes.append(image.size)
        return [p.model_copy() for p in self.people]

    def provenance(self) -> EstimatorProvenance:
        return provenance()


@dataclass
class RecordingCoordinator:
    """A GPU coordinator recording every lease it grants (and every release); can ask for the GPU back."""

    name: str = "recording"
    leases: list[dict[str, Any]] = field(default_factory=list)
    released: int = 0
    yield_after: int | None = None  # ask the first lease to yield once this many images were checked
    error: Exception | None = None
    handles: list[GpuLease] = field(default_factory=list)

    @contextlib.contextmanager
    def lease(self, **kwargs: Any) -> Iterator[GpuLease]:
        if self.error is not None:
            raise self.error
        self.leases.append(kwargs)
        handle = GpuLease()
        if self.yield_after is not None and len(self.leases) == 1:
            handle._yield.set()
        self.handles.append(handle)
        try:
            yield handle
        finally:
            self.released += 1

    def alease(self, **kwargs: Any):  # pragma: no cover - puppet leases from threads only
        raise NotImplementedError


@dataclass
class AsgiReply:
    status: int
    body: bytes
    consumed: int  # request-body bytes the app pulled through receive()


def asgi_request(
    app: Any,
    path: str,
    chunks: Iterable[bytes | float],
    *,
    headers: dict[str, str] | None = None,
    method: str = "POST",
) -> AsgiReply:
    """Drive an ASGI *app* directly, streaming *chunks* (a float entry sleeps that long first) as the body.

    TestClient reads a whole request body before calling the app, so it cannot show how much of an over-cap body
    the server pulls; this can. With no ``content-length`` header the body arrives as a chunked upload would."""
    it = iter(chunks)
    consumed = 0
    pending: bytes | None = None
    done = False

    async def receive() -> dict[str, Any]:
        nonlocal consumed, pending, done
        if done:
            await asyncio.sleep(3600)  # nothing more: the client is waiting for the answer
        while True:
            item = next(it, None)
            if isinstance(item, float):
                await asyncio.sleep(item)
                continue
            break
        if item is None:
            done = True
            return {"type": "http.request", "body": b"", "more_body": False}
        consumed += len(item)
        return {"type": "http.request", "body": item, "more_body": True}

    status = 0
    body = bytearray()

    async def send(message: dict[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body.extend(message.get("body", b""))

    raw_path, _, query = path.partition("?")
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": raw_path,
        "raw_path": raw_path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
    }
    asyncio.run(app(scope, receive, send))
    return AsgiReply(status, bytes(body), consumed)


def megabytes(n: int, head: bytes = b"", tail: bytes = b"") -> Iterator[bytes]:
    """*head*, then *n* MB of zero bytes in 1 MB chunks, then *tail*."""
    yield head
    for _ in range(n):
        yield b"\0" * (1024 * 1024)
    yield tail


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(model_dir=tmp_path / "models", device="cpu")


@pytest.fixture
def fake() -> FakeEstimator:
    return FakeEstimator()


@pytest.fixture
def cuda(monkeypatch) -> None:
    """Make every device hint resolve to CUDA (no real GPU is touched: estimators are fakes)."""
    from argus_puppet import runtime

    monkeypatch.setattr(runtime, "providers", lambda device: [runtime.CUDA, runtime.CPU])
