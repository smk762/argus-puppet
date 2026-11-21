"""Fetching and verifying the pinned model files, with a fake fetcher (no network)."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import zipfile

import pytest

from argus_puppet import registry, weights
from argus_puppet.config import Settings

ONNX = b"pretend onnx graph bytes " * 10


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def s(tmp_path) -> Settings:
    return Settings(model_dir=tmp_path / "models", device="cpu")


def pin(monkeypatch, name: str, onnx: bytes, archive: bytes | None = None) -> registry.PinnedModel:
    m = dataclasses.replace(
        registry.get(name),
        sha256=_sha(onnx),
        bytes=len(onnx),
        archive_sha256=_sha(archive) if archive is not None else None,
        archive_member="end2end.onnx" if archive is not None else None,
    )
    monkeypatch.setitem(registry.MODELS, name, m)
    return m


class Fetcher:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls: list[tuple[str, int]] = []

    def __call__(self, url: str, max_bytes: int) -> bytes:
        self.calls.append((url, max_bytes))
        return self.body


def test_fetches_a_plain_onnx_and_writes_it_verified(monkeypatch, s) -> None:
    m = pin(monkeypatch, "yolox_nano", ONNX)
    f = Fetcher(ONNX)
    path = weights.fetch_model(s, m.name, fetcher=f)
    assert path == s.model_path(m.name) and path.read_bytes() == ONNX
    assert f.calls == [(m.url, len(ONNX) + (1 << 20))]
    assert list(path.parent.iterdir()) == [path]  # no temp file left behind
    weights.fetch_model(s, m.name, fetcher=f)  # already there and verified: no second download
    assert len(f.calls) == 1
    weights.fetch_model(s, m.name, fetcher=f, force=True)
    assert len(f.calls) == 2


def test_extracts_the_member_of_a_verified_archive(monkeypatch, s) -> None:
    archive = _zip({"20230831/rtmpose/README.md": b"hi", "20230831/rtmpose/end2end.onnx": ONNX})
    m = pin(monkeypatch, "dwpose-m-256x192", ONNX, archive)
    weights.fetch_model(s, m.name, fetcher=Fetcher(archive))
    assert s.model_path(m.name).read_bytes() == ONNX


@pytest.mark.parametrize(
    ("served", "match"),
    [
        (b"tampered" + ONNX[8:], "SHA-256"),
        (ONNX[:-1], "bytes"),
    ],
)
def test_a_file_that_does_not_verify_is_never_written(monkeypatch, s, served, match) -> None:
    m = pin(monkeypatch, "yolox_nano", ONNX)
    with pytest.raises(weights.WeightsError, match=match):
        weights.fetch_model(s, m.name, fetcher=Fetcher(served))
    assert not s.model_path(m.name).exists()


def test_an_archive_with_the_wrong_hash_is_refused(monkeypatch, s) -> None:
    archive = _zip({"end2end.onnx": ONNX})
    m = pin(monkeypatch, "dwpose-m-256x192", ONNX, archive)
    other = _zip({"end2end.onnx": ONNX, "extra": b"x"})
    with pytest.raises(weights.WeightsError, match="archive"):
        weights.fetch_model(s, m.name, fetcher=Fetcher(other))
    assert not s.model_path(m.name).exists()


@pytest.mark.parametrize(
    ("members", "match"),
    [
        ({"../../end2end.onnx": ONNX}, "refusing"),
        ({"/abs/end2end.onnx": ONNX}, "refusing"),
        ({"a/end2end.onnx": ONNX, "b/end2end.onnx": ONNX}, "exactly one"),
        ({"model.onnx": ONNX}, "exactly one"),
        ({"end2end.onnx": ONNX + b"more"}, "pinned"),
    ],
)
def test_archive_tricks_are_refused(members, match) -> None:
    with pytest.raises(weights.WeightsError, match=match):
        weights.extract_member(_zip(members), "end2end.onnx", len(ONNX))


def test_not_a_zip() -> None:
    with pytest.raises(weights.WeightsError, match="not a zip"):
        weights.extract_member(b"plain bytes", "end2end.onnx", 3)


def test_a_fetch_error_becomes_a_weights_error(monkeypatch, s) -> None:
    m = pin(monkeypatch, "yolox_nano", ONNX)

    def boom(url, max_bytes):
        raise TimeoutError("slow")

    with pytest.raises(weights.WeightsError, match="could not download"):
        weights.fetch_model(s, m.name, fetcher=boom)


def test_status_reports_presence_and_verification(monkeypatch, s) -> None:
    m = pin(monkeypatch, "yolox_nano", ONNX)
    s.model_dir.mkdir(parents=True)
    s.model_path(m.name).write_bytes(ONNX)
    s.model_path("yolox_m").write_bytes(b"wrong")
    by_name = {r.name: r for r in weights.status(s)}
    assert set(by_name) == set(registry.names())
    assert by_name["yolox_nano"].present and by_name["yolox_nano"].verified
    assert by_name["yolox_m"].present and by_name["yolox_m"].verified is False
    assert not by_name[registry.DEFAULT_POSE].present
    assert {r.verified for r in weights.status(s, verify=False)} == {None}


# ── the real fetcher's host allow-list ───────────────────────────────────────────────────────────────────────────


class Reply:
    def __init__(self, url: str, redirects: list[str]) -> None:
        self.url, self.redirects, self.body = url, redirects, b"data"

    def raise_for_status(self):
        return self


def test_net_fetcher_passes_timeouts_and_a_cap(monkeypatch) -> None:
    seen = {}

    def fetch(url, **kw):
        seen.update(kw, url=url)
        return Reply("https://release-assets.githubusercontent.com/x", ["https://github.com/a/b"])

    monkeypatch.setattr("argus_cortex.net.fetch", fetch)
    assert weights.net_fetcher("https://github.com/a/b", 1234) == b"data"
    assert seen["timeout"] > 0 and seen["deadline"] > 0 and seen["max_bytes"] == 1234


def test_net_fetcher_refuses_other_hosts(monkeypatch) -> None:
    monkeypatch.setattr("argus_cortex.net.fetch", lambda url, **kw: Reply("https://evil.example/x", [url]))
    with pytest.raises(weights.WeightsError, match="not an allowed model host"):
        weights.net_fetcher("https://example.com/model.onnx", 10)
    with pytest.raises(weights.WeightsError, match="redirected"):
        weights.net_fetcher("https://download.openmmlab.com/m.zip", 10)


def test_every_registry_url_is_on_an_allowed_host() -> None:
    for m in registry.MODELS.values():
        assert weights._host(m.url) in weights.ALLOWED_HOSTS, m.name
