"""Checks for rows of CLAUDE.md's "Must never happen" table."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import re
import subprocess
from pathlib import Path

from conftest import FakeEstimator, asgi_request, draw_image, megabytes
from fastapi.testclient import TestClient
from starlette.routing import Route

from argus_puppet import registry
from argus_puppet.config import Settings
from argus_puppet.server.app import MULTIPART_OVERHEAD, create_app

ROOT = Path(__file__).resolve().parent.parent

# --- only permissive, pinned estimator weights ------------------------------------------------------------------

PERMISSIVE = {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_every_model_is_permissive_and_pinned() -> None:
    """OpenPose's CMU weights are non-commercial and YOLO-pose is AGPL-3.0: neither may be listed. Every file is
    pinned by its SHA-256 (and its archive's, when it ships in one), so a result's provenance names exact bytes."""
    for name, m in registry.MODELS.items():
        assert m.license in PERMISSIVE, f"{name}: licence {m.license!r} is not in {sorted(PERMISSIVE)}"
        assert HEX64.match(m.sha256), f"{name}: sha256 is not 64 lower-case hex"
        if m.archive_sha256 is not None:
            assert HEX64.match(m.archive_sha256), f"{name}: archive_sha256 is not 64 lower-case hex"
            assert m.archive_member, f"{name}: an archive needs the member that is the model"


# --- no imagery or model file enters the repo -------------------------------------------------------------------

MEDIA = re.compile(
    r"\.(png|jpe?g|webp|gif|bmp|tiff?|avif|heic|ico|mp4|webm|mov|mkv|avi"
    r"|onnx|safetensors|ckpt|pt|pth|bin|npy|npz|tflite|engine|zip)$",
    re.I,
)
BASE64_RUN = re.compile(rb"[A-Za-z0-9+/]{120,}={0,2}")
IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"RIFF", b"GIF8", b"BM", b"II*\x00", b"MM\x00*", b"\x00\x00\x00")
TEXT_SUFFIXES = (".py", ".json", ".md", ".toml", ".yaml", ".yml", ".txt", ".html", ".js", ".ts", ".css", ".svg")


def tracked() -> list[str]:
    # NUL-separated and never quoted: a path with a space or a non-ASCII character comes back whole.
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.split("\0") if p and (ROOT / p).is_file()]


def test_no_image_video_or_model_file_is_tracked() -> None:
    """Tests draw their people in code and models are fetched by hash, so no such file has a reason to be in the
    repo, and one that is cannot be reviewed in a diff (or may carry a real person or a non-permissive licence)."""
    assert [p for p in tracked() if MEDIA.search(p)] == []


def test_no_image_is_hidden_in_a_base64_literal() -> None:
    """Nor is an image pasted into source as base64."""
    found = []
    for path in tracked():
        if not path.endswith(TEXT_SUFFIXES):
            continue
        for run in BASE64_RUN.findall((ROOT / path).read_bytes()):
            try:
                head = base64.b64decode(run[:64] + b"=" * (-len(run[:64]) % 4), validate=True)
            except (binascii.Error, ValueError):
                continue
            if head.startswith(IMAGE_MAGIC):
                found.append(path)
    assert found == []


# --- every state-changing route is guarded; a caller's path stays under the data root ---------------------------

EVIL = {"origin": "https://evil.example", "sec-fetch-site": "cross-site"}
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}
# Every unsafe route the app has today: a new one is walked automatically, a missing one fails the walk.
EXPECTED_UNSAFE = (
    ("POST", "/pose"),
    ("POST", "/pose/scan"),
    ("POST", "/stats"),
    ("POST", "/compare"),
    ("POST", "/render"),
)


def _app(tmp_path: Path, **changes):
    root = tmp_path / "data"
    (root / "set").mkdir(parents=True, exist_ok=True)
    (root / "set" / "a.png").write_bytes(draw_image())
    settings = Settings(model_dir=tmp_path / "models", device="cpu", data_root=root, **changes)
    return create_app(settings, estimator=FakeEstimator()), root


def _unsafe_routes(app) -> list[tuple[str, str]]:
    found = []
    for route in app.routes:
        if isinstance(route, Route):
            found += [(m, route.path) for m in sorted((route.methods or set()) & UNSAFE)]
    return found


def test_every_unsafe_route_refuses_another_site(tmp_path) -> None:
    """Every POST/PUT/PATCH/DELETE route: another site's page gets 403, whatever the route is (CORS is no write
    boundary: a multipart POST needs no preflight)."""
    app, root = _app(tmp_path)
    routes = _unsafe_routes(app)
    assert set(EXPECTED_UNSAFE) <= set(routes) and len(routes) >= len(EXPECTED_UNSAFE)  # the table was walked
    with TestClient(app) as client:
        for method, path in routes:
            reply = client.request(method, path, headers=EVIL, json={"path": "set"})
            assert reply.status_code == 403, f"{method} {path} answered {reply.status_code}"
    assert not (root / "set" / "a.pose.json").exists()


def test_read_only_mode_refuses_every_unsafe_route(tmp_path) -> None:
    app, root = _app(tmp_path, read_only=True)
    routes = _unsafe_routes(app)
    assert set(EXPECTED_UNSAFE) <= set(routes)
    with TestClient(app) as client:
        for method, path in routes:
            reply = client.request(method, path, json={"path": "set"})
            assert reply.status_code == 403, f"{method} {path} answered {reply.status_code} in read-only mode"
            assert "read-only" in reply.json()["detail"]
        assert client.get("/health").status_code == 200  # reads still work
    assert not (root / "set" / "a.pose.json").exists()


def test_the_guards_run_in_the_documented_order(tmp_path) -> None:
    """CORS outermost (a refusal is readable cross-origin), then read-only, then the origin guard."""
    app, _ = _app(tmp_path, read_only=True, cors_origins=("https://evil.example",))
    with TestClient(app) as client:
        reply = client.post("/stats", headers={**EVIL, "origin": "https://other.example"}, json={"path": "set"})
        assert reply.status_code == 403 and "read-only" in reply.json()["detail"]  # read-only answered first
    app, _ = _app(tmp_path, cors_origins=("https://evil.example",))
    with TestClient(app) as client:
        reply = client.post("/stats", headers={**EVIL, "origin": "https://other.example"}, json={"path": "set"})
        assert reply.status_code == 403 and "read-only" not in reply.json()["detail"]
    app, _ = _app(tmp_path, read_only=True, cors_origins=("https://evil.example",))
    with TestClient(app) as client:
        reply = client.post("/stats", headers=EVIL, json={"path": "set"})
        assert reply.status_code == 403 and reply.headers["access-control-allow-origin"] == "https://evil.example"


def test_a_scan_path_escaping_the_data_root_is_refused(tmp_path) -> None:
    app, root = _app(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "b.png").write_bytes(draw_image())
    (root / "escape").symlink_to(outside)
    with TestClient(app) as client:
        for path in ("../outside", str(outside), "set/../../outside", "escape"):
            reply = client.post("/pose/scan", json={"path": path})
            assert reply.status_code == 400, f"{path!r} answered {reply.status_code}"
    assert not (outside / "b.pose.json").exists()
    with TestClient(app) as client:  # the set routes read sidecars under the same fence
        for path in ("../outside", str(outside), "set/../../outside", "escape"):
            assert client.post("/stats", json={"path": path}).status_code == 400, path
            reply = client.post("/compare", json={"outputs": {"path": "set"}, "reference": {"path": path}})
            assert reply.status_code == 400, path
    unrooted = create_app(dataclasses.replace(app.state.puppet.settings, data_root=None), estimator=FakeEstimator())
    with TestClient(unrooted) as client:
        assert client.post("/pose/scan", json={"path": "set"}).status_code == 409


# --- every body is bounded: an over-cap body is refused unread, a stalled one cut off --------------------------------

MB = 1024 * 1024
# The content type each unsafe route reads; any route not listed reads JSON. A new route that reads something else
# fails the walk below until it is listed here (and caps its body).
BODY_TYPES: dict[str, tuple[str, ...]] = {"/pose": ("image/png", "multipart/form-data; boundary=B")}


def _bodies(app) -> list[tuple[str, str, str]]:
    routes = _unsafe_routes(app)
    assert set(EXPECTED_UNSAFE) <= set(routes)
    return [(m, p, ct) for m, p in routes for ct in BODY_TYPES.get(p, ("application/json",))]


def test_every_unsafe_route_refuses_an_over_cap_body_without_reading_it(tmp_path) -> None:
    """Chunked (no Content-Length): the server stops pulling at its cap, whatever the route. Declared: refused before
    a byte is read. The largest cap here is 1 MB (+ 64 KiB multipart overhead); 32 MB are offered."""
    app, root = _app(tmp_path, max_upload_bytes=MB, max_json_bytes=MB)
    for method, path, content_type in _bodies(app):
        reply = asgi_request(
            app, path, megabytes(32, b'{"path": "set", "pad": "'), headers={"content-type": content_type}, method=method
        )
        assert reply.status == 413, f"{method} {path} ({content_type}) answered {reply.status}: {reply.body[:200]!r}"
        assert reply.consumed <= MB + MULTIPART_OVERHEAD + MB + 64, f"{method} {path} read {reply.consumed} bytes"
        declared = {"content-type": content_type, "content-length": str(32 * MB)}
        reply = asgi_request(app, path, megabytes(32), headers=declared, method=method)
        assert reply.status == 413 and reply.consumed == 0, f"{method} {path} read a declared over-cap body"
    assert not (root / "set" / "a.pose.json").exists()


def test_every_unsafe_route_cuts_off_a_stalled_body_and_frees_its_slot(tmp_path) -> None:
    """A body that stops arriving gets 408 at ARGUS_PUPPET_BODY_TIMEOUT_S on every route, and no admission slot stays
    held: a slow client cannot lock the service up."""
    app, _ = _app(tmp_path, body_timeout_s=0.2, pose_concurrency=1)
    state = app.state.puppet
    for method, path, content_type in _bodies(app):
        reply = asgi_request(app, path, [b'{"path": ', 30.0], headers={"content-type": content_type}, method=method)
        assert reply.status == 408, f"{method} {path} ({content_type}) answered {reply.status}"
        held = [a.in_flight for a in (state.admit, state.scan_admit, state.compute_admit)]
        assert held == [0, 0, 0], f"{method} {path} left a slot held: {held}"


def test_only_a_final_release_moves_latest() -> None:
    """The release workflow runs on every ``v*`` tag: a pre-release (``v0.2.0-rc1``) must not repoint ``latest``."""
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in workflow.splitlines() if "value=latest" in ln and not ln.strip().startswith("#")]
    assert lines, "the release workflow no longer tags latest: update this test"
    for line in lines:
        assert "enable=${{ !contains(needs.release-tag.outputs.tag, '-') }}" in line, line
