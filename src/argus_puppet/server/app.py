"""The argus-puppet HTTP service (``:8220``): pose one uploaded image, or a folder under the data root.

Routes:

* ``GET /health`` -> :class:`~argus_puppet.models.Health`.
* ``POST /pose`` -> :class:`~argus_puppet.models.PoseResult` for one image (raw bytes or a multipart file). Writes
  nothing. ``?name=`` names the image in the result; ``?face=x,y,w,h`` is a primary-face hint. Concurrency is
  bounded (``ARGUS_PUPPET_POSE_CONCURRENCY``): a call past it is answered 503 + Retry-After, not queued.
* ``POST /pose/scan`` :class:`~argus_puppet.models.ScanRequest` ``{path, recursive, overwrite}`` ->
  :class:`~argus_puppet.models.ScanSummary`. Writes sidecars under ``ARGUS_PUPPET_DATA_ROOT``; *path* is fenced under
  it with :func:`argus_cortex.fs.safe_join`.
* ``POST /stats`` :class:`~argus_puppet.models.StatsRequest` (``{results}`` or ``{path, recursive}``, ``params``) ->
  :class:`~argus_puppet.models.PoseSetStats`. Writes nothing.
* ``POST /compare`` :class:`~argus_puppet.models.CompareRequest` (``outputs`` and ``reference``, each results or a
  path; ``threshold``, ``params``) -> :class:`~argus_puppet.models.PoseSetComparison`. Writes nothing.
* ``POST /render`` :class:`~argus_puppet.models.RenderRequest` (``pose``: a PoseResult or ``{keypoints, width,
  height}``; ``options``) -> ``image/png``. Writes nothing.

A set given as ``path`` is the ``*.pose.json`` sidecars of a folder under the data root, fenced as ``/pose/scan``
fences its path (409 without a data root, 400 outside it, 404 missing). A set holds at most
``ARGUS_PUPPET_MAX_SET_RESULTS`` results (413 past it: the distance matrix is O(N^2)); a JSON body at most
``ARGUS_PUPPET_MAX_JSON_BYTES`` (413); a render canvas at most ``ARGUS_PUPPET_MAX_PIXELS`` pixels and
``ARGUS_PUPPET_MAX_PEOPLE`` people (413). The compute routes share one admission bound
(``ARGUS_PUPPET_POSE_CONCURRENCY``; 503 + Retry-After past it). Every body is read through :func:`read_capped`:
counted against its cap as it streams, declared length or not, and cut off at ``ARGUS_PUPPET_BODY_TIMEOUT_S`` (408),
so a slow client cannot hold an admission slot.

There is no auth: reaching the port is host access. Every unsafe method passes the origin
:class:`~argus_cortex.server.WriteGuard` (another site's page is refused), then, in read-only mode, a second guard
refusing every write; CORS is outermost so a refused cross-origin write is still readable. Inference runs off the
event loop (``asyncio.to_thread``).

Extending: add routes to the app :func:`create_app` returns, reaching the shared pieces through
``app.state.puppet`` (:class:`AppState`: ``settings``, ``estimator()``, ``coordinator``, ``admit``).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from argus_cortex.fs import PathRefused, is_within, safe_basename, safe_join
from argus_cortex.gpu import GpuCoordinator, free_vram_mb
from argus_cortex.limits import BadInput, TooLarge, read_body, read_upload
from argus_cortex.server import WriteGuard, constant_refuse, cross_site_refuse
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from argus_puppet import __version__, compare, extract, gpu, metrics, registry, render, runtime, sidecars
from argus_puppet import estimator as estimator_mod
from argus_puppet.config import Settings
from argus_puppet.models import (
    BarePose,
    CompareRequest,
    Health,
    PoseResult,
    PoseSetComparison,
    PoseSetSource,
    PoseSetStats,
    RenderRequest,
    ScanRequest,
    ScanSummary,
    StatsRequest,
)

READ_ONLY_REFUSAL = "argus-puppet is read-only (ARGUS_PUPPET_READ_ONLY): state-changing requests are refused"
# A multipart body counts boundaries and part headers too: the whole body (declared or chunked) may be this much over
# the cap, and read_upload then holds the file itself to the exact cap.
MULTIPART_OVERHEAD = 64 * 1024
RAW_TYPES = ("application/octet-stream", "image/")
# A ScanRequest is a path and two flags: anything past this is not one.
SCAN_JSON_BYTES = 16 * 1024
# A RenderRequest is one pose: 500 people (models.MAX_PEOPLE) with real coordinates are under 2 MB. A bare pose's
# people share one "keypoints" key, so they cannot be counted on the raw bytes like a set's: this cap bounds them
# (zero-filled people cost ~1 KB of JSON and ~18 KB validated).
RENDER_JSON_BYTES = 4 * 1024 * 1024
BUSY_RETRY_S = 1
GPU_RETRY_S = 5


class Admission:
    """At most *limit* calls in flight; a call past it is refused at once (never queued).

    ``Semaphore.locked()`` is not a non-blocking acquire, so the count is kept under a lock."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.in_flight = 0
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._lock:
            if self.in_flight >= self.limit:
                raise HTTPException(503, "busy: try again", headers={"Retry-After": str(BUSY_RETRY_S)})
            self.in_flight += 1
        try:
            yield
        finally:
            async with self._lock:
                self.in_flight -= 1


@dataclass
class AppState:
    """What routes share; ``app.state.puppet``."""

    settings: Settings
    coordinator: GpuCoordinator | None = None
    fixed_estimator: estimator_mod.Estimator | None = None
    admit: Admission = field(default_factory=lambda: Admission(1))
    scan_admit: Admission = field(default_factory=lambda: Admission(1))
    compute_admit: Admission = field(default_factory=lambda: Admission(1))

    def estimator(self) -> estimator_mod.Estimator:
        """The injected estimator, else the shared one for the settings (cheap: its sessions load lazily)."""
        if self.fixed_estimator is not None:
            return self.fixed_estimator
        return estimator_mod.get_estimator(self.settings)


def install_guards(app: FastAPI, settings: Settings) -> Callable[..., str | None]:
    """The write guards and CORS: origin guard, then read-only guard, then CORS outermost. Returns the origin
    predicate."""
    wildcard = "*" in settings.cors_origins
    named = [o for o in settings.cors_origins if o != "*"]
    origin_refusal = cross_site_refuse(named, settings.trusted_hosts)
    # A wildcard grants anonymous reads, never a cross-site write: naming an origin is the way to get those.
    app.add_middleware(WriteGuard, refuse=origin_refusal)
    if settings.read_only:
        app.add_middleware(WriteGuard, refuse=constant_refuse(READ_ONLY_REFUSAL))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else named,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return origin_refusal


def parse_face_hint(raw: str | None) -> tuple[float, float, float, float] | None:
    """``?face=x,y,w,h`` as a box; ``None`` when absent. Raises ``HTTPException(400)`` when malformed."""
    if raw is None or not raw.strip():
        return None
    try:
        values = tuple(float(v) for v in raw.split(","))
    except ValueError:
        values = ()
    if len(values) != 4 or not all(math.isfinite(v) for v in values) or values[2] <= 0 or values[3] <= 0:
        raise HTTPException(400, "face must be x,y,w,h: four finite numbers, w and h > 0")
    return values  # type: ignore[return-value]


def model_errors(exc: Exception) -> HTTPException:
    """The HTTP answer for an error out of the estimator or the lease; re-raises anything else."""
    if isinstance(exc, runtime.ModelUnavailable):
        return HTTPException(503, str(exc))
    if isinstance(exc, gpu.LEASE_ERRORS):
        return HTTPException(503, f"GPU busy: {exc}", headers={"Retry-After": str(GPU_RETRY_S)})
    raise exc


def _replayed(request: Request, body: bytes) -> Request:
    """A copy of *request* whose body is *body*, already read (so its form parses from memory)."""
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(request.scope, receive=receive)


async def read_capped(request: Request, cap: int, *, timeout: float, what: str) -> bytes:
    """The body of a ``POST``, at most *cap* bytes, arriving within *timeout* seconds: every route's body is read
    through here. 413 over the cap (a declared length at once, a chunked body as it streams), 400 malformed, 408 when
    it has not all arrived in time (the caller's admission slot is then freed, and the connection closed)."""
    try:
        async with asyncio.timeout(timeout):
            return await read_body(request, cap, what=what)
    except TooLarge as exc:
        raise HTTPException(413, str(exc)) from None
    except BadInput as exc:
        raise HTTPException(400, str(exc)) from None
    except TimeoutError:
        raise HTTPException(
            408,
            f"the {what} did not arrive within {timeout:g}s (ARGUS_PUPPET_BODY_TIMEOUT_S)",
            headers={"Connection": "close"},
        ) from None


async def read_image(request: Request, cap: int, *, timeout: float) -> tuple[bytes, str | None]:
    """The image of a ``POST``: raw bytes or the first file of a multipart form, at most *cap* bytes. Returns the
    bytes and the multipart filename, if any. 413 over the cap, 400 malformed, 408 too slow, 415 an unsupported
    content type."""
    content_type = (request.headers.get("content-type") or "").lower()
    if content_type.startswith("multipart/form-data"):
        # The whole body, declared or chunked, is read against the cap plus the multipart overhead (an over-cap
        # stream stops there), then parsed from memory; read_upload holds the file to the exact cap.
        raw = await read_capped(request, cap + MULTIPART_OVERHEAD, timeout=timeout, what="image")
        try:
            async with _replayed(request, raw).form() as form:
                upload = next((v for v in form.values() if hasattr(v, "read")), None)
                if upload is None:
                    raise HTTPException(400, "no image file in the form")
                return await read_upload(upload, cap, what="image"), getattr(upload, "filename", None)
        except TooLarge as exc:
            raise HTTPException(413, str(exc)) from None
    if content_type and not content_type.startswith(RAW_TYPES):
        raise HTTPException(415, "send the image as raw bytes (application/octet-stream or image/*) or multipart")
    return await read_capped(request, cap, timeout=timeout, what="image"), None


def _name(raw: str | None, filename: str | None) -> str:
    """The upload's name in the result: ``?name=``, else the multipart filename, else ``upload``; sanitised."""
    for candidate, label in ((raw, "name"), (filename, "filename")):
        if candidate:
            try:
                return safe_basename(candidate, label=label)
            except PathRefused as exc:
                if label == "name":
                    raise HTTPException(400, str(exc)) from None
    return "upload"


def health(state: AppState) -> Health:
    """``GET /health``'s body (blocking: it may hash the model files once; call it off the event loop)."""
    settings = state.settings
    cuda = runtime.uses_cuda(settings.device)
    est = state.fixed_estimator or estimator_mod.get_estimator(settings)
    try:
        coordinator = (state.coordinator or gpu.default_coordinator()).name
    except ValueError:  # an unknown ARGUS_GPU_COORDINATOR: leases would fail the same way
        coordinator = "invalid"
    return Health(
        version=__version__,
        read_only=settings.read_only,
        models={n: runtime.model_verified(settings, n) for n in registry.names()},
        pose_model=settings.pose_model,
        detector_model=settings.detector_model,
        provider=getattr(est, "provider", None),
        device="cuda" if cuda else "cpu",
        gpu_coordinator=coordinator,
        free_vram_mb=free_vram_mb(0) if cuda else None,
    )


# ── JSON bodies and pose sets ────────────────────────────────────────────────────────────────────────────────────


M = TypeVar("M", bound=BaseModel)


async def read_json(
    request: Request,
    model: type[M],
    cap: int,
    *,
    timeout: float,
    precheck: Callable[[bytes], None] | None = None,
) -> M:
    """The JSON body of a ``POST`` as *model*: at most *cap* bytes (413), within *timeout* seconds (408), JSON (415
    otherwise), valid (422 with the errors, never echoing the input). *precheck* looks at the raw bytes first (it raises
    to refuse). Both run off the event loop: a large set takes a while."""
    content_type = (request.headers.get("content-type") or "").lower()
    if content_type and not content_type.startswith("application/json"):
        raise HTTPException(415, "send the request as application/json")
    body = await read_capped(request, cap, timeout=timeout, what="request body")

    def validate() -> M:
        if precheck is not None:
            precheck(body)
        return model.model_validate_json(body)

    try:
        return await asyncio.to_thread(validate)
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_input=False, include_context=False)) from None


def _key_counter(key: str) -> Callable[[bytes], int]:
    """An upper bound on how many times JSON object key *key* occurs in a raw body, from its bytes alone (C-speed
    scans, no parse): written literally (a ``"`` inside a string is escaped, so a string value cannot fake one), or
    with at least one of its characters escaped as ``\\u00XX`` (each such escape counted once, as a decoder accepts
    them). No JSON encoder escapes ASCII letters, digits or ``_`` unasked, so a legitimate body's escapes (non-ASCII
    names) never inflate the count."""
    literal = b'"' + key.encode("ascii") + b'"'
    codes = b"|".join(sorted({f"{ord(c):02x}".encode() for c in key}))
    escape = re.compile(rb"\\u00(?:" + codes + rb")", re.IGNORECASE)

    def count(body: bytes) -> int:
        return body.count(literal) + len(escape.findall(body))

    return count


max_results_in = _key_counter("image_sha256")  # one per PoseResult
max_people_in = _key_counter("keypoints")  # one per person (PersonPose, or a bare render pose)
# The least JSON a real person takes (133 points with real coordinates are ~3.3 KB). A set body may carry at most
# max_json_bytes / this many people, so zero-filled people (~1 KB each, ~18 KB once validated) buy no more of them
# than real ones of the same size would.
PERSON_JSON_FLOOR = 2048


def set_precheck(settings: Settings, sets: int) -> Callable[[bytes], None]:
    """413 for a body that cannot fit *sets* sets of ``max_set_results``, or holds more people than its size allows,
    decided on the raw bytes before anything is validated. Validation costs 15-40 times the body's size in memory,
    and pydantic's own list ``max_length`` does not help: its error holds the whole list, converted."""
    results_limit = settings.max_set_results * sets
    people_limit = settings.max_json_bytes // PERSON_JSON_FLOOR

    def check(body: bytes) -> None:
        n = max_results_in(body)
        if n > results_limit:
            if sets == 1:
                raise _too_many(n, settings)
            raise HTTPException(
                413,
                f"{n} poses in one request is over the cap of {sets} sets of {settings.max_set_results} "
                "(ARGUS_PUPPET_MAX_SET_RESULTS)",
            )
        people = max_people_in(body)
        if people > people_limit:
            raise HTTPException(
                413,
                f"{people} people in one request is over the cap of {people_limit} "
                f"(ARGUS_PUPPET_MAX_JSON_BYTES / {PERSON_JSON_FLOOR})",
            )

    return check


def _too_many(n: int, settings: Settings) -> HTTPException:
    return HTTPException(
        413, f"{n} poses in one set is over the cap of {settings.max_set_results} (ARGUS_PUPPET_MAX_SET_RESULTS)"
    )


def fence_set_path(settings: Settings, path: str, label: str = "path") -> Path:
    """A set's folder under the data root: 409 without a root, 400 outside it or not a folder, 404 missing."""
    root = settings.data_root
    if root is None:
        raise HTTPException(409, "no data root is configured (ARGUS_PUPPET_DATA_ROOT): sets given by path are off")
    try:
        target = safe_join(root, path, allow_root=True, label=label)
    except PathRefused as exc:
        raise HTTPException(400, str(exc)) from None
    if not target.exists():
        raise HTTPException(404, f"{path!r} does not exist under the data root")
    if not target.is_dir():
        raise HTTPException(400, f"{label}: {path!r} is not a folder")
    return target


def load_set(settings: Settings, source: PoseSetSource, label: str) -> list[PoseResult]:
    """The poses of a set: inline, or every sidecar under its fenced folder (blocking: call it off the event loop).

    A sidecar resolving outside the data root (a symlink) is skipped; an invalid one is 422, named relative to the
    data root."""
    if source.results is not None:
        if len(source.results) > settings.max_set_results:
            raise _too_many(len(source.results), settings)
        return source.results
    assert source.path is not None  # the model allows exactly one of results and path
    target = fence_set_path(settings, source.path, label=f"{label}.path")
    root = settings.data_root
    assert root is not None  # fence_set_path refused otherwise
    found = [p for p in sidecars.find_sidecars(target, recursive=source.recursive) if is_within(root, p)]
    if len(found) > settings.max_set_results:
        raise _too_many(len(found), settings)
    out = []
    for path in found:
        try:
            out.append(sidecars.load_sidecar(path))
        except sidecars.SidecarError:
            raise HTTPException(422, f"{path.relative_to(root).as_posix()} is not a valid pose sidecar") from None
    return out


def check_render(settings: Settings, req: RenderRequest) -> None:
    """413 for a canvas over the pixel cap or more people than the people cap."""
    pose = req.pose
    if isinstance(pose, BarePose):
        n = len(pose.people())
    else:
        n = (1 if pose.primary is not None else 0) if req.options.primary_only else len(pose.people)
    source = (pose.width, pose.height)
    if n > settings.max_people:
        raise HTTPException(413, f"{n} people is over the cap of {settings.max_people} (ARGUS_PUPPET_MAX_PEOPLE)")
    w, h = render.canvas_size(source, req.options)
    if w * h > settings.max_pixels:
        raise HTTPException(413, f"a {w}x{h} canvas is over the cap of {settings.max_pixels} pixels")


def render_request(req: RenderRequest) -> bytes:
    """The PNG for a checked render request (blocking)."""
    pose = req.pose
    if isinstance(pose, BarePose):
        return render.render_png(pose.people(), req.options, source_size=(pose.width, pose.height))
    return render.render_png(pose, req.options)


def create_app(
    settings: Settings | None = None,
    *,
    estimator: estimator_mod.Estimator | None = None,
    coordinator: GpuCoordinator | None = None,
) -> FastAPI:
    """The app. *estimator* and *coordinator* replace the real ones (tests); both default lazily."""
    settings = settings or Settings.from_env()
    state = AppState(
        settings=settings,
        coordinator=coordinator,
        fixed_estimator=estimator,
        admit=Admission(settings.pose_concurrency),
        compute_admit=Admission(settings.pose_concurrency),
    )
    app = FastAPI(
        title="argus-puppet",
        version=__version__,
        description="Whole-body pose: keypoints, pose metrics and ControlNet pose conditioning.",
    )
    app.state.puppet = state
    install_guards(app, settings)

    @app.get("/health", response_model=Health)
    async def get_health() -> Health:
        """Model files present and verified, the provider and device, free VRAM, read-only mode."""
        return await asyncio.to_thread(health, state)

    @app.post("/pose", response_model=PoseResult)
    async def post_pose(
        request: Request,
        name: str | None = Query(None, description="The image's name in the result (a file name, no path)."),
        face: str | None = Query(None, description="Primary-face hint 'x,y,w,h' in pixels: picks the primary."),
    ) -> PoseResult:
        """Pose one image, from memory. Raw image bytes or a multipart file; nothing is written."""
        face_hint = parse_face_hint(face)
        try:
            priority = gpu.parse_priority(request.headers.get(gpu.PRIORITY_HEADER))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        async with state.admit.slot():
            data, filename = await read_image(request, settings.max_upload_bytes, timeout=settings.body_timeout_s)
            if not data:
                raise HTTPException(400, "no image: the body is empty")
            image_name = _name(name, filename)
            try:
                return await asyncio.to_thread(
                    extract.pose_image,
                    data,
                    name=image_name,
                    settings=settings,
                    estimator=state.fixed_estimator,
                    face_hint=face_hint,
                    priority=priority,
                    coordinator=state.coordinator,
                )
            except TooLarge as exc:
                raise HTTPException(413, str(exc)) from None
            except BadInput as exc:
                raise HTTPException(415 if "not an image" in str(exc) else 400, str(exc)) from None
            except Exception as exc:  # noqa: BLE001 - model_errors maps what it knows and re-raises the rest
                raise model_errors(exc) from None

    @app.post("/pose/scan", response_model=ScanSummary)
    async def post_scan(request: Request) -> ScanSummary:
        """Pose a folder under the data root into ``<stem>.pose.json`` sidecars (the one state-changing route):
        ``ScanRequest``."""
        root = settings.data_root
        if root is None:  # before the body is read
            raise HTTPException(409, "no data root is configured (ARGUS_PUPPET_DATA_ROOT): scanning is off")
        req = await read_json(request, ScanRequest, SCAN_JSON_BYTES, timeout=settings.body_timeout_s)
        try:
            target = safe_join(root, req.path, allow_root=True, label="path")
        except PathRefused as exc:
            raise HTTPException(400, str(exc)) from None
        if not target.exists():
            raise HTTPException(404, f"{req.path!r} does not exist under the data root")
        async with state.scan_admit.slot():
            try:
                return await asyncio.to_thread(
                    extract.scan,
                    target,
                    label=req.path,
                    recursive=req.recursive,
                    overwrite=req.overwrite,
                    settings=settings,
                    estimator=state.fixed_estimator,
                    coordinator=state.coordinator,
                    max_images=settings.max_scan_images,
                    fence=root,
                )
            except Exception as exc:  # noqa: BLE001 - model_errors maps what it knows and re-raises the rest
                raise model_errors(exc) from None

    cap = settings.max_json_bytes

    @app.post("/stats", response_model=PoseSetStats)
    async def post_stats(request: Request) -> PoseSetStats:
        """Pose diversity, clusters and coverage of one set: ``StatsRequest`` (results, or a sidecar folder under the
        data root, and optional ``params``). Writes nothing."""
        async with state.compute_admit.slot():
            req = await read_json(
                request, StatsRequest, cap, timeout=settings.body_timeout_s, precheck=set_precheck(settings, 1)
            )
            results = await asyncio.to_thread(load_set, settings, req, "set")
            return await asyncio.to_thread(metrics.pose_set_stats, results, params=req.params)

    @app.post("/compare", response_model=PoseSetComparison)
    async def post_compare(request: Request) -> PoseSetComparison:
        """An output set's poses against a reference set: ``CompareRequest``. Writes nothing."""
        async with state.compute_admit.slot():
            req = await read_json(
                request, CompareRequest, cap, timeout=settings.body_timeout_s, precheck=set_precheck(settings, 2)
            )
            outputs = await asyncio.to_thread(load_set, settings, req.outputs, "outputs")
            reference = await asyncio.to_thread(load_set, settings, req.reference, "reference")
            threshold = compare.DEFAULT_CLOSE_THRESHOLD if req.threshold is None else req.threshold
            return await asyncio.to_thread(
                compare.compare_pose_sets, outputs, reference, threshold=threshold, params=req.params
            )

    @app.post("/render", response_class=Response, responses={200: {"content": {"image/png": {}}}})
    async def post_render(request: Request) -> Response:
        """An OpenPose-convention control PNG for one pose: ``RenderRequest``. Writes nothing."""
        async with state.compute_admit.slot():
            req = await read_json(request, RenderRequest, min(cap, RENDER_JSON_BYTES), timeout=settings.body_timeout_s)
            check_render(settings, req)
            try:
                png = await asyncio.to_thread(render_request, req)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None
            return Response(content=png, media_type="image/png")

    return app


def __getattr__(name: str) -> Any:
    """``argus_puppet.server.app:app`` (the Dockerfile's uvicorn target): built from the environment on first use."""
    if name == "app":
        built = create_app()
        globals()["app"] = built
        return built
    raise AttributeError(name)
