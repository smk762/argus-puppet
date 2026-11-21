"""The argus-puppet HTTP service (``:8110``): pose one uploaded image, or a folder under the data root.

Routes:

* ``GET /health`` -> :class:`~argus_puppet.models.Health`.
* ``POST /pose`` -> :class:`~argus_puppet.models.PoseResult` for one image (raw bytes or a multipart file). Writes
  nothing. ``?name=`` names the image in the result; ``?face=x,y,w,h`` is a primary-face hint. Concurrency is
  bounded (``ARGUS_PUPPET_POSE_CONCURRENCY``): a call past it is answered 503 + Retry-After, not queued.
* ``POST /pose/scan`` ``{path, recursive, overwrite}`` -> :class:`~argus_puppet.extract.ScanSummary`. Writes
  sidecars under ``ARGUS_PUPPET_DATA_ROOT``; *path* is fenced under it with :func:`argus_cortex.fs.safe_join`.

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
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from argus_cortex.fs import PathRefused, safe_basename, safe_join
from argus_cortex.gpu import GpuCoordinator, free_vram_mb
from argus_cortex.limits import BadInput, TooLarge, read_body, read_upload
from argus_cortex.server import WriteGuard, constant_refuse, cross_site_refuse
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from argus_puppet import __version__, extract, gpu, registry, runtime
from argus_puppet import estimator as estimator_mod
from argus_puppet.config import Settings
from argus_puppet.models import Health, PoseResult

READ_ONLY_REFUSAL = "argus-puppet is read-only (ARGUS_PUPPET_READ_ONLY): state-changing requests are refused"
# A multipart body's Content-Length counts boundaries and part headers too: the declared-length pre-check allows this
# much over the cap, and read_upload then holds the file itself to the exact cap.
MULTIPART_OVERHEAD = 64 * 1024
RAW_TYPES = ("application/octet-stream", "image/")
BUSY_RETRY_S = 1
GPU_RETRY_S = 5


class ScanRequest(BaseModel):
    """``POST /pose/scan``: a folder under the data root to pose into sidecars."""

    path: str = Field(..., description="Folder (or one image) relative to ARGUS_PUPPET_DATA_ROOT.")
    recursive: bool = Field(False, description="Include subfolders.")
    overwrite: bool = Field(False, description="Re-pose images that already have a sidecar.")


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
    """What routes share; ``app.state.puppet``. Later routes (``/stats``, ``/compare``, ``/render``) use it too."""

    settings: Settings
    coordinator: GpuCoordinator | None = None
    fixed_estimator: estimator_mod.Estimator | None = None
    admit: Admission = field(default_factory=lambda: Admission(1))
    scan_admit: Admission = field(default_factory=lambda: Admission(1))

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


async def read_image(request: Request, cap: int) -> tuple[bytes, str | None]:
    """The image of a ``POST``: raw bytes or the first file of a multipart form, at most *cap* bytes. Returns the
    bytes and the multipart filename, if any. 413 over the cap, 400 malformed, 415 an unsupported content type."""
    content_type = (request.headers.get("content-type") or "").lower()
    try:
        if content_type.startswith("multipart/form-data"):
            declared = (request.headers.get("content-length") or "").strip()
            if declared and not (declared.isascii() and declared.isdigit()):
                raise BadInput("Content-Length is not a number")
            if declared and int(declared) > cap + MULTIPART_OVERHEAD:
                raise TooLarge("image", cap)
            async with request.form() as form:
                upload = next((v for v in form.values() if hasattr(v, "read")), None)
                if upload is None:
                    raise HTTPException(400, "no image file in the form")
                return await read_upload(upload, cap, what="image"), getattr(upload, "filename", None)
        if content_type and not content_type.startswith(RAW_TYPES):
            raise HTTPException(415, "send the image as raw bytes (application/octet-stream or image/*) or multipart")
        return await read_body(request, cap, what="image"), None
    except TooLarge as exc:
        raise HTTPException(413, str(exc)) from None
    except BadInput as exc:
        raise HTTPException(400, str(exc)) from None


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
    return Health(
        version=__version__,
        read_only=settings.read_only,
        models={n: runtime.model_verified(settings, n) for n in registry.names()},
        provider=getattr(est, "provider", None),
        device="cuda" if cuda else "cpu",
        free_vram_mb=free_vram_mb(0) if cuda else None,
    )


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
            data, filename = await read_image(request, settings.max_upload_bytes)
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

    @app.post("/pose/scan", response_model=extract.ScanSummary)
    async def post_scan(req: ScanRequest) -> extract.ScanSummary:
        """Pose a folder under the data root into ``<stem>.pose.json`` sidecars (the one state-changing route)."""
        root = settings.data_root
        if root is None:
            raise HTTPException(409, "no data root is configured (ARGUS_PUPPET_DATA_ROOT): scanning is off")
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

    return app


def __getattr__(name: str) -> Any:
    """``argus_puppet.server.app:app`` (the Dockerfile's uvicorn target): built from the environment on first use."""
    if name == "app":
        built = create_app()
        globals()["app"] = built
        return built
    raise AttributeError(name)
