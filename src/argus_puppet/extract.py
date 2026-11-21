"""The library API: pose one image, or a folder of images into ``<stem>.pose.json`` sidecars.

* :func:`pose_image` takes bytes or a path and returns a :class:`~argus_puppet.models.PoseResult`. It writes nothing.
  Every person carries ``labels`` (:func:`argus_puppet.labels.pose_labels` at the default joint score).
* :func:`pose_folder` walks a folder (or one file) in sorted order and yields one
  :class:`~argus_puppet.models.ImageOutcome` per image,
  writing each sidecar atomically (temp file + ``os.replace``, sorted keys). An existing sidecar is kept unless
  ``overwrite`` (a sidecar written by an older version, e.g. a puppet_version 1.0 one without ``PersonPose.labels``,
  is kept as it is: pass ``overwrite`` to re-pose it); two images sharing a stem (``a.jpg`` + ``a.png``) would share
  a sidecar, so only the first in sorted order is posed and the other is skipped (``sidecar_collision``). An
  unreadable image, or a sidecar that cannot be written (``unwritable``: an image in a folder that cannot take a file
  is not posed at all), is reported, never fatal; a model that cannot run or a GPU lease that is not granted stops
  the run (raised).
* :func:`scan` runs :func:`pose_folder` to the end and returns a :class:`~argus_puppet.models.ScanSummary`.

Images are decoded through :func:`argus_cortex.limits.open_image` (the pixel and byte caps from settings), with
EXIF orientation applied: keypoints are in the oriented image's pixels.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog
from argus_cortex.fs import is_within
from argus_cortex.gpu import GpuCoordinator
from argus_cortex.gpu_wire import PriorityClass
from argus_cortex.limits import BadInput, TooLarge, open_image
from PIL import Image, ImageOps

from argus_puppet import estimator as estimator_mod
from argus_puppet import gpu
from argus_puppet.config import Settings
from argus_puppet.estimator import to_rgb
from argus_puppet.labels import pose_labels
from argus_puppet.models import SKIP_MAX_IMAGES, ImageOutcome, PoseResult, ScanSummary, sidecar_path
from argus_puppet.primary import BoxXYWH, assign_primary

log = structlog.get_logger(__name__)

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

__all__ = [
    "IMAGE_SUFFIXES",
    "ImageOutcome",
    "ScanSummary",
    "build_result",
    "decode",
    "list_images",
    "pose_folder",
    "pose_image",
    "scan",
    "to_rgb",
    "write_sidecar",
]

# Why an image was skipped or failed (ImageOutcome.reason).
SKIP_EXISTS = "exists"
SKIP_COLLISION = "sidecar_collision"
SKIP_LIMIT = SKIP_MAX_IMAGES
SKIP_OUTSIDE = "outside_root"
FAIL_UNREADABLE = "unreadable"
FAIL_TOO_LARGE = "too_large"
FAIL_UNWRITABLE = "unwritable"


# ── one image ────────────────────────────────────────────────────────────────────────────────────────────────────


def decode(data: bytes, settings: Settings) -> Image.Image:
    """*data* as an RGB image with EXIF orientation applied, within the settings' caps.

    Raises :class:`~argus_cortex.limits.TooLarge` or :class:`~argus_cortex.limits.BadInput`."""
    img = open_image(data, max_pixels=settings.max_pixels, max_bytes=settings.max_upload_bytes)
    try:
        img.load()
        oriented = ImageOps.exif_transpose(img)
        return to_rgb(oriented)
    except (OSError, ValueError, SyntaxError) as exc:  # truncated or corrupt past the header
        raise BadInput(f"the image could not be decoded: {exc}") from exc
    except Image.DecompressionBombError as exc:  # pragma: no cover - open_image's pixel cap comes first
        raise TooLarge("image", settings.max_pixels, "pixels") from exc


def _area(box: BoxXYWH) -> float:
    return float(box[2]) * float(box[3])


def build_result(
    image: Image.Image,
    *,
    name: str,
    sha256: str,
    estimator: estimator_mod.Estimator,
    face_hint: BoxXYWH | None = None,
) -> PoseResult:
    """Run *estimator* on a decoded *image* and wrap the people (largest first, one primary, each labelled with
    :func:`~argus_puppet.labels.pose_labels`) in a PoseResult."""
    people = sorted(estimator.estimate(image), key=lambda p: -_area(p.bbox))  # stable: ties keep detector order
    people, primary = assign_primary(people, face_hint=face_hint)
    people = [p.model_copy(update={"labels": pose_labels(p.keypoints_array())}) for p in people]
    return PoseResult(
        image=name,
        image_sha256=sha256,
        width=image.width,
        height=image.height,
        people=people,
        primary=primary,
        estimator=estimator.provenance(),
    )


def _resolve_estimator(
    settings: Settings, estimator: estimator_mod.Estimator | None, device: str | None
) -> estimator_mod.Estimator:
    return estimator if estimator is not None else estimator_mod.get_estimator(settings, device=device)


def pose_image(
    image: bytes | str | os.PathLike[str],
    *,
    name: str | None = None,
    settings: Settings | None = None,
    estimator: estimator_mod.Estimator | None = None,
    face_hint: BoxXYWH | None = None,
    device: str | None = None,
    priority: PriorityClass = gpu.DEFAULT_PRIORITY,
    coordinator: GpuCoordinator | None = None,
) -> PoseResult:
    """Pose one image (bytes or a path). *name* becomes ``PoseResult.image`` (default: the file's name, or
    ``"upload"`` for bytes). On CUDA a GPU lease is held at *priority* for the call.

    Raises :class:`~argus_cortex.limits.BadInput` / :class:`~argus_cortex.limits.TooLarge` for a bad image,
    :class:`~argus_puppet.runtime.ModelUnavailable`, or a lease error (:data:`argus_puppet.gpu.LEASE_ERRORS`)."""
    settings = settings or Settings.from_env()
    if isinstance(image, bytes):
        data, default_name = image, "upload"
    else:
        path = Path(image)
        data, default_name = path.read_bytes(), path.name
    decoded = decode(data, settings)
    digest = hashlib.sha256(data).hexdigest()
    with gpu.lease(device or settings.device, priority=priority, coordinator=coordinator, caller="image"):
        est = _resolve_estimator(settings, estimator, device)
        return build_result(decoded, name=name or default_name, sha256=digest, estimator=est, face_hint=face_hint)


# ── a folder ─────────────────────────────────────────────────────────────────────────────────────────────────────


def list_images(root: Path, *, recursive: bool = False) -> list[Path]:
    """Image files under *root* (or *root* itself, if it is one), sorted; hidden names are left out."""
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    candidates = root.rglob("*") if recursive else root.iterdir()
    found = []
    for p in candidates:
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file():
            found.append(p)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def write_sidecar(result: PoseResult, path: Path) -> None:
    """Write *result* to *path* atomically (a temp file beside it, then ``os.replace``), keys sorted."""
    text = json.dumps(result.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):  # best effort: the temp file may already be gone
            os.unlink(tmp)
        raise


@dataclass
class _Plan:
    path: Path
    rel: str
    skip: str | None = None


def _plan(root: Path, images: Sequence[Path], *, overwrite: bool, fence: Path | None) -> list[_Plan]:
    base = root.parent if root.is_file() else root
    claimed: set[Path] = set()
    plans = []
    for p in images:
        plan = _Plan(p, p.relative_to(base).as_posix())
        sidecar = sidecar_path(p)
        if fence is not None and not is_within(fence, p):
            plan.skip = SKIP_OUTSIDE  # a symlink pointing out of the root it was asked to stay under
        elif sidecar in claimed:
            plan.skip = SKIP_COLLISION
        elif sidecar.exists() and not overwrite:
            plan.skip = SKIP_EXISTS
        claimed.add(sidecar)
        plans.append(plan)
    return plans


def pose_folder(
    root: str | os.PathLike[str],
    *,
    recursive: bool = False,
    overwrite: bool = False,
    settings: Settings | None = None,
    estimator: estimator_mod.Estimator | None = None,
    device: str | None = None,
    priority: PriorityClass = gpu.BATCH_PRIORITY,
    coordinator: GpuCoordinator | None = None,
    max_images: int | None = None,
    fence: Path | None = None,
) -> Iterator[ImageOutcome]:
    """Pose every image under *root* (a folder, or one image file) into sidecars, yielding an outcome per image.

    One GPU lease (on CUDA) is held across the run and handed back between images when the broker asks. At most
    *max_images* images go through the estimator; the rest are skipped (``max_images``). With *fence*, an image
    that resolves outside it (a symlink) is skipped (``outside_root``)."""
    settings = settings or Settings.from_env()
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    plans = _plan(root, list_images(root, recursive=recursive), overwrite=overwrite, fence=fence)
    todo = sum(p.skip is None for p in plans)
    if todo == 0:
        for plan in plans:
            yield ImageOutcome(image=plan.rel, status="skipped", reason=plan.skip)
        return
    run = 0
    with gpu.folder_lease(device or settings.device, priority=priority, coordinator=coordinator) as held:
        est = _resolve_estimator(settings, estimator, device)
        for plan in plans:
            if plan.skip is None and max_images is not None and run >= max_images:
                plan.skip = SKIP_LIMIT
            if plan.skip is not None:
                yield ImageOutcome(image=plan.rel, status="skipped", reason=plan.skip)
                continue
            if run:
                held.checkpoint()
            run += 1
            yield _pose_one(plan, settings=settings, estimator=est, base=root if root.is_dir() else root.parent)
            gpu.report_progress(held.handle, run, min(todo, max_images or todo))


def _pose_one(plan: _Plan, *, settings: Settings, estimator: estimator_mod.Estimator, base: Path) -> ImageOutcome:
    sidecar = sidecar_path(plan.path)
    if not os.access(sidecar.parent, os.W_OK):  # checked first: no point posing what cannot be kept
        detail = f"cannot write to {sidecar.parent.relative_to(base).as_posix() or '.'}: the folder is not writable"
        log.warning("sidecar_unwritable", image=plan.rel, error=detail)
        return ImageOutcome(image=plan.rel, status="failed", reason=FAIL_UNWRITABLE, detail=detail)
    try:
        data = plan.path.read_bytes()
        image = decode(data, settings)
    except TooLarge as exc:
        return ImageOutcome(image=plan.rel, status="failed", reason=FAIL_TOO_LARGE, detail=str(exc))
    except (BadInput, OSError) as exc:
        log.warning("image_unreadable", image=plan.rel, error=str(exc))
        return ImageOutcome(image=plan.rel, status="failed", reason=FAIL_UNREADABLE, detail=str(exc))
    result = build_result(image, name=plan.rel, sha256=hashlib.sha256(data).hexdigest(), estimator=estimator)
    try:
        write_sidecar(result, sidecar)
    except OSError as exc:  # a directory in its place, a full disk, a read-only mount
        log.warning("sidecar_unwritable", image=plan.rel, error=str(exc))
        detail = f"the sidecar could not be written: {exc.strerror or type(exc).__name__}"
        return ImageOutcome(image=plan.rel, status="failed", reason=FAIL_UNWRITABLE, detail=detail)
    return ImageOutcome(
        image=plan.rel, status="written", people=len(result.people), sidecar=sidecar.relative_to(base).as_posix()
    )


def scan(root: str | os.PathLike[str], *, label: str | None = None, **kwargs: object) -> ScanSummary:
    """:func:`pose_folder` run to the end, summarised. *label* names the root in the summary (default: as given)."""
    outcomes = list(pose_folder(root, **kwargs))  # type: ignore[arg-type]
    return ScanSummary.of(label if label is not None else os.fspath(root), outcomes)
