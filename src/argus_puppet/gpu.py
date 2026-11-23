"""GPU leases from the suite's broker (argus-iris), through argus-cortex's coordinator.

The person detector and the pose model run in puppet's own process on ONNX Runtime CUDA, under one lease of the
kind ``pose`` that ``gpu-tenant.yaml`` declares (``tests/test_gpu.py`` keeps the figures here in step with it).

* On CPU no lease is taken (``nullcontext``).
* A folder run holds **one** lease across the folder at ``batch`` priority, checking ``should_yield()`` between
  images; ``POST /pose`` takes one per call at ``interactive``, or what its ``X-Argus-Priority`` header asks for,
  clamped (puppet's work is never ``realtime``).

The coordinator comes from the host-wide ``ARGUS_GPU_*`` environment (:func:`argus_cortex.gpu.coordinator_from_env`):
unset is ``none``, no gating. A lease that is not granted raises one of :data:`LEASE_ERRORS` (the GPU is busy: retry
later) or :data:`OPERATOR_ERRORS` (the coordinator is misconfigured or the broker refused the request itself: retrying
will not help); neither is a bad image.
"""

from __future__ import annotations

import contextlib
import threading
import weakref
from collections.abc import Callable, Iterator
from typing import Any

from argus_cortex import settings as suite_settings
from argus_cortex.gpu import GpuCoordinator, GpuLease, GpuLeaseRefused, GpuLeaseTimeout, coordinator_from_env
from argus_cortex.gpu_wire import PRIORITY_CLASSES, PriorityClass

from argus_puppet import runtime

TENANT = "argus-puppet"
KIND = "pose"
# gpu-tenant.yaml's figures for the kind: the whole footprint, and the weights that stay loaded after a lease.
VRAM_MB = 1600
MODEL_MB = 240
PRIORITY_HEADER = "X-Argus-Priority"
DEFAULT_PRIORITY: PriorityClass = "interactive"
BATCH_PRIORITY: PriorityClass = "batch"
# Puppet's work is never realtime: a caller asking for it gets interactive, the highest class puppet leases at.
_CLAMP: dict[str, PriorityClass] = {"realtime": "interactive"}

# A lease that was not granted because the GPU is busy: the caller may retry later (503 with Retry-After).
LEASE_ERRORS: tuple[type[Exception], ...] = (GpuLeaseRefused, GpuLeaseTimeout)
# The broker answered but refused the request itself: a 4xx (a missing or wrong ARGUS_GPU_BROKER_TOKEN, a request it
# will not accept) or a payload this client cannot read (an incompatible lease version). cortex raises BrokerError for
# exactly these; an unreachable broker (connection error, timeout, 5xx) falls back instead, and a busy GPU is one of
# LEASE_ERRORS. Retrying will not help: an operator's fix (see OPERATOR_ERRORS).
BROKER_REFUSALS: tuple[type[Exception], ...] = ()
try:  # the broker client is behind cortex's [remote] extra
    from argus_cortex.gpu_broker import BrokerError

    BROKER_REFUSALS = (BrokerError,)
except ImportError:  # pragma: no cover - without [remote] there is no broker to refuse
    pass


def parse_priority(raw: str | None) -> PriorityClass:
    """The priority class a caller asked for (``X-Argus-Priority``); :data:`DEFAULT_PRIORITY` when none.

    Raises :class:`ValueError` for a name that is not a class."""
    value = (raw or "").strip().lower()
    if not value:
        return DEFAULT_PRIORITY
    if value not in PRIORITY_CLASSES:
        raise ValueError(f"{PRIORITY_HEADER} must be one of {', '.join(PRIORITY_CLASSES)}, not {raw!r}")
    return _CLAMP.get(value, value)  # type: ignore[return-value]


_coordinator: GpuCoordinator | None = None
_coordinator_lock = threading.Lock()


class CoordinatorInvalid(RuntimeError):
    """``ARGUS_GPU_*`` names no usable coordinator (``GET /health`` reports ``invalid``): an operator's fix, not a
    busy GPU, so it is not one of :data:`LEASE_ERRORS` (it is one of :data:`OPERATOR_ERRORS`)."""


# Every lease failure that is the operator's to fix, not a busy GPU: a 503 without Retry-After (operator_message).
OPERATOR_ERRORS: tuple[type[Exception], ...] = (CoordinatorInvalid, *BROKER_REFUSALS)


def operator_message(exc: Exception) -> str:
    """What to tell a caller about one of :data:`OPERATOR_ERRORS` (the broker token is never in it)."""
    if isinstance(exc, CoordinatorInvalid):
        return str(exc)
    detail = str(exc)
    # Read where cortex's coordinator reads it: the environment, then the suite file (argus_cortex.settings).
    token = suite_settings.get("ARGUS_GPU_BROKER_TOKEN")
    if token:
        detail = detail.replace(token, "***")
    return (
        "the GPU broker refused the lease request; this is a configuration problem, not a busy GPU (check "
        f"ARGUS_GPU_BROKER_TOKEN and ARGUS_GPU_BROKER_URL, and that argus-iris and argus-cortex agree): {detail}"
    )


def default_coordinator() -> GpuCoordinator:
    """The host-wide coordinator from ``ARGUS_GPU_*``, built once. Raises :class:`CoordinatorInvalid`."""
    global _coordinator
    with _coordinator_lock:
        if _coordinator is None:
            try:
                _coordinator = coordinator_from_env()
            except ValueError as exc:
                raise CoordinatorInvalid(
                    f"the GPU coordinator is misconfigured (ARGUS_GPU_COORDINATOR): {exc}"
                ) from exc
        return _coordinator


def lease(
    device: str,
    *,
    priority: PriorityClass,
    coordinator: GpuCoordinator | None = None,
    caller: str = "run",
) -> contextlib.AbstractContextManager[GpuLease | None]:
    """Hold a GPU lease for pose work on *device*; ``nullcontext(None)`` when that work runs on the CPU (including
    ``auto`` or ``cuda`` once CUDA proved unusable: :func:`argus_puppet.runtime.uses_cuda`)."""
    if not runtime.uses_cuda(device):
        return contextlib.nullcontext(None)
    coord = coordinator or default_coordinator()
    return _noting_via(
        coord,
        coord.lease(
            caller=f"{TENANT}:{caller}",
            min_vram_mb=VRAM_MB,
            model_mb=MODEL_MB,
            kind=KIND,
            priority=priority,
        ),
    )


# How each coordinator granted its last lease (GpuLease.via): "local", "broker", or "fallback" (the broker could not be
# reached and its fallback coordinator granted it). GET /health reports it as last_lease_via. Keyed by id() with a weak
# reference to check it is still that coordinator (coordinators need not be hashable); a process holds one or two.
_last_via: dict[int, tuple[Callable[[], Any], str]] = {}


def _ref(coord: GpuCoordinator) -> Callable[[], Any]:
    try:
        return weakref.ref(coord)
    except TypeError:  # cannot be weakly referenced: hold it (there is one per process)
        return lambda: coord


@contextlib.contextmanager
def _noting_via(coord: GpuCoordinator, held: contextlib.AbstractContextManager[GpuLease]) -> Iterator[GpuLease]:
    with held as handle:
        via = getattr(handle, "via", None)
        if isinstance(via, str):
            _last_via[id(coord)] = (_ref(coord), via)
        yield handle


def last_via(coordinator: GpuCoordinator) -> str | None:
    """How *coordinator* granted its last lease (``local``, ``broker`` or ``fallback``); ``None`` before any."""
    entry = _last_via.get(id(coordinator))
    return entry[1] if entry is not None and entry[0]() is coordinator else None


def should_yield(handle: Any) -> bool:
    """Whether a held lease asks for the GPU back (``False`` for no lease, or one that cannot ask)."""
    check = getattr(handle, "should_yield", None)
    return bool(check()) if callable(check) else False


def report_progress(handle: Any, done: int, total: int) -> None:
    """Tell the broker how far a folder run has got, when the lease can carry it."""
    report = getattr(handle, "report_progress", None)
    if callable(report):
        report(done, total, stage=f"image {done}/{total}")


class FolderLease:
    """One lease held across a folder run, given back and retaken when the broker asks for it (``should_yield``)."""

    def __init__(self, device: str, *, priority: PriorityClass, coordinator: GpuCoordinator | None, caller: str):
        self._args: dict[str, Any] = {"priority": priority, "coordinator": coordinator, "caller": caller}
        self._device = device
        self._stack = contextlib.ExitStack()
        self.handle: GpuLease | None = None
        self.taken = 0

    def _take(self) -> None:
        self.handle = self._stack.enter_context(lease(self._device, **self._args))
        self.taken += 1

    def checkpoint(self) -> None:
        """Between images: if the broker asked for the GPU back, release it and queue for it again."""
        if should_yield(self.handle):
            self._stack.close()
            self._stack = contextlib.ExitStack()
            self._take()

    def __enter__(self) -> FolderLease:
        self._take()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stack.close()


@contextlib.contextmanager
def folder_lease(
    device: str, *, priority: PriorityClass = BATCH_PRIORITY, coordinator: GpuCoordinator | None = None
) -> Iterator[FolderLease]:
    """:class:`FolderLease` as a context manager (``batch`` priority by default)."""
    with FolderLease(device, priority=priority, coordinator=coordinator, caller="folder") as held:
        yield held
