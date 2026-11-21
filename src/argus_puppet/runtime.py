"""ONNX Runtime plumbing: execution providers, the CUDA arena cap, verified and cached sessions.

onnxruntime is an optional extra (``[cpu]`` or ``[gpu]``): nothing here imports it at module level, so
``import argus_puppet`` works without it and a missing install surfaces as :class:`ModelUnavailable`.

Every model file is checked against its registry pin (size, then SHA-256) before its first load: a file whose bytes
are not the pinned ones is refused, never run. Hashes are memoised by path, mtime and size, so a 134 MB file is
hashed once per process (and again only if it changes on disk).
"""

from __future__ import annotations

import contextlib
import hashlib
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import structlog

from argus_puppet import registry
from argus_puppet.config import Settings

log = structlog.get_logger(__name__)

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
FETCH_HINT = "Run `argus-puppet models fetch` to download the pinned models."
INSTALL_HINT = "Install an ONNX Runtime: pip install 'argus-puppet[cpu]' (or '[gpu]' for CUDA)."

# ONNX Runtime's CUDA arena, by default, grows in power-of-two steps with no ceiling and never hands memory back:
# in a sibling service it once grew one process to ~20 GB. Every CUDA session here grows only by what an
# allocation asks, up to this many MB (Settings.gpu_mem_limit_mb overrides it).
DEFAULT_GPU_MEM_LIMIT_MB = 1024


class ModelUnavailable(RuntimeError):
    """A model cannot be run: onnxruntime is missing, or the file is missing or not the pinned bytes.

    The message ends with what to do about it (``hint``)."""

    def __init__(self, message: str, hint: str = FETCH_HINT) -> None:
        super().__init__(f"{message} {hint}".strip())
        self.hint = hint


def onnxruntime_available() -> bool:
    """Whether onnxruntime can be imported."""
    try:
        import onnxruntime  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _ort() -> Any:
    try:
        import onnxruntime as ort  # noqa: PLC0415 - optional extra
    except ImportError as exc:
        raise ModelUnavailable("onnxruntime is not installed.", INSTALL_HINT) from exc
    return ort


_cuda_preloaded = False
_cuda_lock = threading.Lock()


def _preload_cuda(ort: Any) -> None:
    """Load CUDA/cuDNN from NVIDIA's pip wheels once (onnxruntime-gpu[cuda,cudnn] puts them off the loader path).

    A no-op when they come from the system, or with an onnxruntime too old to have ``preload_dlls``."""
    global _cuda_preloaded
    with _cuda_lock:
        if _cuda_preloaded:
            return
        _cuda_preloaded = True
        preload = getattr(ort, "preload_dlls", None)
        if preload is not None:
            # Deliberate: a failure leaves system CUDA, if any, to serve the provider.
            with contextlib.suppress(Exception):
                preload()


def providers(device: str) -> list[str]:
    """Execution provider names for a device hint (``auto`` | ``cpu`` | ``cuda``).

    CUDA is named only when onnxruntime really has ``CUDAExecutionProvider``: ``cuda`` on a CPU-only install falls
    back to CPU (with a warning) rather than taking a GPU lease for work that runs on the CPU."""
    dev = (device or "auto").lower()
    if dev == "cpu":
        return [CPU]
    try:
        ort = _ort()
        available = set(ort.get_available_providers())
    except ModelUnavailable:
        return [CPU]
    if CUDA in available:
        _preload_cuda(ort)
        return [CUDA, CPU]
    if dev == "cuda":
        log.warning("cuda_unavailable", detail="onnxruntime has no CUDAExecutionProvider; running on CPU")
    return [CPU]


def uses_cuda(device: str) -> bool:
    """Whether work for *device* runs on CUDA (and so needs a GPU lease)."""
    return CUDA in providers(device)


def cuda_options(gpu_mem_limit_mb: int | None) -> dict[str, str]:
    """The CUDA provider options every session is built with: a capped arena that grows only as asked."""
    mb = gpu_mem_limit_mb or DEFAULT_GPU_MEM_LIMIT_MB
    return {"arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": str(mb * 1024 * 1024)}


def session_providers(names: Sequence[str], gpu_mem_limit_mb: int | None) -> list[str | tuple[str, dict[str, str]]]:
    """*names* as onnxruntime takes them, CUDA with its options."""
    return [(n, cuda_options(gpu_mem_limit_mb)) if n == CUDA else n for n in names]


# ── verification ─────────────────────────────────────────────────────────────────────────────────────────────────

_hash_cache: dict[tuple[str, int, int], str] = {}
_hash_lock = threading.Lock()


def file_sha256(path: Path) -> str:
    """SHA-256 of *path* (lower-case hex), memoised by path, mtime and size."""
    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size)
    with _hash_lock:
        hit = _hash_cache.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    with _hash_lock:
        _hash_cache[key] = digest
    return digest


def check_model(settings: Settings, name: str) -> Path:
    """The path of pinned model *name*, after checking it is there and is the pinned bytes.

    Raises :class:`ModelUnavailable` when it is missing, the wrong size, or hashes to something else."""
    pinned = registry.get(name)
    path = settings.model_path(name)
    if not path.is_file():
        raise ModelUnavailable(f"Model {name} not found at {path}.")
    size = path.stat().st_size
    if size != pinned.bytes:
        raise ModelUnavailable(f"Model {name} at {path} is {size} bytes, not the pinned {pinned.bytes}.")
    digest = file_sha256(path)
    if digest != pinned.sha256:
        raise ModelUnavailable(f"Model {name} at {path} does not match its pinned SHA-256 (got {digest[:12]}...).")
    return path


def model_verified(settings: Settings, name: str) -> bool:
    """Whether pinned model *name* is present and verified (never raises for a bad file)."""
    try:
        check_model(settings, name)
    except (ModelUnavailable, OSError):
        return False
    return True


# ── sessions ─────────────────────────────────────────────────────────────────────────────────────────────────────

_sessions: dict[tuple[str, tuple[str, ...], int | None], Any] = {}
_session_lock = threading.Lock()


def load_session(settings: Settings, name: str, device: str | None = None) -> Any:
    """A cached ``onnxruntime.InferenceSession`` for pinned model *name*, verified before its first load.

    Raises :class:`ModelUnavailable` (onnxruntime missing, file missing or not the pinned bytes)."""
    names = tuple(providers(device or settings.device))
    path = settings.model_path(name)
    key = (str(path), names, settings.gpu_mem_limit_mb)
    with _session_lock:
        hit = _sessions.get(key)
        if hit is not None:
            return hit
        ort = _ort()
        check_model(settings, name)
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only: ORT's warnings about unused initialisers are noise
        try:
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=session_providers(names, settings.gpu_mem_limit_mb)
            )
        except Exception as exc:  # noqa: BLE001 - any ORT load failure is "cannot run this model"
            raise ModelUnavailable(f"Model {name} at {path} could not be loaded: {exc}.") from exc
        log.info("session_loaded", model=name, providers=session.get_providers())
        _sessions[key] = session
        return session


def clear_sessions() -> None:
    """Drop every cached session (tests; freeing memory)."""
    with _session_lock:
        _sessions.clear()
