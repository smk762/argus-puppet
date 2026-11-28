"""ONNX Runtime plumbing: execution providers, the CUDA arena cap, verified and cached sessions.

onnxruntime is an optional extra (``[cpu]`` or ``[gpu]``): nothing here imports it at module level, so
``import argus_puppet`` works without it and a missing install surfaces as :class:`ModelUnavailable`.

Every model file is checked against its registry pin (size, then SHA-256) before its first load: a file whose bytes
are not the pinned ones is refused, never run. A load reads the file once, hashes those bytes and hands the same
bytes to ONNX Runtime, so nothing can change between the check and the load, and no memo vouches for it. (The 134 MB
pose model is then held twice for a moment, as Python bytes and as ORT parses them; the bytes are dropped once the
session is built.) ``GET /health`` and ``models verify`` only ask whether a file verifies: their hashes are memoised
by path, device, inode, size, mtime and ctime, so a file is hashed once per process and again after any change. A file
changed in the last 100 ms is not memoised: its time stamps can't yet tell it from a rewrite in the same clock tick.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import sys
import threading
import time
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


# ── which device work really runs on ─────────────────────────────────────────────────────────────────────────────
#
# ``ort.get_available_providers()`` only says CUDA is compiled in. With no driver (a container started without
# ``--gpus``) or no visible card, ONNX Runtime logs "Failed to create CUDAExecutionProvider" and quietly builds the
# session on the CPU. So CUDA counts as usable only when the provider is compiled in, the driver loads and sees a card
# (:func:`_probe_cuda_driver`, a few ctypes calls: no context, no VRAM), and no session has come back without it
# (:func:`load_session` checks ``session.get_providers()``). The first reason found is kept for the life of the
# process, logged once as ``cuda_unavailable`` and reported by ``GET /health``: from then on ``auto`` runs on the CPU
# and takes no GPU lease, and ``cuda`` (asked for explicitly) is refused with :class:`CudaUnavailable`.

_cuda_problem: str | None = None  # why CUDA cannot be used here, once known
_cuda_probed = False  # whether the driver probe has run (it runs once per process)
_cuda_state_lock = threading.RLock()


class CudaUnavailable(ModelUnavailable):
    """CUDA was asked for explicitly (``ARGUS_PUPPET_DEVICE=cuda`` / ``--device cuda``) and cannot be had: nothing is
    run on the CPU in its place."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"CUDA was asked for (device 'cuda') but cannot be used: {reason}{'' if reason.endswith(('.', '?')) else '.'}",
            "Run with a GPU (docker: --gpus all), or set ARGUS_PUPPET_DEVICE=auto (or cpu) to run on the CPU.",
        )
        self.reason = reason


def _probe_cuda_driver() -> str | None:
    """``None`` when the CUDA driver loads and sees at least one device; otherwise why not.

    Asks the driver API directly through ctypes (``cuInit``, ``cuDeviceGetCount``): cheap, creates no context and
    holds no VRAM. Off Linux it trusts ONNX Runtime and leaves the session check to catch a fallback."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        return f"the CUDA driver (libcuda.so.1) cannot be loaded ({exc}); is the GPU passed to the container?"
    rc = lib.cuInit(0)
    if rc != 0:
        return f"the CUDA driver did not initialise (cuInit error {rc})"
    count = ctypes.c_int(0)
    rc = lib.cuDeviceGetCount(ctypes.byref(count))
    if rc != 0 or count.value < 1:
        return "the CUDA driver sees no GPU" if rc == 0 else f"cuDeviceGetCount failed (error {rc})"
    return None


def note_cuda_unavailable(reason: str) -> None:
    """Record that CUDA cannot be used here (the first reason wins) and log ``cuda_unavailable`` once."""
    global _cuda_problem
    with _cuda_state_lock:
        if _cuda_problem is not None:
            return
        _cuda_problem = reason
    log.warning(
        "cuda_unavailable",
        reason=reason,
        detail="device auto runs on the CPU with no GPU lease; device cuda is refused",
    )


def cuda_problem() -> str | None:
    """Why CUDA cannot be used in this process, or ``None`` while it can (as far as anything has shown)."""
    global _cuda_probed
    with _cuda_state_lock:
        if _cuda_problem is not None:
            return _cuda_problem
        try:
            ort = _ort()
        except ModelUnavailable:
            return "onnxruntime is not installed"
        if CUDA not in set(ort.get_available_providers()):
            reason = "this onnxruntime has no CUDAExecutionProvider (a CPU-only install)"
        elif not _cuda_probed:
            _cuda_probed = True
            reason = _probe_cuda_driver()
            if reason is None:
                _preload_cuda(ort)
                return None
        else:
            return None
        # Recorded before the lock is let go: no other thread may see "probed, no problem" for a failed probe.
        note_cuda_unavailable(reason)
    return reason


def wants_cuda(device: str) -> bool:
    """Whether a device hint asks for CUDA at all (``auto`` and ``cuda`` do; ``cpu`` does not)."""
    return (device or "auto").lower() != "cpu"


def requires_cuda(device: str) -> bool:
    """Whether a device hint insists on CUDA (``cuda``): it is refused, never run on the CPU, when CUDA is missing."""
    return (device or "auto").lower() == "cuda"


def providers(device: str) -> list[str]:
    """Execution provider names for a device hint (``auto`` | ``cpu`` | ``cuda``): what its work really runs on.

    CUDA is named only while it is usable (:func:`cuda_problem`); otherwise the CPU (the warning is logged once).
    For ``auto`` that is the fallback; ``cuda`` work is refused instead (:func:`load_session` raises
    :class:`CudaUnavailable`), so a GPU lease is never taken for work that would run on the CPU."""
    if not wants_cuda(device):
        return [CPU]
    return [CUDA, CPU] if cuda_problem() is None else [CPU]


def uses_cuda(device: str) -> bool:
    """Whether work for *device* runs on CUDA (and so needs a GPU lease): never once CUDA is known to be unusable."""
    return CUDA in providers(device)


def effective_device(device: str) -> str:
    """``cuda`` or ``cpu``: where work for *device* runs. ``cuda`` asked for but unusable stays ``cuda`` (its work is
    refused, never moved to the CPU); :func:`cuda_problem` says why."""
    if not wants_cuda(device):
        return "cpu"
    return "cuda" if requires_cuda(device) or cuda_problem() is None else "cpu"


def cuda_options(gpu_mem_limit_mb: int | None) -> dict[str, str]:
    """The CUDA provider options every session is built with: a capped arena that grows only as asked."""
    mb = gpu_mem_limit_mb or DEFAULT_GPU_MEM_LIMIT_MB
    return {"arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": str(mb * 1024 * 1024)}


def session_providers(names: Sequence[str], gpu_mem_limit_mb: int | None) -> list[str | tuple[str, dict[str, str]]]:
    """*names* as onnxruntime takes them, CUDA with its options."""
    return [(n, cuda_options(gpu_mem_limit_mb)) if n == CUDA else n for n in names]


# ── verification ─────────────────────────────────────────────────────────────────────────────────────────────────

_hash_cache: dict[tuple[str, int, int, int, int, int], str] = {}
_hash_lock = threading.Lock()


def _stat_key(path: Path) -> tuple[str, int, int, int, int, int]:
    """What the hash memo is keyed on. mtime alone can be put back (``os.utime``) after a rewrite; the ctime cannot be
    set, and a file moved into place has another inode."""
    st = path.stat()
    return (str(path.resolve()), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


# A file changed this recently is hashed but not memoised. File time stamps come from a coarse clock (a kernel tick,
# not the nanoseconds they show), so a second same-size write within the same tick keeps every part of the key, and a
# memo taken between the two would vouch for bytes that are gone. Git skips its "racily clean" entries the same way.
_RACY_NS = 100_000_000


def file_sha256(path: Path) -> str:
    """SHA-256 of *path* (lower-case hex), memoised by its :func:`_stat_key` (for checks that load nothing)."""
    key = _stat_key(path)
    with _hash_lock:
        hit = _hash_cache.get(key)
    if hit is not None:
        return hit
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if time.time_ns() - max(key[4], key[5]) >= _RACY_NS:
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


def read_model(settings: Settings, name: str) -> bytes:
    """The bytes of pinned model *name*, read once and verified (size, then SHA-256 of these very bytes).

    Raises :class:`ModelUnavailable` when it is missing, unreadable, the wrong size, or hashes to something else."""
    pinned = registry.get(name)
    path = settings.model_path(name)
    if not path.is_file():
        raise ModelUnavailable(f"Model {name} not found at {path}.")
    try:
        with path.open("rb") as f:
            data = f.read(pinned.bytes + 1)
    except OSError as exc:
        raise ModelUnavailable(f"Model {name} at {path} could not be read: {exc.strerror or exc}.") from exc
    if len(data) != pinned.bytes:
        raise ModelUnavailable(f"Model {name} at {path} is not the pinned {pinned.bytes} bytes.")
    digest = hashlib.sha256(data).hexdigest()
    if digest != pinned.sha256:
        raise ModelUnavailable(f"Model {name} at {path} does not match its pinned SHA-256 (got {digest[:12]}...).")
    return data


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
    """A cached ``onnxruntime.InferenceSession`` for pinned model *name*, built from the bytes :func:`read_model`
    verified (the provider list and the CUDA arena options apply as for a path).

    Raises :class:`ModelUnavailable` (onnxruntime missing, file missing or not the pinned bytes), or
    :class:`CudaUnavailable` when *device* is ``cuda`` and CUDA cannot be had (before or after building the session).
    A session that comes back without CUDA marks CUDA unusable for the process (:func:`note_cuda_unavailable`)."""
    hint = device or settings.device
    _ort()  # missing onnxruntime is said as such (how to install it), whatever the device
    names = tuple(providers(hint))
    if requires_cuda(hint):
        problem = cuda_problem()
        if problem is not None:
            raise CudaUnavailable(problem)
    path = settings.model_path(name)
    key = (str(path), names, settings.gpu_mem_limit_mb)
    with _session_lock:
        hit = _sessions.get(key)
        if hit is not None:
            return hit
        ort = _ort()
        data = read_model(settings, name)
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only: ORT's warnings about unused initialisers are noise
        try:
            session = ort.InferenceSession(
                data, sess_options=options, providers=session_providers(names, settings.gpu_mem_limit_mb)
            )
        except Exception as exc:  # noqa: BLE001 - any ORT load failure is "cannot run this model"
            raise ModelUnavailable(f"Model {name} at {path} could not be loaded: {exc}.") from exc
        finally:
            del data
        got = list(session.get_providers())
        log.info("session_loaded", model=name, providers=got)
        if CUDA in names and CUDA not in got:
            # ORT could not create the CUDA provider (it logs why) and built the session on the CPU.
            note_cuda_unavailable(
                f"onnxruntime built the {name} session without CUDAExecutionProvider (it got {', '.join(got) or 'none'})"
            )
            if requires_cuda(hint):
                raise CudaUnavailable(cuda_problem() or "the CUDA provider could not be created")
            key = (str(path), (CPU,), settings.gpu_mem_limit_mb)  # what it is: a CPU session, reused as one
        _sessions[key] = session
        return session


def clear_sessions() -> None:
    """Drop every cached session (tests; freeing memory)."""
    with _session_lock:
        _sessions.clear()


def reset_cuda_state() -> None:
    """Forget what is known about CUDA here, so the next question probes again (tests)."""
    global _cuda_problem, _cuda_probed
    with _cuda_state_lock:
        _cuda_problem = None
        _cuda_probed = False
