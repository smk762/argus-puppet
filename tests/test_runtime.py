"""Providers, the CUDA arena cap, model verification and session caching (a fake onnxruntime; no model files)."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import sys
import time

import pytest
import structlog.testing
from conftest import fake_ort

from argus_puppet import registry, runtime
from argus_puppet.config import Settings


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """A tiny file pinned in the registry as yolox_nano, written into a model dir."""
    data = b"not really an onnx file"
    m = registry.get("yolox_nano")
    monkeypatch.setitem(
        registry.MODELS, m.name, dataclasses.replace(m, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    )
    s = Settings(model_dir=tmp_path, device="cpu", detector_model="yolox_nano")
    s.model_path(m.name).write_bytes(data)
    return s, m.name


def test_providers_name_cuda_only_when_ort_has_it(monkeypatch) -> None:
    fake_ort(monkeypatch)
    assert runtime.providers("cuda") == ["CPUExecutionProvider"]
    assert runtime.providers("auto") == ["CPUExecutionProvider"]
    assert not runtime.uses_cuda("cuda")
    fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"))
    assert runtime.providers("auto") == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime.providers("cpu") == ["CPUExecutionProvider"]
    assert runtime.uses_cuda("cuda")


def test_providers_without_onnxruntime(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import raises ImportError
    assert runtime.providers("cuda") == ["CPUExecutionProvider"]
    assert not runtime.onnxruntime_available()


def test_every_cuda_session_has_a_capped_arena() -> None:
    names = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    (cuda, opts), cpu = runtime.session_providers(names, None)
    assert cuda == "CUDAExecutionProvider" and cpu == "CPUExecutionProvider"
    assert opts["arena_extend_strategy"] == "kSameAsRequested"
    assert opts["gpu_mem_limit"] == str(runtime.DEFAULT_GPU_MEM_LIMIT_MB * 1024 * 1024)
    assert runtime.session_providers(names, 2048)[0][1]["gpu_mem_limit"] == str(2048 * 1024 * 1024)


def test_check_model_verifies_size_and_hash(pinned) -> None:
    s, name = pinned
    assert runtime.check_model(s, name) == s.model_path(name)
    assert runtime.model_verified(s, name)
    s.model_path(name).write_bytes(b"x" * len(b"not really an onnx file"))  # same size, other bytes
    with pytest.raises(runtime.ModelUnavailable, match="SHA-256"):
        runtime.check_model(s, name)
    s.model_path(name).write_bytes(b"short")
    with pytest.raises(runtime.ModelUnavailable, match="bytes"):
        runtime.check_model(s, name)
    s.model_path(name).unlink()
    with pytest.raises(runtime.ModelUnavailable, match="models fetch"):
        runtime.check_model(s, name)
    assert not runtime.model_verified(s, name)


def _counting_sha256(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = hashlib.sha256
    monkeypatch.setattr(runtime.hashlib, "sha256", lambda *a: calls.append(1) or real(*a))
    return calls


def test_the_hash_memo_is_invalidated_by_any_change_to_the_file(pinned, monkeypatch) -> None:
    """The memo (health, models verify) is keyed on path, device, inode, size, mtime and ctime: every rewrite below
    keeps the size, so only the other parts of the key can notice it."""
    s, name = pinned
    monkeypatch.setattr(runtime, "_RACY_NS", 0)  # memoise the file just written: this test moves its stamps itself
    path = s.model_path(name)
    size = path.stat().st_size
    first = runtime.file_sha256(path)
    want = {c: hashlib.sha256(c * size).hexdigest() for c in (b"B", b"C")}
    calls = _counting_sha256(monkeypatch)  # counts every hashlib.sha256 call from here on
    assert runtime.file_sha256(path) == first and calls == []  # unchanged: memoised

    path.write_bytes(b"A" * size)  # in place, same size, a new mtime
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    a = runtime.file_sha256(path)
    assert a != first and calls == [1]

    before = path.stat()
    path.write_bytes(b"B" * size)  # in place, same size, the mtime put back: only the ctime can tell
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    while path.stat().st_ctime_ns == before.st_ctime_ns:  # a coarse clock: wait for the ctime to move
        time.sleep(0.005)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert runtime.file_sha256(path) == want[b"B"] and calls == [1, 1]

    swapped = path.with_name("swap.tmp")  # a new file moved into place, every time stamp copied over
    swapped.write_bytes(b"C" * size)
    st = path.stat()
    os.utime(swapped, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(swapped, path)
    assert runtime.file_sha256(path) == want[b"C"] and calls == [1, 1, 1]


def test_a_file_changed_within_one_clock_tick_is_not_memoised(pinned, monkeypatch) -> None:
    """File stamps come from a coarse clock: two same-size writes in one tick share every part of the key (the key is
    pinned here, so the test doesn't depend on the kernel's tick). A memo taken between them would vouch for the first
    bytes, so a file that recent is hashed every time; once it has been still for the window, it is memoised."""
    s, name = pinned
    path = s.model_path(name)
    size = path.stat().st_size
    monkeypatch.setattr(runtime, "_stat_key", lambda p: ("same", 0, 0, size, time.time_ns(), time.time_ns()))
    runtime.file_sha256(path)
    path.write_bytes(b"Z" * size)
    assert runtime.file_sha256(path) == hashlib.sha256(b"Z" * size).hexdigest()

    old = time.time_ns() - runtime._RACY_NS
    monkeypatch.setattr(runtime, "_stat_key", lambda p: ("still", 0, 0, size, old, old))
    first = runtime.file_sha256(path)
    calls = _counting_sha256(monkeypatch)
    assert runtime.file_sha256(path) == first and calls == []


def test_load_session_hashes_the_bytes_it_loads(pinned, monkeypatch) -> None:
    """The bytes handed to onnxruntime are the bytes just hashed: no memo can vouch for them, and the file is not
    reopened by path after the check."""
    s, name = pinned
    built = fake_ort(monkeypatch)
    path = s.model_path(name)
    pinned_bytes = path.read_bytes()
    runtime.load_session(s, name)
    assert built[0].path == pinned_bytes  # the verified bytes themselves, not a path to reopen
    runtime.clear_sessions()
    assert runtime.model_verified(s, name)  # the memo now vouches for this path's stat key
    forged = b"Z" * len(pinned_bytes)
    st = path.stat()
    with path.open("r+b") as f:  # other bytes, written in place
        f.write(forged)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    key = runtime._stat_key(path)
    monkeypatch.setitem(runtime._hash_cache, key, registry.get(name).sha256)  # a memo that wrongly vouches
    with pytest.raises(runtime.ModelUnavailable, match="SHA-256"):
        runtime.load_session(s, name)
    assert len(built) == 1


def test_load_session_verifies_caps_and_caches(pinned, monkeypatch) -> None:
    s, name = pinned
    built = fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"))
    s = dataclasses.replace(s, device="cuda", gpu_mem_limit_mb=512)
    a = runtime.load_session(s, name)
    assert runtime.load_session(s, name) is a and len(built) == 1
    cuda, opts = a.providers[0]
    assert cuda == "CUDAExecutionProvider" and opts["gpu_mem_limit"] == str(512 * 1024 * 1024)


def test_load_session_refuses_a_file_that_is_not_the_pinned_bytes(pinned, monkeypatch) -> None:
    s, name = pinned
    built = fake_ort(monkeypatch)
    s.model_path(name).write_bytes(b"Y" * len(b"not really an onnx file"))
    with pytest.raises(runtime.ModelUnavailable, match="SHA-256"):
        runtime.load_session(s, name)
    assert built == []  # never handed to onnxruntime


def test_load_session_without_onnxruntime_says_how_to_install(pinned, monkeypatch) -> None:
    s, name = pinned
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    runtime.clear_sessions()
    with pytest.raises(runtime.ModelUnavailable, match=r"argus-puppet\[cpu\]"):
        runtime.load_session(s, name)


def test_importing_the_package_does_not_need_onnxruntime() -> None:
    code = (
        "import sys; sys.modules['onnxruntime'] = None\n"
        "import argus_puppet, argus_puppet.estimator, argus_puppet.runtime, argus_puppet.extract, argus_puppet.cli\n"
        "import argus_puppet.server.app, argus_puppet.weights, argus_puppet.gpu, argus_puppet.primary\n"
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# ── device truth: CUDA counts only when a session really got it ─────────────────────────────────────────────────────

REAL_PROBE = runtime._probe_cuda_driver  # the autouse fixture stubs it per test; this is the real one


def _cuda_warnings(logs) -> list[dict]:
    return [e for e in logs if e["event"] == "cuda_unavailable"]


def test_a_session_that_came_back_without_cuda_moves_auto_to_the_cpu(pinned, monkeypatch) -> None:
    """ORT has CUDA compiled in but could not create it: the session runs on the CPU. That is seen, logged once, and
    from then on auto names the CPU (so no GPU lease), and the CPU session is reused rather than rebuilt."""
    s, name = pinned
    s = dataclasses.replace(s, device="auto")
    built = fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"), drops_cuda=True)
    assert runtime.uses_cuda("auto")  # all that is known before a session: compiled in, driver sees a GPU
    with structlog.testing.capture_logs() as logs:
        session = runtime.load_session(s, name)
        assert runtime.load_session(s, name) is session and len(built) == 1
        runtime.providers("auto")
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert not runtime.uses_cuda("auto") and runtime.providers("auto") == ["CPUExecutionProvider"]
    assert runtime.effective_device("auto") == "cpu"
    assert "without CUDAExecutionProvider" in runtime.cuda_problem()
    warned = _cuda_warnings(logs)
    assert len(warned) == 1 and warned[0]["log_level"] == "warning" and "CUDAExecutionProvider" in warned[0]["reason"]


def test_explicit_cuda_is_refused_rather_than_run_on_the_cpu(pinned, monkeypatch) -> None:
    """ARGUS_PUPPET_DEVICE=cuda and the session came back on the CPU: refused, loudly, and so is every later load."""
    s, name = pinned
    s = dataclasses.replace(s, device="cuda")
    built = fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"), drops_cuda=True)
    with pytest.raises(runtime.CudaUnavailable, match="ARGUS_PUPPET_DEVICE=auto"):
        runtime.load_session(s, name)
    with pytest.raises(runtime.CudaUnavailable):
        runtime.load_session(s, name)
    assert len(built) == 1  # the second refusal builds nothing
    assert not runtime.uses_cuda("cuda") and runtime.effective_device("cuda") == "cuda"
    assert isinstance(runtime.CudaUnavailable("x"), runtime.ModelUnavailable)  # so /pose answers 503


def test_no_driver_is_known_before_any_session(pinned, monkeypatch) -> None:
    """A CUDA build in a container started without --gpus: the driver probe says so up front (no session needed),
    auto runs on the CPU and explicit cuda is refused before the model is even read."""
    s, name = pinned
    built = fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"))
    probes = []
    monkeypatch.setattr(runtime, "_probe_cuda_driver", lambda: probes.append(1) or "libcuda.so.1 cannot be loaded")
    with structlog.testing.capture_logs() as logs:
        assert runtime.providers("auto") == ["CPUExecutionProvider"] and not runtime.uses_cuda("auto")
        assert runtime.effective_device("auto") == "cpu" and runtime.providers("cpu") == ["CPUExecutionProvider"]
        with pytest.raises(runtime.CudaUnavailable, match="libcuda"):
            runtime.load_session(dataclasses.replace(s, device="cuda"), name)
        assert built == []
        session = runtime.load_session(dataclasses.replace(s, device="auto"), name)
    assert session.providers == ["CPUExecutionProvider"]  # never asked ORT for CUDA
    assert len(probes) == 1 and len(_cuda_warnings(logs)) == 1


def test_no_thread_sees_cuda_usable_while_a_failed_probe_is_being_recorded(monkeypatch) -> None:
    """A second thread asking while the first is recording a failed probe waits for it, never reads "usable"."""
    import threading

    fake_ort(monkeypatch, ("CUDAExecutionProvider", "CPUExecutionProvider"))
    monkeypatch.setattr(runtime, "_probe_cuda_driver", lambda: "no CUDA device")
    real_note = runtime.note_cuda_unavailable
    seen: list[str | None] = []

    def note_while_another_thread_asks(reason: str) -> None:
        other = threading.Thread(target=lambda: seen.append(runtime.cuda_problem()))
        other.start()
        other.join(0.5)  # on a racy build it answers at once; otherwise it waits on the lock until we return
        real_note(reason)

    monkeypatch.setattr(runtime, "note_cuda_unavailable", note_while_another_thread_asks)
    assert runtime.cuda_problem() == "no CUDA device"
    deadline = time.monotonic() + 5
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen == ["no CUDA device"]


def test_explicit_cuda_on_a_cpu_only_onnxruntime_is_refused(pinned, monkeypatch) -> None:
    s, name = pinned
    built = fake_ort(monkeypatch)
    with pytest.raises(runtime.CudaUnavailable, match="no CUDAExecutionProvider"):
        runtime.load_session(dataclasses.replace(s, device="cuda"), name)
    assert built == []


def test_cpu_never_probes_or_warns(monkeypatch) -> None:
    fake_ort(monkeypatch)
    monkeypatch.setattr(runtime, "_probe_cuda_driver", lambda: pytest.fail("probed for the cpu device"))
    with structlog.testing.capture_logs() as logs:
        assert runtime.providers("cpu") == ["CPUExecutionProvider"] and runtime.effective_device("cpu") == "cpu"
    assert _cuda_warnings(logs) == []


class _FakeLibcuda:
    def __init__(self, init_rc: int = 0, count_rc: int = 0) -> None:
        self.init_rc, self.count_rc = init_rc, count_rc

    def cuInit(self, flags):  # noqa: N802 - the driver's name
        return self.init_rc

    def cuDeviceGetCount(self, ref):  # noqa: N802 - leaves the count at 0
        return self.count_rc


def test_the_driver_probe_says_why(monkeypatch) -> None:
    monkeypatch.setattr(runtime.sys, "platform", "linux")

    def missing(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(runtime.ctypes, "CDLL", missing)
    assert "libcuda.so.1" in REAL_PROBE() and "container" in REAL_PROBE()
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda name: _FakeLibcuda(init_rc=100))
    assert "cuInit error 100" in REAL_PROBE()
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda name: _FakeLibcuda())
    assert "sees no GPU" in REAL_PROBE()
    monkeypatch.setattr(runtime.ctypes, "CDLL", lambda name: _FakeLibcuda(count_rc=3))
    assert "error 3" in REAL_PROBE()
