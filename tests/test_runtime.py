"""Providers, the CUDA arena cap, model verification and session caching (a fake onnxruntime; no model files)."""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
import sys
import types

import pytest

from argus_puppet import registry, runtime
from argus_puppet.config import Settings


def fake_ort(monkeypatch, available=("CPUExecutionProvider",)):
    """Install a fake ``onnxruntime`` module that records the sessions it builds."""
    built = []

    class SessionOptions:
        log_severity_level = 2

    class InferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path, self.providers = path, providers
            built.append(self)

        def get_providers(self):
            return [p[0] if isinstance(p, tuple) else p for p in self.providers]

    mod = types.ModuleType("onnxruntime")
    mod.get_available_providers = lambda: list(available)
    mod.SessionOptions = SessionOptions
    mod.InferenceSession = InferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    monkeypatch.setattr(runtime, "_cuda_preloaded", True)
    runtime.clear_sessions()
    return built


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


def test_the_hash_is_memoised_by_mtime_and_size(pinned, monkeypatch) -> None:
    s, name = pinned
    path = s.model_path(name)
    first = runtime.file_sha256(path)
    calls = []
    real = hashlib.sha256
    monkeypatch.setattr(runtime.hashlib, "sha256", lambda *a: calls.append(1) or real(*a))
    assert runtime.file_sha256(path) == first and calls == []
    path.write_bytes(b"changed!")
    assert runtime.file_sha256(path) != first and calls == [1]


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
