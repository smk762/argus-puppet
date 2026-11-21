from __future__ import annotations

from pathlib import Path

import pytest

from argus_puppet import registry
from argus_puppet.config import DEFAULT_PORT, ConfigError, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch) -> None:
    import os

    for key in list(os.environ):
        if key.startswith("ARGUS_PUPPET_"):
            monkeypatch.delenv(key)


def test_defaults() -> None:
    s = Settings.from_env()
    assert s.port == DEFAULT_PORT == 8220
    assert s.model_dir == Path("~/.cache/argus_puppet/models").expanduser()
    assert (s.pose_model, s.detector_model) == (registry.DEFAULT_POSE, registry.DEFAULT_DETECTOR)
    assert s.device == "auto" and s.gpu_mem_limit_mb is None
    assert s.data_root is None and s.read_only is False
    assert s.cors_origins == () and s.trusted_hosts == ()
    assert s.body_timeout_s == 60.0
    assert s.model_path("yolox_m") == s.model_dir / "yolox_m.onnx"


def test_parses_every_setting(monkeypatch, tmp_path) -> None:
    env = {
        "PORT": "9000",
        "MODEL_DIR": str(tmp_path / "m"),
        "POSE_MODEL": registry.LIGHT_POSE,
        "DETECTOR_MODEL": registry.LIGHT_DETECTOR,
        "DEVICE": "CUDA",
        "GPU_MEM_LIMIT_MB": "2048",
        "DATA_ROOT": str(tmp_path / "data" / ".." / "data"),
        "READ_ONLY": "yes",
        "CORS_ORIGINS": "http://a.example/, http://b.example",
        "TRUSTED_HOSTS": "Morpheus, ,box",
        "MAX_UPLOAD_BYTES": "1048576",
        "MAX_PIXELS": "1000000",
        "BODY_TIMEOUT_S": "12.5",
    }
    for k, v in env.items():
        monkeypatch.setenv(f"ARGUS_PUPPET_{k}", v)
    s = Settings.from_env()
    assert s.port == 9000 and s.model_dir == tmp_path / "m"
    assert (s.pose_model, s.detector_model) == (registry.LIGHT_POSE, registry.LIGHT_DETECTOR)
    assert s.device == "cuda" and s.gpu_mem_limit_mb == 2048
    assert s.data_root == (tmp_path / "data").resolve()
    assert s.read_only is True
    assert s.cors_origins == ("http://a.example", "http://b.example")
    assert s.trusted_hosts == ("morpheus", "box")
    assert (s.max_upload_bytes, s.max_pixels) == (1048576, 1000000)
    assert s.body_timeout_s == 12.5


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("PORT", "eighty", "not a whole number"),
        ("PORT", "70000", "between 1 and 65535"),
        ("POSE_MODEL", "yolox_m", "POSE_MODEL"),
        ("DETECTOR_MODEL", "nope", "DETECTOR_MODEL"),
        ("DEVICE", "tpu", "choose from"),
        ("GPU_MEM_LIMIT_MB", "10", "at least 256"),
        ("MAX_UPLOAD_BYTES", "1", "at least 1024"),
        ("MAX_PIXELS", "-5", "at least"),
        ("BODY_TIMEOUT_S", "0", "above 0"),
        ("BODY_TIMEOUT_S", "soon", "not a number"),
    ],
)
def test_bad_values_raise_config_error(monkeypatch, key, value, match) -> None:
    monkeypatch.setenv(f"ARGUS_PUPPET_{key}", value)
    with pytest.raises(ConfigError, match=match):
        Settings.from_env()


def test_a_mistyped_read_only_flag_is_off(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_PUPPET_READ_ONLY", "enabled")
    assert Settings.from_env().read_only is False
