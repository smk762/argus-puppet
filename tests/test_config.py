from __future__ import annotations

from pathlib import Path

import pytest

from argus_puppet import registry
from argus_puppet.config import DEFAULT_PORT, ConfigError, Settings


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


# ── the age check's URL: puppet's own variable, then the suite's name for curator ────────────────────────────────


def _suite_file(tmp_path: Path, monkeypatch, text: str) -> None:
    path = tmp_path / "suite.env"
    path.write_text(text)
    path.chmod(0o600)
    monkeypatch.setenv("ARGUS_ENV_FILE", str(path))


def test_the_age_url_is_unset_by_default() -> None:
    assert Settings.from_env().age_url == ""


def test_the_age_url_falls_back_to_the_suite_wide_curator_url(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_CURATOR_URL", "http://suite-curator.test:8101/")
    assert Settings.from_env().age_url == "http://suite-curator.test:8101"
    monkeypatch.setenv("ARGUS_PUPPET_AGE_URL", "http://own-curator.test:8101")
    assert Settings.from_env().age_url == "http://own-curator.test:8101"  # puppet's own variable still wins


def test_the_age_url_is_read_from_the_suite_file_behind_the_environment(tmp_path, monkeypatch) -> None:
    _suite_file(tmp_path, monkeypatch, "ARGUS_CURATOR_URL=http://file-curator.test:8101\n")
    assert Settings.from_env().age_url == "http://file-curator.test:8101"
    monkeypatch.setenv("ARGUS_PUPPET_AGE_URL", "")  # blank (compose's ${NAME:-}) is unset
    assert Settings.from_env().age_url == "http://file-curator.test:8101"
    monkeypatch.setenv("ARGUS_CURATOR_URL", "http://env-curator.test:8101")
    assert Settings.from_env().age_url == "http://env-curator.test:8101"


def test_off_keeps_the_age_check_unconfigured_whatever_the_suite_says(tmp_path, monkeypatch) -> None:
    _suite_file(tmp_path, monkeypatch, "ARGUS_CURATOR_URL=http://file-curator.test:8101\n")
    monkeypatch.setenv("ARGUS_PUPPET_AGE_URL", "off")
    assert Settings.from_env().age_url == ""


@pytest.mark.parametrize("name", ["ARGUS_PUPPET_AGE_URL", "ARGUS_CURATOR_URL"])
@pytest.mark.parametrize("value", ["curator.test:8101", "/api/curator", "ftp://curator.test", "http://user:hunter2@"])
def test_a_bad_age_url_is_a_config_error_not_unset_and_is_never_echoed(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError) as caught:
        Settings.from_env()
    assert name in str(caught.value) and value not in str(caught.value) and "hunter2" not in str(caught.value)


def test_a_bad_own_age_url_is_refused_even_with_a_good_suite_one(monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_CURATOR_URL", "http://suite-curator.test:8101")
    monkeypatch.setenv("ARGUS_PUPPET_AGE_URL", "not a url")
    with pytest.raises(ConfigError, match="ARGUS_PUPPET_AGE_URL"):
        Settings.from_env()
