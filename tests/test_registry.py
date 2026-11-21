from __future__ import annotations

import pytest

from argus_puppet import registry


def test_defaults_name_models_of_the_right_role() -> None:
    assert registry.get(registry.DEFAULT_POSE).role == "pose"
    assert registry.get(registry.LIGHT_POSE).role == "pose"
    assert registry.get(registry.DEFAULT_DETECTOR).role == "detector"
    assert registry.get(registry.LIGHT_DETECTOR).role == "detector"
    assert set(registry.names("pose")) == {"dwpose-l-384x288", "dwpose-m-256x192"}
    assert set(registry.names("detector")) == {"yolox_m", "yolox_nano"}


def test_entries_are_consistent() -> None:
    for name, m in registry.MODELS.items():
        assert m.name == name and m.filename == f"{name}.onnx"
        assert m.url.startswith("https://")
        assert m.bytes > 0 and m.caveat and m.training_data
        assert (m.archive_sha256 is None) == (m.archive_member is None) == (not m.url.endswith(".zip"))
        assert m.format == ("rtmpose-simcc" if m.role == "pose" else "yolox-raw")
    assert "non-commercial" in registry.MODELS["dwpose-l-384x288"].caveat


def test_get_refuses_unknown_or_wrong_role() -> None:
    with pytest.raises(KeyError, match="choose from"):
        registry.get("openpose")
    with pytest.raises(KeyError, match="yolox_m"):
        registry.get("dwpose-l-384x288", "detector")


def test_model_ref() -> None:
    ref = registry.model_ref("yolox_nano")
    assert ref.name == "yolox_nano" and ref.bytes == 3_659_407 and ref.license == "Apache-2.0"
    assert ref.sha256 == registry.MODELS["yolox_nano"].sha256
