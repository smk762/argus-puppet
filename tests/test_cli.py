from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from conftest import FakeEstimator, draw_image
from typer.testing import CliRunner

from argus_puppet import estimator as estimator_mod
from argus_puppet import registry, weights
from argus_puppet.cli import app
from argus_puppet.models import PUPPET_VERSION, SCHEMA_PATH, render_wire_schema

ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()


def test_the_committed_schema_is_current() -> None:
    assert (ROOT / SCHEMA_PATH).read_text(encoding="utf-8") == render_wire_schema()


def test_schema_writes_and_checks(tmp_path) -> None:
    out = tmp_path / "s.json"
    assert runner.invoke(app, ["schema", "--check", "-o", str(out)]).exit_code == 1  # missing
    assert runner.invoke(app, ["schema", "-o", str(out)]).exit_code == 0
    assert out.read_text(encoding="utf-8") == render_wire_schema()
    result = runner.invoke(app, ["schema", "--check", "-o", str(out)])
    assert result.exit_code == 0, result.output
    out.write_text("{}", encoding="utf-8")
    assert runner.invoke(app, ["schema", "--check", "-o", str(out)]).exit_code == 1


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0 and f"puppet_version {PUPPET_VERSION}" in result.output


# ── pose ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def _fake_estimator(monkeypatch):
    fake = FakeEstimator()
    seen = {}

    def get(settings, **kw):
        seen["settings"] = settings
        return fake

    monkeypatch.setattr(estimator_mod, "get_estimator", get)
    return fake, seen


def test_pose_writes_sidecars_and_summarises(monkeypatch, tmp_path) -> None:
    _, seen = _fake_estimator(monkeypatch)
    (tmp_path / "a.png").write_bytes(draw_image())
    (tmp_path / "b.png").write_bytes(b"junk")
    result = runner.invoke(app, ["pose", str(tmp_path), "--device", "CPU", "--pose-model", "dwpose-m-256x192"])
    assert result.exit_code == 0, result.output
    assert "people found" in result.output and "unreadable" in result.output
    assert (tmp_path / "a.pose.json").exists()
    assert seen["settings"].device == "cpu" and seen["settings"].pose_model == "dwpose-m-256x192"


def test_pose_json(monkeypatch, tmp_path) -> None:
    _fake_estimator(monkeypatch)
    (tmp_path / "a.png").write_bytes(draw_image())
    (tmp_path / "a.pose.json").write_text("{}")
    result = runner.invoke(app, ["pose", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["skipped"] == {"exists": 1} and body["written"] == 0


def test_pose_exits_non_zero_when_nothing_could_be_processed(monkeypatch, tmp_path) -> None:
    _fake_estimator(monkeypatch)
    assert runner.invoke(app, ["pose", str(tmp_path)]).exit_code == 1  # no images
    (tmp_path / "bad.png").write_bytes(b"junk")
    assert runner.invoke(app, ["pose", str(tmp_path)]).exit_code == 1  # every image failed


def test_pose_refuses_bad_settings_and_missing_models(monkeypatch, tmp_path) -> None:
    (tmp_path / "a.png").write_bytes(draw_image())
    result = runner.invoke(app, ["pose", str(tmp_path), "--detector", "nope"])
    assert result.exit_code == 1 and "DETECTOR_MODEL" in result.output
    monkeypatch.setenv("ARGUS_PUPPET_MODEL_DIR", str(tmp_path / "empty"))
    result = runner.invoke(app, ["pose", str(tmp_path), "--device", "cpu"])
    assert result.exit_code == 1 and "models fetch" in result.output


# ── models ───────────────────────────────────────────────────────────────────────────────────────────────────────


def test_models_list_and_verify(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARGUS_PUPPET_MODEL_DIR", str(tmp_path))
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert {r["name"] for r in rows} == set(registry.names())
    assert all(r["present"] is False and r["license"] == "Apache-2.0" for r in rows)
    table = runner.invoke(app, ["models", "list"])
    assert table.exit_code == 0 and "yolox_nano" in table.output
    assert runner.invoke(app, ["models", "verify"]).exit_code == 1
    assert runner.invoke(app, ["models", "verify", "bogus"]).exit_code == 1


def test_models_fetch_reports_each_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ARGUS_PUPPET_MODEL_DIR", str(tmp_path))
    fetched = []

    def fetch_model(settings, name, *, force=False):
        fetched.append((name, force))
        if name == "yolox_m":
            raise weights.WeightsError("does not match its pinned SHA-256")
        return settings.model_path(name)

    monkeypatch.setattr(weights, "fetch_model", fetch_model)
    result = runner.invoke(app, ["models", "fetch", "yolox_nano", "yolox_m", "--force"])
    assert result.exit_code == 1
    assert fetched == [("yolox_nano", True), ("yolox_m", True)]
    assert "yolox_m: does not match" in result.output
    assert runner.invoke(app, ["models", "fetch", "yolox_nano"]).exit_code == 0


# ── serve ────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_serve_binds_loopback_by_default(monkeypatch) -> None:
    calls = {}
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app_, host, port: calls.update(host=host, port=port, app=app_)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    assert runner.invoke(app, ["serve"]).exit_code == 0
    assert (calls["host"], calls["port"]) == ("127.0.0.1", 8110)
    assert runner.invoke(app, ["serve", "--port", "9001", "--host", "0.0.0.0"]).exit_code == 0
    assert (calls["host"], calls["port"]) == ("0.0.0.0", 9001)


def test_serve_without_the_server_extra_fails_cleanly(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1 and "[server]" in result.output
