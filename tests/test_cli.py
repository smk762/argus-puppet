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
from argus_puppet.config import DEFAULT_PORT
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
    assert (calls["host"], calls["port"]) == ("127.0.0.1", DEFAULT_PORT)
    assert runner.invoke(app, ["serve", "--port", "9001", "--host", "0.0.0.0"]).exit_code == 0
    assert (calls["host"], calls["port"]) == ("0.0.0.0", 9001)


def test_serve_without_the_server_extra_fails_cleanly(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1 and "[server]" in result.output


# ── stats / compare / render / library ───────────────────────────────────────────────────────────────────────────


def _sidecar_folder(path: Path, names: list[str], *, images: bool = False) -> Path:
    """Sidecars for synthetic poses (one per posegen preset), optionally beside a drawn image whose hash they carry."""
    import hashlib

    from posegen import make, result

    from argus_puppet.extract import write_sidecar
    from argus_puppet.models import sidecar_path

    path.mkdir(parents=True, exist_ok=True)
    for k, name in enumerate(names):
        img = path / f"{k:02d}_{name}.png"
        r = result(img.name, make(name))
        if images:
            data = draw_image(colour=(k * 20 % 255, 40, 90))
            img.write_bytes(data)
            r = r.model_copy(update={"image_sha256": hashlib.sha256(data).hexdigest()})
        write_sidecar(r, sidecar_path(img))
    return path


SET = ["standing", "t_pose", "arms_up", "walk", "floor_sit", "standing"]


def test_stats_summary_json_out_and_min_diversity(tmp_path) -> None:
    from argus_puppet.models import PoseSetStats

    folder = _sidecar_folder(tmp_path / "set", SET)
    result = runner.invoke(app, ["stats", str(folder)])
    assert result.exit_code == 0, result.output
    assert "diversity" in result.output and "clusters" in result.output and "standing" in result.output
    out = tmp_path / "out" / "stats.json"
    result = runner.invoke(app, ["stats", str(folder), "--json", "--out", str(out), "--cluster-threshold", "50"])
    assert result.exit_code == 0, result.output
    stats = PoseSetStats.model_validate_json(result.stdout)
    assert stats.count == 6 and stats.cluster_count == 1 and stats.params.cluster_threshold == 50
    assert PoseSetStats.model_validate_json(out.read_text()) == stats
    assert runner.invoke(app, ["stats", str(folder), "--min-diversity", "0.01"]).exit_code == 0
    low = runner.invoke(app, ["stats", str(folder), "--min-diversity", "0.99"])
    assert low.exit_code == 1 and "below --min-diversity" in low.output
    assert runner.invoke(app, ["stats", str(folder), "--min-usable-joints", "99"]).exit_code == 1


def test_stats_without_sidecars_fails_unless_extracting(monkeypatch, tmp_path) -> None:
    (tmp_path / "a.png").write_bytes(draw_image())
    result = runner.invoke(app, ["stats", str(tmp_path)])
    assert result.exit_code == 1 and "--extract" in result.output
    fake, _ = _fake_estimator(monkeypatch)
    result = runner.invoke(app, ["stats", str(tmp_path), "--extract", "--json"])
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["count"] == 1 and body["estimator"]["provider"] == "FakeExecutionProvider"
    assert fake.calls == 1 and (tmp_path / "a.pose.json").exists()


def test_compare_a_folder_against_itself_is_all_close(tmp_path) -> None:
    from argus_puppet.models import PoseSetComparison

    folder = _sidecar_folder(tmp_path / "set", SET)
    result = runner.invoke(app, ["compare", str(folder), str(folder), "--json"])
    assert result.exit_code == 0, result.output
    cmp = PoseSetComparison.model_validate_json(result.stdout)
    assert cmp.close_share == 1.0 and all(m.distance == 0.0 for m in cmp.matches)
    table = runner.invoke(app, ["compare", str(folder), str(folder), "--threshold", "0.01"])
    assert table.exit_code == 0 and "closest matches" in table.output and "close share" in table.output
    assert "reference baseline" in table.output and "close excess" in table.output
    assert runner.invoke(app, ["compare", str(folder), str(folder), "--threshold", "-1"]).exit_code == 1


def test_render_a_sidecar_a_folder_and_an_image(monkeypatch, tmp_path) -> None:
    from PIL import Image

    folder = _sidecar_folder(tmp_path / "set", SET[:3])
    one = tmp_path / "one.png"
    result = runner.invoke(
        app, ["render", str(folder / "00_standing.pose.json"), "--out", str(one), "--width", "256", "--height", "128"]
    )
    assert result.exit_code == 0, result.output
    assert Image.open(one).size == (256, 128)
    out = tmp_path / "controls"
    result = runner.invoke(app, ["render", str(folder), "--out", str(out), "--no-face", "--stroke-scale", "auto"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in out.iterdir()) == [
        "00_standing.openpose.png",
        "01_t_pose.openpose.png",
        "02_arms_up.openpose.png",
    ]
    fake, _ = _fake_estimator(monkeypatch)
    photo = tmp_path / "photo.png"
    photo.write_bytes(draw_image(80, 60))
    result = runner.invoke(app, ["render", str(photo), "--out", str(tmp_path / "r"), "--primary-only"])
    assert result.exit_code == 0, result.output
    assert Image.open(tmp_path / "r" / "photo.openpose.png").size == (80, 60)
    assert fake.calls == 1 and not (tmp_path / "photo.pose.json").exists()  # no sidecar unless asked
    assert runner.invoke(app, ["render", str(photo), "--out", str(tmp_path / "r"), "--write-sidecar"]).exit_code == 0
    assert (tmp_path / "photo.pose.json").exists()
    assert runner.invoke(app, ["render", str(photo), "--out", str(tmp_path), "--fit", "zoom"]).exit_code == 1
    assert runner.invoke(app, ["render", str(photo), "--out", str(tmp_path), "--stroke-scale", "big"]).exit_code == 1


def test_library_build_and_sample(tmp_path) -> None:
    from PIL import Image

    from argus_puppet.library import load_library

    folder = _sidecar_folder(tmp_path / "set", SET)
    lib_path = tmp_path / "lib.json"
    result = runner.invoke(app, ["library", "build", str(folder), "--out", str(lib_path), "--json"])
    assert result.exit_code == 0, result.output
    counts = json.loads(result.stdout)
    assert counts["considered"] == 6 and counts["entries"] == 5
    assert counts["duplicate_ids"] + counts["near_duplicates"] == 1  # the repeated standing pose
    assert counts["unembeddable"] == 0
    table = runner.invoke(app, ["library", "build", str(folder), "--out", str(tmp_path / "lib2.json")])
    assert table.exit_code == 0 and "unembeddable" in table.output
    lib = load_library(lib_path)
    assert len(lib.entries) == 5 and lib.age_checked is False
    assert all(e.cluster is not None for e in lib.entries)
    renders = tmp_path / "renders"
    result = runner.invoke(
        app, ["library", "sample", str(lib_path), "-n", "3", "--render-dir", str(renders), "--size", "128x192"]
    )
    assert result.exit_code == 0, result.output
    files = sorted(renders.iterdir())
    assert len(files) == 3 and all(Image.open(f).size == (128, 192) for f in files)
    picked = json.loads(runner.invoke(app, ["library", "sample", str(lib_path), "-n", "2", "--json"]).stdout)
    assert len(picked) == 2 and "keypoints" not in picked[0]
    standing = runner.invoke(app, ["library", "sample", str(lib_path), "-n", "9", "--stance", "lying"])
    assert standing.exit_code == 1
    assert runner.invoke(app, ["library", "sample", str(lib_path), "-n", "2", "--stance", "flying"]).exit_code == 1
    assert runner.invoke(app, ["library", "sample", str(lib_path), "-n", "2", "--size", "big"]).exit_code == 0


def test_library_build_age_check(monkeypatch, tmp_path) -> None:
    from argus_puppet import age
    from argus_puppet.library import load_library

    folder = _sidecar_folder(tmp_path / "set", ["standing", "t_pose", "walk"], images=True)
    lib_path = tmp_path / "lib.json"
    no_url = runner.invoke(app, ["library", "build", str(folder), "--out", str(lib_path), "--age-check"])
    assert no_url.exit_code == 1 and "ARGUS_PUPPET_AGE_URL" in no_url.output

    sent: list[bytes] = []
    readings = iter([35.0, 15.0, age.AgeUnavailable("curator answered HTTP 503")])

    class Fake:
        def __init__(self, url: str) -> None:
            assert url == "http://curator.test:8100"

        def min_age(self, data: bytes):
            sent.append(data)
            r = next(readings)
            if isinstance(r, Exception):
                raise r
            return r

    monkeypatch.setattr(age, "HttpAgeEstimator", Fake)
    monkeypatch.setenv("ARGUS_PUPPET_AGE_URL", "http://curator.test:8100")
    result = runner.invoke(app, ["library", "build", str(folder), "--out", str(lib_path), "--age-check", "--json"])
    assert result.exit_code == 0, result.output
    counts = json.loads(result.stdout)
    assert (counts["age_passed"], counts["under_age"], counts["age_unchecked"], counts["entries"]) == (1, 1, 1, 1)
    assert len(sent) == 3 and sent[0] == (folder / "00_standing.png").read_bytes()
    lib = load_library(lib_path)
    assert lib.age_checked and [e.source_image for e in lib.entries] == ["00_standing.png"]

    readings = iter([age.AgeUnavailable("down")] * 3)
    lib_path.unlink()
    down = runner.invoke(app, ["library", "build", str(folder), "--out", str(lib_path), "--age-check"])
    assert down.exit_code == 1 and "could not check any image" in down.output and not lib_path.exists()


def test_without_age_check_no_estimator_is_built(monkeypatch, tmp_path) -> None:
    from argus_puppet import age

    def boom(*a, **kw):
        raise AssertionError("no age check was asked for")

    monkeypatch.setattr(age, "HttpAgeEstimator", boom)
    monkeypatch.setenv("ARGUS_PUPPET_AGE_URL", "http://curator.test:8100")
    folder = _sidecar_folder(tmp_path / "set", ["standing"], images=True)
    assert runner.invoke(app, ["library", "build", str(folder), "--out", str(tmp_path / "l.json")]).exit_code == 0
