from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

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
