"""The ``argus-puppet`` command (the ``[cli]`` extra).

One Typer app; each feature adds its own ``@app.command()`` (or a sub-app with ``app.add_typer``). Keep heavy
imports (onnxruntime, numpy work) inside the command functions so ``--help`` stays fast.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from typer import Argument, Option

from argus_puppet.config import DEVICES, ConfigError, Settings
from argus_puppet.models import SCHEMA_PATH

if TYPE_CHECKING:
    from argus_puppet.models import ScanSummary

app = typer.Typer(
    help="Body pose for the Argus suite: whole-body keypoints, pose metrics and ControlNet pose conditioning.",
    no_args_is_help=True,
    add_completion=False,
)


models_app = typer.Typer(help="List, download and verify the pinned model files.", no_args_is_help=True)
app.add_typer(models_app, name="models")


def fail(message: str) -> typer.Exit:
    """Print *message* to stderr and return the exit to raise (``raise fail(...)``)."""
    typer.echo(message, err=True)
    return typer.Exit(1)


def load_settings(**overrides: object) -> Settings:
    """Settings from ``ARGUS_PUPPET_*`` with the non-``None`` *overrides* applied; a bad value exits 1."""
    try:
        settings = Settings.from_env()
        changes = {k: v for k, v in overrides.items() if v is not None}
        return dataclasses.replace(settings, **changes) if changes else settings
    except ConfigError as exc:
        raise fail(str(exc)) from None


@app.command()
def version() -> None:
    """Print the package version and the wire-contract version."""
    from argus_puppet import __version__
    from argus_puppet.models import PUPPET_VERSION

    typer.echo(f"argus-puppet {__version__} (puppet_version {PUPPET_VERSION})")


@app.command()
def schema(
    output: Path = Option(SCHEMA_PATH, "--output", "-o", help="Where to write the JSON Schema"),
    check: bool = Option(False, "--check", help="Exit non-zero if the committed schema is stale (for CI)"),
) -> None:
    """Emit the wire-contract JSON Schema consumers codegen against."""
    from argus_puppet.models import render_wire_schema

    rendered = render_wire_schema()
    if check:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        if existing != rendered:
            raise fail(f"{output} is stale: run `argus-puppet schema` and commit the result.")
        typer.echo(f"{output} is up to date.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    typer.echo(f"wrote {output}")


# ── pose ─────────────────────────────────────────────────────────────────────────────────────────────────────────


@app.command()
def pose(
    target: Path = Argument(..., help="A folder of images, or one image file", exists=True),
    recursive: bool = Option(False, "--recursive", "-r", help="Include subfolders"),
    overwrite: bool = Option(False, "--overwrite", help="Re-pose images that already have a sidecar"),
    device: str | None = Option(None, "--device", help=f"Where to run: {', '.join(DEVICES)} (default: settings)"),
    pose_model: str | None = Option(None, "--pose-model", help="Pose model registry name (default: settings)"),
    detector: str | None = Option(None, "--detector", help="Person detector registry name (default: settings)"),
    as_json: bool = Option(False, "--json", help="Print the run summary as JSON"),
) -> None:
    """Write a <stem>.pose.json sidecar beside each image: every person's 133 whole-body keypoints."""
    from argus_puppet import extract, gpu, runtime

    settings = load_settings(device=device.lower() if device else None, pose_model=pose_model, detector_model=detector)
    try:
        summary = extract.scan(target, settings=settings, recursive=recursive, overwrite=overwrite)
    except runtime.ModelUnavailable as exc:
        raise fail(str(exc)) from None
    except gpu.LEASE_ERRORS as exc:
        raise fail(f"no GPU lease: {exc}") from None
    if as_json:
        typer.echo(summary.model_dump_json(indent=2))
    else:
        _print_summary(summary)
    if summary.images == 0 or (summary.written == 0 and summary.failed):
        raise typer.Exit(1)


def _print_summary(summary: ScanSummary) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()
    console.print(f"[bold]argus-puppet pose[/bold] {escape(summary.root)}", soft_wrap=True)
    table = Table(show_header=False)
    table.add_row("images", str(summary.images))
    table.add_row("sidecars written", str(summary.written))
    table.add_row("people found", str(summary.people))
    for reason, n in summary.skipped.items():
        table.add_row(f"skipped ({reason})", str(n))
    for reason, n in summary.failed.items():
        table.add_row(f"failed ({reason})", f"[red]{n}[/red]")
    console.print(table)
    for o in summary.outcomes:
        if o.status == "failed":
            console.print(f"[red]failed[/red] {escape(o.image)}: {escape(o.detail or o.reason or '')}")
    if summary.images == 0:
        console.print("[yellow]no images found[/yellow] (.jpg, .jpeg, .png, .webp)")


# ── serve ────────────────────────────────────────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: str = Option("127.0.0.1", "--host", help="Interface to bind (loopback by default: there is no auth)"),
    port: int | None = Option(None, "--port", help="Port (default: ARGUS_PUPPET_PORT, else 8110)"),
) -> None:
    """Run the HTTP service (needs the [server] extra)."""
    try:
        import uvicorn

        from argus_puppet.server.app import create_app
    except ImportError as exc:
        raise fail(f"the server needs the [server] extra: pip install 'argus-puppet[server]' ({exc})") from None
    settings = load_settings(port=port)
    uvicorn.run(create_app(settings), host=host, port=settings.port)


# ── models ───────────────────────────────────────────────────────────────────────────────────────────────────────


def _model_names(names: list[str] | None) -> list[str]:
    from argus_puppet import registry

    try:
        return [registry.get(n).name for n in names] if names else list(registry.names())
    except KeyError as exc:
        raise fail(exc.args[0]) from None


@models_app.command("list")
def models_list(
    verify: bool = Option(True, "--verify/--no-verify", help="Hash the files that are present"),
    as_json: bool = Option(False, "--json", help="Print as JSON"),
) -> None:
    """Show each pinned model: role, licence, size, and whether its file is present and verified."""
    from rich.console import Console
    from rich.table import Table

    from argus_puppet import weights

    settings = load_settings()
    rows = weights.status(settings, verify=verify)
    if as_json:
        out = [{**dataclasses.asdict(r), "path": str(r.path)} for r in rows]
        typer.echo(json.dumps(out, indent=2))
        return
    table = Table(title=f"models in {settings.model_dir}")
    for col in ("name", "role", "licence", "MB", "present", "verified"):
        table.add_column(col)
    for r in rows:
        verified = "-" if r.verified is None else ("[green]yes[/green]" if r.verified else "[red]no[/red]")
        table.add_row(r.name, r.role, r.license, f"{r.bytes / 1e6:.1f}", "yes" if r.present else "no", verified)
    Console().print(table)


@models_app.command("fetch")
def models_fetch(
    names: list[str] | None = Argument(None, help="Registry names (default: every pinned model)"),
    force: bool = Option(False, "--force", help="Download even when a verified file is present"),
) -> None:
    """Download the pinned models into the model directory, verifying each before it is written."""
    from argus_puppet import weights

    settings = load_settings()
    failed = 0
    for name in _model_names(names):
        try:
            path = weights.fetch_model(settings, name, force=force)
        except weights.WeightsError as exc:
            typer.echo(f"{name}: {exc}", err=True)
            failed += 1
            continue
        typer.echo(f"{name}: {path}")
    if failed:
        raise typer.Exit(1)


@models_app.command("verify")
def models_verify(
    names: list[str] | None = Argument(None, help="Registry names (default: every pinned model)"),
) -> None:
    """Check the model files against their pinned size and SHA-256; exit 1 if any is missing or wrong."""
    from argus_puppet import runtime

    settings = load_settings()
    bad = 0
    for name in _model_names(names):
        try:
            path = runtime.check_model(settings, name)
        except runtime.ModelUnavailable as exc:
            typer.echo(f"{name}: {exc}", err=True)
            bad += 1
            continue
        typer.echo(f"{name}: ok ({path})")
    if bad:
        raise typer.Exit(1)
