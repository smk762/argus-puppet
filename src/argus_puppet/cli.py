"""The ``argus-puppet`` command (the ``[cli]`` extra).

One Typer app; each feature adds its own ``@app.command()`` (or a sub-app with ``app.add_typer``). Keep heavy
imports (onnxruntime, numpy work) inside the command functions so ``--help`` stays fast.
"""

from __future__ import annotations

from pathlib import Path

import typer
from typer import Option

from argus_puppet.models import SCHEMA_PATH

app = typer.Typer(
    help="Body pose for the Argus suite: whole-body keypoints, pose metrics and ControlNet pose conditioning.",
    no_args_is_help=True,
    add_completion=False,
)


def fail(message: str) -> typer.Exit:
    """Print *message* to stderr and return the exit to raise (``raise fail(...)``)."""
    typer.echo(message, err=True)
    return typer.Exit(1)


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
