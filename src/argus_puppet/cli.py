"""The ``argus-puppet`` command (the ``[cli]`` extra).

One Typer app; each feature adds its own ``@app.command()`` (or a sub-app with ``app.add_typer``). Keep heavy
imports (onnxruntime, numpy work) inside the command functions so ``--help`` stays fast.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, get_args

import typer
from typer import Argument, Option

from argus_puppet.config import DEFAULT_PORT, DEVICES, ConfigError, Settings
from argus_puppet.models import SCHEMA_PATH, Arms, Facing, Fit, Stance

if TYPE_CHECKING:
    from rich.table import Table

    from argus_puppet.models import MetricParams, PoseResult, PoseSetComparison, PoseSetStats, ScanSummary

app = typer.Typer(
    help="Body pose for the Argus suite: whole-body keypoints, pose metrics and ControlNet pose conditioning.",
    no_args_is_help=True,
    add_completion=False,
)


models_app = typer.Typer(help="List, download and verify the pinned model files.", no_args_is_help=True)
app.add_typer(models_app, name="models")
library_app = typer.Typer(
    help="Build a pose library from a folder, and sample diverse poses from it.", no_args_is_help=True
)
app.add_typer(library_app, name="library")

FITS: tuple[str, ...] = get_args(Fit)


@app.callback()
def _main(verbose: bool = Option(False, "--verbose", "-v", help="Log progress (to stderr)")) -> None:
    """Keep stdout for results (``--json`` is parseable): logs go to stderr, warnings only unless --verbose."""
    import logging
    import sys

    import structlog

    structlog.configure(
        logger_factory=lambda *_: structlog.PrintLogger(sys.stderr),  # resolved per call: stderr may be swapped
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO if verbose else logging.WARNING),
    )


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
    except gpu.CoordinatorInvalid as exc:
        raise fail(str(exc)) from None
    if as_json:
        typer.echo(summary.model_dump_json(indent=2))
    else:
        _print_summary(summary)
    unwritable = summary.failed.get(extract.FAIL_UNWRITABLE, 0)
    if unwritable:
        raise fail(f"{unwritable} sidecar(s) could not be written: is the folder writable?")
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


# ── shared helpers ───────────────────────────────────────────────────────────────────────────────────────────────

DEVICE_HELP = f"Where to run extraction: {', '.join(DEVICES)} (default: settings)"


def _extract_settings(device: str | None) -> Settings:
    return load_settings(device=device.lower() if device else None)


def _run_extraction(folder: Path, *, recursive: bool, settings: Settings) -> None:
    """Pose the images under *folder* that have no sidecar yet (``argus-puppet pose`` without ``--overwrite``)."""
    from argus_puppet import extract, gpu, runtime

    try:
        summary = extract.scan(folder, settings=settings, recursive=recursive, overwrite=False)
    except runtime.ModelUnavailable as exc:
        raise fail(str(exc)) from None
    except gpu.LEASE_ERRORS as exc:
        raise fail(f"no GPU lease: {exc}") from None
    except gpu.CoordinatorInvalid as exc:
        raise fail(str(exc)) from None
    failed = sum(summary.failed.values())
    typer.echo(f"extracted: {summary.written} sidecar(s) written, {failed} image(s) failed", err=True)


def _load_set(folder: Path, *, recursive: bool, run_extract: bool, device: str | None) -> list[PoseResult]:
    """The pose results of a folder's sidecars, optionally extracting the images that lack one first."""
    from argus_puppet import sidecars

    if not folder.is_dir():
        raise fail(f"{folder} is not a folder")
    if run_extract:
        _run_extraction(folder, recursive=recursive, settings=_extract_settings(device))
    try:
        results = sidecars.load_sidecars(folder, recursive=recursive)
    except sidecars.SidecarError as exc:
        raise fail(f"invalid sidecar: {exc}") from None
    if not results:
        raise fail(f"no *.pose.json sidecars under {folder}: run `argus-puppet pose` first, or pass --extract")
    return results


def _params(
    min_joint_score: float | None,
    min_usable_joints: int | None,
    cluster_threshold: float | None,
    min_shared_joints: int | None,
) -> MetricParams:
    from pydantic import ValidationError

    from argus_puppet.models import MetricParams

    given = {
        "min_joint_score": min_joint_score,
        "min_usable_joints": min_usable_joints,
        "cluster_threshold": cluster_threshold,
        "min_shared_joints": min_shared_joints,
    }
    try:
        return MetricParams(**{k: v for k, v in given.items() if v is not None})
    except ValidationError as exc:
        raise fail(f"bad metric parameter: {exc.errors()[0]['loc'][0]}: {exc.errors()[0]['msg']}") from None


def _write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


MIN_JOINT_SCORE = Option(None, "--min-joint-score", help="A joint is confident at or above this score [0.3]")
MIN_USABLE_JOINTS = Option(None, "--min-usable-joints", help="Confident body joints a pose needs to count [8]")
CLUSTER_THRESHOLD = Option(None, "--cluster-threshold", help="Average-linkage cut, in torso units [0.25]")
MIN_SHARED_JOINTS = Option(None, "--min-shared-joints", help="Joints two poses must share to be compared [6]")


def _bar(n: int, total: int, width: int = 20) -> str:
    filled = round(width * n / total) if total else 0
    return "█" * filled + "·" * (width - filled)


# ── stats ────────────────────────────────────────────────────────────────────────────────────────────────────────


@app.command()
def stats(
    folder: Path = Argument(..., help="A folder of images with <stem>.pose.json sidecars"),
    recursive: bool = Option(True, "--recursive/--no-recursive", help="Include subfolders"),
    run_extract: bool = Option(False, "--extract", help="Pose the images that have no sidecar yet, first"),
    device: str | None = Option(None, "--device", help=DEVICE_HELP),
    min_joint_score: float | None = MIN_JOINT_SCORE,
    min_usable_joints: int | None = MIN_USABLE_JOINTS,
    cluster_threshold: float | None = CLUSTER_THRESHOLD,
    min_shared_joints: int | None = MIN_SHARED_JOINTS,
    out: Path | None = Option(None, "--out", "-o", help="Write the PoseSetStats JSON here"),
    as_json: bool = Option(False, "--json", help="Print the PoseSetStats JSON instead of the summary"),
    min_diversity: float | None = Option(
        None, "--min-diversity", help="Exit 1 when the set's diversity is below this (for scripts)"
    ),
) -> None:
    """Pose diversity, clusters and coverage of a folder's poses (from its sidecars)."""
    from argus_puppet.metrics import pose_set_stats

    params = _params(min_joint_score, min_usable_joints, cluster_threshold, min_shared_joints)
    results = _load_set(folder, recursive=recursive, run_extract=run_extract, device=device)
    result = pose_set_stats(results, params=params)
    if run_extract and result.estimator is None:
        typer.echo("note: the sidecars come from more than one estimator; stats.estimator is left empty", err=True)
    text = result.model_dump_json(indent=2)
    if out is not None:
        _write_json(out, text)
    if as_json:
        typer.echo(text)
    else:
        _print_stats(folder, result, out)
    if min_diversity is not None and result.diversity < min_diversity:
        typer.echo(f"diversity {result.diversity:.3f} is below --min-diversity {min_diversity}", err=True)
        raise typer.Exit(1)


def _reason_rows(table: Table, label: str | None, reasons: dict[str, int]) -> None:
    """Rows of a rich summary table: how many images are unusable (under *label*, unless it is None), then one
    indented row per reason. Nothing when every image is usable."""
    if not reasons:
        return
    if label is not None:
        table.add_row(label, str(sum(reasons.values())))
    for reason, n in reasons.items():
        table.add_row(f"  {reason.replace('_', ' ')}", str(n))


def _print_stats(folder: Path, s: PoseSetStats, out: Path | None) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()
    console.print(f"[bold]argus-puppet stats[/bold] {escape(str(folder))}", soft_wrap=True)
    summary = Table(show_header=False)
    summary.add_row("images", str(s.count))
    summary.add_row("usable poses", str(s.usable))
    _reason_rows(summary, "unusable, by reason", s.unusable_reasons)
    summary.add_row("comparable pairs", f"{s.comparable_pairs} ({s.comparable_share:.0%} of usable pairs)")
    summary.add_row("diversity", f"{s.diversity:.3f}")
    summary.add_row("mean pairwise distance", f"{s.mean_pairwise_distance:.3f}")
    summary.add_row("clusters", str(s.cluster_count))
    summary.add_row("largest cluster share", f"{s.largest_cluster_share:.0%}")
    console.print(summary)
    if s.clusters:
        table = Table(title="clusters")
        for col in ("id", "size", "share", "medoid"):
            table.add_column(col)
        for c in s.clusters[:20]:
            table.add_row(str(c.id), str(c.size), f"{c.share:.0%}", escape(c.medoid))
        if len(s.clusters) > 20:
            table.caption = f"... and {len(s.clusters) - 20} more"
        console.print(table)
    cov = Table(title="coverage (usable poses)")
    for col in ("kind", "label", "count", ""):
        cov.add_column(col)
    for kind, hist in (("stance", s.coverage.stance), ("arms", s.coverage.arms), ("facing", s.coverage.facing)):
        for label, n in hist.items():
            cov.add_row(kind, label, str(n), _bar(n, s.usable))
            kind = ""
    console.print(cov)
    if out is not None:
        console.print(f"wrote {escape(str(out))}")


# ── compare ──────────────────────────────────────────────────────────────────────────────────────────────────────


@app.command()
def compare(
    outputs: Path = Argument(..., help="Folder of output images' sidecars (e.g. a LoRA's samples)"),
    reference: Path = Argument(..., help="Folder of reference (training) images' sidecars"),
    threshold: float | None = Option(None, "--threshold", help="Close pose match distance, torso units [0.04]"),
    recursive: bool = Option(True, "--recursive/--no-recursive", help="Include subfolders"),
    run_extract: bool = Option(False, "--extract", help="Pose images that have no sidecar yet, first"),
    device: str | None = Option(None, "--device", help=DEVICE_HELP),
    min_joint_score: float | None = MIN_JOINT_SCORE,
    min_usable_joints: int | None = MIN_USABLE_JOINTS,
    min_shared_joints: int | None = MIN_SHARED_JOINTS,
    top: int = Option(20, "--top", help="Closest matches to list"),
    out: Path | None = Option(None, "--out", "-o", help="Write the PoseSetComparison JSON here"),
    as_json: bool = Option(False, "--json", help="Print the PoseSetComparison JSON instead of the summary"),
) -> None:
    """Each output's nearest reference pose, close pose matches against the reference set's own baseline, and how
    far the pose mix drifted."""
    from argus_puppet.compare import DEFAULT_CLOSE_THRESHOLD, compare_pose_sets

    if threshold is not None and threshold < 0:
        raise fail("--threshold must be >= 0")
    params = _params(min_joint_score, min_usable_joints, None, min_shared_joints)
    out_set = _load_set(outputs, recursive=recursive, run_extract=run_extract, device=device)
    ref_set = _load_set(reference, recursive=recursive, run_extract=run_extract, device=device)
    result = compare_pose_sets(
        out_set, ref_set, threshold=DEFAULT_CLOSE_THRESHOLD if threshold is None else threshold, params=params
    )
    text = result.model_dump_json(indent=2)
    if out is not None:
        _write_json(out, text)
    if as_json:
        typer.echo(text)
    else:
        _print_comparison(result, top, out)


def _print_comparison(c: PoseSetComparison, top: int, out: Path | None) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()
    summary = Table(show_header=False, title="argus-puppet compare")
    summary.add_row("outputs", str(c.output_count))
    _reason_rows(summary, "unusable outputs, by reason", c.output_unusable_reasons)
    summary.add_row("matched outputs", str(c.matched_count))
    summary.add_row("usable reference poses", str(c.reference_count))
    _reason_rows(summary, "unusable reference images, by reason", c.reference_unusable_reasons)
    summary.add_row("close-match eligible outputs", str(c.close_eligible))
    summary.add_row("close share", f"{c.close_share:.0%} (distance < {c.threshold:g}, whole limb skeleton shared)")
    if c.baseline_close_share is None:
        summary.add_row("reference baseline", "n/a (fewer than 2 comparable reference poses)")
    else:
        summary.add_row("reference baseline", f"{c.baseline_close_share:.0%} (reference vs reference, leave-one-out)")
        summary.add_row("close excess", f"{c.close_excess:.0%} (the run-level signal; per-output matches are pointers)")
    for label, q in (
        ("output nearest distance", c.output_nn_distance),
        ("reference nearest distance", c.reference_nn_distance),
    ):
        if q is not None:
            summary.add_row(label, f"p01 {q.p01:.3f}  p10 {q.p10:.3f}  p50 {q.p50:.3f}  p90 {q.p90:.3f}  (n={q.n})")
    summary.add_row("coverage divergence", f"{c.coverage_divergence:.3f}")
    console.print(summary)
    matched = sorted((m for m in c.matches if m.distance is not None), key=lambda m: (m.distance, m.image))
    if matched:
        table = Table(title=f"closest matches (top {min(top, len(matched))})")
        for col in ("output", "nearest reference", "distance", "shared joints", "close pose match"):
            table.add_column(col)
        for m in matched[:top]:
            flag = "[yellow]yes[/yellow]" if m.close else "no"
            table.add_row(escape(m.image), escape(m.nearest or ""), f"{m.distance:.3f}", f"{m.shared_joints}/12", flag)
        console.print(table)
    if out is not None:
        console.print(f"wrote {escape(str(out))}")


# ── render ───────────────────────────────────────────────────────────────────────────────────────────────────────

CONTROL_SUFFIX = ".openpose.png"


def _stroke(raw: str) -> float | str:
    if raw.strip().lower() == "auto":
        return "auto"
    try:
        return float(raw)
    except ValueError:
        raise fail(f"--stroke-scale must be a number or 'auto', not {raw!r}") from None


def _render_sources(source: Path, *, recursive: bool) -> list[tuple[Path, Path | None]]:
    """(name path, sidecar or None) per pose to draw: a sidecar, an image, or a folder's images and sidecars."""
    from argus_puppet import extract, sidecars
    from argus_puppet.models import SIDECAR_SUFFIX, sidecar_path

    if source.is_file():
        return [(source, source if source.name.endswith(SIDECAR_SUFFIX) else None)]
    images = extract.list_images(source, recursive=recursive)
    found = [(img, sidecar_path(img) if sidecar_path(img).is_file() else None) for img in images]
    covered = {s for _, s in found if s is not None} | {sidecar_path(img) for img in images}
    orphans = [(p, p) for p in sidecars.find_sidecars(source, recursive=recursive) if p not in covered]
    return sorted(found + orphans, key=lambda t: t[0].as_posix())


def _stem(path: Path) -> str:
    from argus_puppet.models import SIDECAR_SUFFIX

    return path.name[: -len(SIDECAR_SUFFIX)] if path.name.endswith(SIDECAR_SUFFIX) else path.stem


@app.command()
def render(
    source: Path = Argument(..., help="A <stem>.pose.json sidecar, an image, or a folder of either", exists=True),
    out: Path = Option(..., "--out", "-o", help="Output folder, or a .png file when the source is one pose"),
    body: bool = Option(True, "--body/--no-body", help="Draw the body"),
    hands: bool = Option(True, "--hands/--no-hands", help="Draw the hands"),
    face: bool = Option(True, "--face/--no-face", help="Draw the face landmarks"),
    width: int | None = Option(None, "--width", min=1, max=8192, help="Canvas width (default: the source's)"),
    height: int | None = Option(None, "--height", min=1, max=8192, help="Canvas height (default: the source's)"),
    fit: str = Option("contain", "--fit", help=f"How the source frame maps onto the canvas: {', '.join(FITS)}"),
    primary_only: bool = Option(False, "--primary-only", help="Draw only the primary person"),
    stroke_scale: str = Option(
        "auto", "--stroke-scale", help="Stroke width factor, or 'auto' (min side / 512, the annotators' scaling)"
    ),
    min_score: float = Option(0.3, "--min-score", help="Points scoring below this are not drawn"),
    recursive: bool = Option(False, "--recursive", "-r", help="With a folder: include subfolders"),
    write_sidecar: bool = Option(False, "--write-sidecar", help="Save the sidecar of any image posed here"),
    device: str | None = Option(None, "--device", help=DEVICE_HELP),
) -> None:
    """Draw OpenPose-convention skeleton control images (PNG) for pose ControlNets.

    An image without a sidecar is posed first (in memory; --write-sidecar keeps the sidecar). Outputs are named
    <stem>.openpose.png."""
    from pydantic import ValidationError

    from argus_puppet import extract, gpu, runtime, sidecars
    from argus_puppet.models import RenderOptions
    from argus_puppet.render import render_png

    if fit not in FITS:
        raise fail(f"--fit must be one of {', '.join(FITS)}")
    try:
        options = RenderOptions(
            body=body,
            hands=hands,
            face=face,
            width=width,
            height=height,
            fit=fit,  # type: ignore[arg-type]
            primary_only=primary_only,
            stroke_scale=_stroke(stroke_scale),  # type: ignore[arg-type]
            min_score=min_score,
        )
    except ValidationError as exc:
        raise fail(f"bad render option: {exc.errors()[0]['loc'][0]}: {exc.errors()[0]['msg']}") from None
    items = _render_sources(source, recursive=recursive)
    if not items:
        raise fail(f"nothing to render under {source} (no images or sidecars)")
    single_file = len(items) == 1 and out.suffix.lower() == ".png"
    if not single_file:
        out.mkdir(parents=True, exist_ok=True)
        if source.is_dir() and out.resolve() == source.resolve():
            typer.echo(f"note: control images land beside the photos ({CONTROL_SUFFIX})", err=True)
    base = source if source.is_dir() else source.parent
    settings = None
    written = failed = 0
    for path, sidecar in items:
        try:
            if sidecar is not None:
                pose = sidecars.load_sidecar(sidecar)
            else:
                settings = settings or _extract_settings(device)
                pose = extract.pose_image(path, name=path.relative_to(base).as_posix(), settings=settings)
                if write_sidecar:
                    from argus_puppet.models import sidecar_path

                    extract.write_sidecar(pose, sidecar_path(path))
            png = render_png(pose, options)
        except (runtime.ModelUnavailable, gpu.CoordinatorInvalid, *gpu.LEASE_ERRORS) as exc:
            raise fail(str(exc)) from None
        except (sidecars.SidecarError, ValueError, OSError) as exc:
            typer.echo(f"failed {path}: {exc}", err=True)
            failed += 1
            continue
        if single_file:
            target = out
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            rel = path.parent.relative_to(base) if path.is_relative_to(base) else Path()
            target = out / rel / (_stem(path) + CONTROL_SUFFIX)
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(png)
        written += 1
        typer.echo(f"wrote {target}")
    if failed or not written:
        raise typer.Exit(1)


# ── library ──────────────────────────────────────────────────────────────────────────────────────────────────────


def _choice(value: str | None, kind: object, flag: str) -> str | None:
    allowed = get_args(kind)
    if value is not None and value not in allowed:
        raise fail(f"{flag} must be one of {', '.join(allowed)}")
    return value


def _size(raw: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in raw.lower().split("x"))
    except ValueError:
        raise fail(f"--size must be WIDTHxHEIGHT, e.g. 1024x1024, not {raw!r}") from None
    if not (1 <= w <= 8192 and 1 <= h <= 8192):
        raise fail("--size sides must be between 1 and 8192")
    return w, h


@library_app.command("build")
def library_build(
    folder: Path = Argument(..., help="A folder of images with <stem>.pose.json sidecars"),
    out: Path = Option(..., "--out", "-o", help="Where to write the library JSON"),
    dedupe_threshold: float | None = Option(
        None,
        "--dedupe-threshold",
        help="Drop poses closer than this to a better one (torso units; 0 = off) [0.04]",
    ),
    max_entries: int | None = Option(None, "--max-entries", min=1, help="Keep a diverse subset of this size"),
    min_body_score: float = Option(0.3, "--min-body-score", help="Minimum mean body-joint score of a pose"),
    created_from: str = Option("", "--created-from", help="Provenance note, e.g. 'CC0 photos of adults'"),
    seed: int | None = Option(None, "--seed", help="Seed for the --max-entries subset (default: deterministic)"),
    recursive: bool = Option(True, "--recursive/--no-recursive", help="Include subfolders"),
    run_extract: bool = Option(False, "--extract", help="Pose the images that have no sidecar yet, first"),
    age_check: bool = Option(
        False,
        "--age-check",
        help="Send each source image to argus-curator (ARGUS_PUPPET_AGE_URL) and drop any frame where a face or "
        "body reads under 18, or that cannot be checked",
    ),
    device: str | None = Option(None, "--device", help=DEVICE_HELP),
    as_json: bool = Option(False, "--json", help="Print the build counts as JSON"),
) -> None:
    """Build a pose library: one entry per image's primary pose, labelled, deduplicated and clustered."""
    from argus_puppet import sidecars
    from argus_puppet.library import DEFAULT_DEDUPE_THRESHOLD, AgeGate, build_library, save_library

    if dedupe_threshold is None:
        dedupe_threshold = DEFAULT_DEDUPE_THRESHOLD
    if dedupe_threshold < 0:
        raise fail("--dedupe-threshold must be >= 0")
    gate = None
    if age_check:
        from argus_puppet.age import HttpAgeEstimator

        settings = load_settings()
        if not settings.age_url:
            raise fail("--age-check needs ARGUS_PUPPET_AGE_URL (argus-curator's base URL, e.g. http://localhost:8100)")
        estimator = HttpAgeEstimator(settings.age_url)
    if not folder.is_dir():
        raise fail(f"{folder} is not a folder")
    if run_extract:
        _run_extraction(folder, recursive=recursive, settings=_extract_settings(device))
    try:
        pairs = sidecars.load_sidecars_with_paths(folder, recursive=recursive)
    except sidecars.SidecarError as exc:
        raise fail(f"invalid sidecar: {exc}") from None
    if not pairs:
        raise fail(f"no *.pose.json sidecars under {folder}: run `argus-puppet pose` first, or pass --extract")
    if age_check:
        image_of = {id(r): sidecars.image_for_sidecar(p) for p, r in pairs}

        def load_image(result: PoseResult) -> bytes:
            path = image_of.get(id(result))
            if path is None:
                raise LookupError("no image beside the sidecar")
            return path.read_bytes()

        gate = AgeGate(estimator, load_image)
    build = build_library(
        [r for _, r in pairs],
        min_body_score=min_body_score,
        created_from=created_from,
        dedupe_threshold=dedupe_threshold or None,
        max_entries=max_entries,
        seed=seed,
        age_gate=gate,
    )
    lib = build.library
    nothing_checked = age_check and build.age_checked == 0 and build.age_unchecked > 0
    if lib.entries and not nothing_checked:
        save_library(lib, out)
    counts = {
        "library": str(out) if lib.entries and not nothing_checked else None,
        "entries": len(lib.entries),
        "considered": build.considered,
        "no_person": build.no_person,
        "low_score": build.low_score,
        "unembeddable": build.unembeddable,
        "unembeddable_reasons": build.unembeddable_reasons,
        "duplicate_ids": build.duplicate_ids,
        "near_duplicates": build.near_duplicates,
        "sampled_out": build.sampled_out,
        "clusters": len({e.cluster for e in lib.entries if e.cluster is not None}),
        "age_check": age_check,
        "age_passed": build.age_passed,
        "under_age": build.under_age,
        "age_unchecked": build.age_unchecked,
        "age_excluded": [{"image": i, "why": w} for i, w in build.age_excluded],
    }
    if as_json:
        typer.echo(json.dumps(counts, indent=2))
    else:
        _print_build(counts)
    if nothing_checked:
        typer.echo("the age check could not check any image: no library written", err=True)
        raise typer.Exit(1)
    if not lib.entries:
        typer.echo("no usable poses: no library written", err=True)
        raise typer.Exit(1)


def _print_build(counts: dict) -> None:
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()
    table = Table(show_header=False, title="argus-puppet library build")
    rows = ["entries", "considered", "no_person", "low_score", "unembeddable", "duplicate_ids", "near_duplicates"]
    rows.append("sampled_out")
    rows.append("clusters")
    if counts["age_check"]:
        rows += ["age_passed", "under_age", "age_unchecked"]
    for key in rows:
        table.add_row(key.replace("_", " "), str(counts[key]))
        if key == "unembeddable":
            _reason_rows(table, None, counts["unembeddable_reasons"])
    console.print(table)
    for item in counts["age_excluded"]:
        console.print(f"[yellow]age check kept out[/yellow] {escape(item['image'])}: {escape(item['why'])}")
    if counts["library"]:
        console.print(f"wrote {escape(counts['library'])}")


@library_app.command("sample")
def library_sample(
    library: Path = Argument(..., help="A library JSON written by `library build`", exists=True, dir_okay=False),
    n: int = Option(..., "-n", "--count", min=1, help="How many poses to pick"),
    seed: int | None = Option(None, "--seed", help="Random first pick, reproducibly (default: the most central)"),
    stance: str | None = Option(None, "--stance", help=f"Only this stance: {', '.join(get_args(Stance))}"),
    arms: str | None = Option(None, "--arms", help=f"Only these arms: {', '.join(get_args(Arms))}"),
    facing: str | None = Option(None, "--facing", help=f"Only this facing: {', '.join(get_args(Facing))}"),
    render_dir: Path | None = Option(None, "--render-dir", help="Write a control PNG per picked pose here"),
    size: str = Option("1024x1024", "--size", help="Control image size WIDTHxHEIGHT (with --render-dir)"),
    hands: bool = Option(True, "--hands/--no-hands", help="Draw the hands"),
    face: bool = Option(True, "--face/--no-face", help="Draw the face landmarks"),
    stroke_scale: str = Option("auto", "--stroke-scale", help="Stroke width factor, or 'auto' (min side / 512)"),
    as_json: bool = Option(False, "--json", help="Print the picked entries as JSON"),
) -> None:
    """Pick N mutually distant poses (farthest-point sampling, every cluster before any twice)."""
    from pydantic import ValidationError

    from argus_puppet.library import load_library, render_entries, sample
    from argus_puppet.models import RenderOptions

    try:
        lib = load_library(library)
    except (OSError, ValueError, ValidationError) as exc:
        raise fail(f"{library} is not a pose library: {exc}") from None
    picked = sample(
        lib,
        n,
        seed=seed,
        stance=_choice(stance, Stance, "--stance"),  # type: ignore[arg-type]
        arms=_choice(arms, Arms, "--arms"),  # type: ignore[arg-type]
        facing=_choice(facing, Facing, "--facing"),  # type: ignore[arg-type]
    )
    paths: list[str] = []
    if render_dir is not None and picked:
        canvas = _size(size)
        render_dir.mkdir(parents=True, exist_ok=True)
        options = RenderOptions(hands=hands, face=face, stroke_scale=_stroke(stroke_scale))  # type: ignore[arg-type]
        for i, (entry, img) in enumerate(zip(picked, render_entries(picked, canvas, options), strict=True)):
            target = render_dir / f"{i:02d}_{entry.id}.png"
            img.save(target, format="PNG")
            paths.append(str(target))
    if as_json:
        rows = [
            {**e.model_dump(mode="json", exclude={"keypoints"}), "control_image": paths[i] if paths else None}
            for i, e in enumerate(picked)
        ]
        typer.echo(json.dumps(rows, indent=2))
    else:
        from rich.console import Console
        from rich.markup import escape
        from rich.table import Table

        table = Table(title=f"{len(picked)} of {len(lib.entries)} poses")
        for col in ("#", "source", "stance", "arms", "facing", "cluster"):
            table.add_column(col)
        for i, e in enumerate(picked):
            cluster = "-" if e.cluster is None else str(e.cluster)
            table.add_row(str(i), escape(e.source_image), e.labels.stance, e.labels.arms, e.labels.facing, cluster)
        Console().print(table)
        for p in paths:
            typer.echo(f"wrote {p}")
    if not picked:
        typer.echo("no pose matched the filters", err=True)
        raise typer.Exit(1)


# ── serve ────────────────────────────────────────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: str = Option("127.0.0.1", "--host", help="Interface to bind (loopback by default: there is no auth)"),
    port: int | None = Option(None, "--port", help=f"Port (default: ARGUS_PUPPET_PORT, else {DEFAULT_PORT})"),
) -> None:
    """Run the HTTP service (needs the server extra)."""
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
