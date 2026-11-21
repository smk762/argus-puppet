"""Read a folder's pose sidecars (``<stem>.pose.json``) back into :class:`~argus_puppet.models.PoseResult` s.

So a caller can write ``pose_set_stats(load_sidecars(folder))`` without re-running the estimator.
"""

from __future__ import annotations

import glob
from pathlib import Path

from pydantic import ValidationError

from argus_puppet.models import SIDECAR_SUFFIX, PoseResult


class SidecarError(ValueError):
    """A sidecar that cannot be read or does not validate."""


def find_sidecars(folder: Path, *, recursive: bool = True) -> list[Path]:
    """Every ``*.pose.json`` under *folder* (or directly in it), sorted by path. Symlinked files are skipped."""
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")
    pattern = f"**/*{SIDECAR_SUFFIX}" if recursive else f"*{SIDECAR_SUFFIX}"
    return sorted(p for p in root.glob(pattern) if p.is_file() and not p.is_symlink())


def load_sidecar(path: Path) -> PoseResult:
    """One sidecar, validated (including its ``puppet_version``)."""
    try:
        return PoseResult.model_validate_json(Path(path).read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise SidecarError(f"{path}: {exc}") from exc


def load_sidecars(folder: Path, *, recursive: bool = True, skip_invalid: bool = False) -> list[PoseResult]:
    """Every pose sidecar under *folder*, sorted by path. An unreadable or invalid sidecar raises
    :class:`SidecarError`, or is skipped when *skip_invalid* is set."""
    out: list[PoseResult] = []
    for path in find_sidecars(folder, recursive=recursive):
        try:
            out.append(load_sidecar(path))
        except SidecarError:
            if not skip_invalid:
                raise
    return out


def load_sidecars_with_paths(
    folder: Path, *, recursive: bool = True, skip_invalid: bool = False
) -> list[tuple[Path, PoseResult]]:
    """As :func:`load_sidecars`, each result paired with the sidecar file it came from."""
    out: list[tuple[Path, PoseResult]] = []
    for path in find_sidecars(folder, recursive=recursive):
        try:
            out.append((path, load_sidecar(path)))
        except SidecarError:
            if not skip_invalid:
                raise
    return out


def image_for_sidecar(sidecar: Path) -> Path | None:
    """The image a sidecar sits beside (``photo.pose.json`` -> ``photo.jpg``, any supported suffix and case), or
    ``None`` when there is none. With several (``photo.jpg`` and ``photo.png``) the first in sorted order, the one
    ``argus-puppet pose`` posed."""
    from argus_puppet.extract import IMAGE_SUFFIXES  # noqa: PLC0415 - extract pulls in the estimator stack

    sidecar = Path(sidecar)
    stem = sidecar.name[: -len(SIDECAR_SUFFIX)] if sidecar.name.endswith(SIDECAR_SUFFIX) else sidecar.stem
    found = sorted(
        p
        for p in sidecar.parent.glob(f"{glob.escape(stem)}.*")
        if p.stem == stem and p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()
    )
    return found[0] if found else None
