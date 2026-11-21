"""Fetch and verify the pinned model files (``argus-puppet models list|fetch|verify``).

Downloads go through :func:`argus_cortex.net.fetch` (SSRF-checked, pinned connection, capped body, timeouts), and
only from the registry's two upstream hosts and the hosts they redirect to (:data:`ALLOWED_HOSTS`): fetch resolves
every hop through :func:`_allowed_resolver`, which refuses any other host before it is connected to. The hash pins
below are the real guarantee; the host list keeps the download from going anywhere else. A file is written only
after its bytes are verified:

* a zip (DWPose ships in OpenMMLab's SDK zips) must hash to the pinned ``archive_sha256``; then exactly one member
  named ``archive_member`` is read (never ``extractall``; a member with an absolute path or ``..`` is refused, and
  one whose declared size is not the pinned size is refused before it is inflated);
* the ``.onnx`` bytes must be the pinned size and hash to the pinned ``sha256``;
* the file is written beside its destination and moved into place with ``os.replace``.

Tests pass a fake *fetcher* (``url -> bytes``), so no test touches the network.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import socket
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import structlog

from argus_puppet import registry, runtime
from argus_puppet.config import Settings

log = structlog.get_logger(__name__)

# The registry's upstreams and where they redirect (GitHub release assets land on its object stores).
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "download.openmmlab.com",
        "github.com",
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)
FETCH_TIMEOUT_S = 60.0  # per read
FETCH_DEADLINE_S = 1800.0  # the whole download, across redirects

Fetcher = Callable[[str, int], bytes]  # (url, max_bytes) -> body


class WeightsError(RuntimeError):
    """A model file could not be fetched or did not verify."""


@dataclass(frozen=True)
class ModelStatus:
    """One pinned model, and whether its file is in the model directory."""

    name: str
    role: str
    license: str
    bytes: int
    path: Path
    present: bool
    verified: bool | None  # None: not checked


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _system_resolve(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(i[4][0]) for i in infos))


def _allowed_resolver(refused: list[str]) -> Callable[[str, int], list[str]]:
    """A resolver for :func:`argus_cortex.net.fetch` (its *resolve* hook, called for every hop before connecting) that
    refuses a host off :data:`ALLOWED_HOSTS`, noting it in *refused*. fetch still checks the addresses it returns."""

    def resolve(host: str, port: int) -> list[str]:
        if host.lower().rstrip(".") not in ALLOWED_HOSTS:
            refused.append(host)
            raise OSError(f"{host} is not an allowed model host")
        return _system_resolve(host, port)

    return resolve


def net_fetcher(url: str, max_bytes: int) -> bytes:
    """Download *url* with :func:`argus_cortex.net.fetch`, refusing any hop off :data:`ALLOWED_HOSTS` before it is
    followed."""
    try:
        from argus_cortex.net import UrlRefused, fetch  # noqa: PLC0415 - httpx is behind cortex's [remote] extra
    except ImportError as exc:  # pragma: no cover - only without httpx
        raise WeightsError("downloading needs httpx: pip install 'argus-cortex[remote]'") from exc
    if _host(url) not in ALLOWED_HOSTS:
        raise WeightsError(f"refusing to download from {_host(url) or url!r}: not an allowed model host")
    refused: list[str] = []
    try:
        reply = fetch(
            url,
            timeout=FETCH_TIMEOUT_S,
            deadline=FETCH_DEADLINE_S,
            max_bytes=max_bytes,
            resolve=_allowed_resolver(refused),
        )
    except UrlRefused as exc:
        if refused:
            raise WeightsError(
                f"the download of {url} was redirected to {refused[-1]!r}, not an allowed model host"
            ) from exc
        raise
    return reply.raise_for_status().body


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_member(archive: bytes, member: str, expected_bytes: int) -> bytes:
    """The one member of zip *archive* named *member* (its last path component), read without ``extractall``.

    Refuses: no match or several, an absolute or ``..`` member path, a directory, or a declared size other than
    *expected_bytes* (checked before inflating, so a zip bomb never expands)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise WeightsError(f"not a zip archive: {exc}") from exc
    with zf:
        matches = []
        for info in zf.infolist():
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if path.name != member:
                continue
            if path.is_absolute() or ".." in path.parts or info.is_dir():
                raise WeightsError(f"refusing archive member {info.filename!r}")
            matches.append(info)
        if len(matches) != 1:
            raise WeightsError(f"expected exactly one {member!r} in the archive, found {len(matches)}")
        info = matches[0]
        if info.file_size != expected_bytes:
            raise WeightsError(f"{info.filename} is {info.file_size} bytes, not the pinned {expected_bytes}")
        with zf.open(info) as f:
            data = f.read(expected_bytes + 1)
    if len(data) != expected_bytes:
        raise WeightsError(f"{info.filename} inflated to {len(data)} bytes, not the pinned {expected_bytes}")
    return data


def write_atomic(path: Path, data: bytes) -> None:
    """Write *data* to *path* via a temp file in the same directory and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):  # best effort: the temp file may already be gone
            os.unlink(tmp)
        raise


def status(settings: Settings, names: Iterable[str] | None = None, *, verify: bool = True) -> list[ModelStatus]:
    """Each pinned model's file status (*verify* hashes the files that are present)."""
    out = []
    for name in names or registry.names():
        m = registry.get(name)
        path = settings.model_path(name)
        present = path.is_file()
        verified = runtime.model_verified(settings, name) if (verify and present) else (False if verify else None)
        out.append(ModelStatus(m.name, m.role, m.license, m.bytes, path, present, verified))
    return out


def fetch_model(settings: Settings, name: str, *, fetcher: Fetcher | None = None, force: bool = False) -> Path:
    """Download pinned model *name* into the model directory, verified; a no-op if it is already there and verified
    (unless *force*). Returns its path. Raises :class:`WeightsError`."""
    m = registry.get(name)
    path = settings.model_path(name)
    if not force and runtime.model_verified(settings, name):
        log.info("model_present", model=name, path=str(path))
        return path
    fetch = fetcher or net_fetcher
    limit = (m.bytes * 2 if m.archive_sha256 else m.bytes) + (1 << 20)
    log.info("model_fetch", model=name, url=m.url)
    try:
        body = fetch(m.url, limit)
    except WeightsError:
        raise
    except Exception as exc:  # noqa: BLE001 - net refusals, HTTP errors, timeouts: all "could not fetch"
        raise WeightsError(f"could not download {name} from {_host(m.url)}: {exc}") from exc
    if m.archive_sha256 is not None:
        if _sha256(body) != m.archive_sha256:
            raise WeightsError(f"the archive for {name} does not match its pinned SHA-256")
        body = extract_member(body, m.archive_member or "", m.bytes)
    if len(body) != m.bytes:
        raise WeightsError(f"{name} is {len(body)} bytes, not the pinned {m.bytes}")
    if _sha256(body) != m.sha256:
        raise WeightsError(f"{name} does not match its pinned SHA-256")
    try:
        write_atomic(path, body)
    except OSError as exc:  # a read-only model directory, a full disk
        raise WeightsError(f"could not write {name} to {path}: {exc.strerror or exc}") from exc
    log.info("model_written", model=name, path=str(path))
    return path
