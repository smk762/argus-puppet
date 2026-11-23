"""Settings, read from ``ARGUS_PUPPET_*``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from argus_puppet import registry
from argus_puppet.models import MAX_PEOPLE

DEFAULT_PORT = 8220
DEFAULT_MODEL_DIR = Path("~/.cache/argus_puppet/models")
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_PIXELS = 40_000_000
# People kept per image, highest detector score first: bounds the pose batch (and the CUDA arena it grows).
DEFAULT_MAX_PEOPLE = 20
DEFAULT_DET_THRESHOLD = 0.3
DEFAULT_NMS_IOU = 0.45
# Images one POST /pose/scan runs through the estimator; past it the rest are skipped (reason "max_images").
DEFAULT_MAX_SCAN_IMAGES = 2000
# POST /pose calls in flight at once; past it a call is answered 503 + Retry-After instead of queued.
DEFAULT_POSE_CONCURRENCY = 2
# Pose results one POST /stats or /compare takes per set: the distance matrix is O(N^2) (5000 is a 100 MB matrix).
DEFAULT_MAX_SET_RESULTS = 5000
# Largest JSON body POST /stats and /compare read (/render reads at most 4 MiB). A one-person result is ~3.5 KB of
# JSON, so two sets of 5000 fit with room for crowds. A body may also carry at most this / 2048 people.
DEFAULT_MAX_JSON_BYTES = 128 * 1024 * 1024
# Seconds a request body may take to arrive (every POST); past it the call is answered 408 and its admission slot
# freed. 60 s carries a full 50 MiB upload at about 1 MB/s.
DEFAULT_BODY_TIMEOUT_S = 60.0
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
# The age check asks argus-curator (POST /faces): puppet's own variable first, then the suite-wide ARGUS_CURATOR_URL.
AGE_PEER = "curator"
AGE_URL_VAR = "ARGUS_PUPPET_AGE_URL"

Device = Literal["auto", "cpu", "cuda"]


class ConfigError(ValueError):
    """A setting that cannot be used."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"ARGUS_PUPPET_{name}", default).strip()


def _age_url() -> str:
    """argus-curator's base URL for the age check: ``ARGUS_PUPPET_AGE_URL``, else the suite-wide
    ``ARGUS_CURATOR_URL`` (environment, then the suite file), else unset. ``off`` leaves it unset whatever the suite
    file says. A value that is set but is not an absolute http(s) URL is a :class:`ConfigError` naming the variable
    (never the value), not "unset"."""
    from argus_cortex.peers import PeerRefused, peer_url  # noqa: PLC0415 - keeps `import argus_puppet.config` light

    try:
        return peer_url(AGE_PEER, AGE_URL_VAR) or ""
    except PeerRefused as exc:
        raise ConfigError(str(exc)) from None


def _int(name: str, default: int | None) -> int | None:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"ARGUS_PUPPET_{name} {raw!r} is not a whole number") from exc


def _float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"ARGUS_PUPPET_{name} {raw!r} is not a number") from exc


def _list(name: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in _env(name).split(",") if v.strip())


@dataclass(frozen=True)
class Settings:
    port: int = DEFAULT_PORT
    # Where the pinned model files live, under their registry file names.
    model_dir: Path = DEFAULT_MODEL_DIR.expanduser()
    pose_model: str = registry.DEFAULT_POSE
    detector_model: str = registry.DEFAULT_DETECTOR
    device: Device = "auto"
    # A cap on each ONNX Runtime CUDA session's arena, in MB; None: runtime.DEFAULT_GPU_MEM_LIMIT_MB.
    gpu_mem_limit_mb: int | None = None
    # The only root a caller-supplied path may fall under. None: routes that take a path refuse.
    data_root: Path | None = None
    read_only: bool = False
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    # Host names (no port) a page may be served from and still write to its own server. An IP address and
    # localhost need no entry (argus_cortex.server.cross_site_refuse).
    trusted_hosts: tuple[str, ...] = field(default_factory=tuple)
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_people: int = DEFAULT_MAX_PEOPLE
    det_threshold: float = DEFAULT_DET_THRESHOLD
    nms_iou: float = DEFAULT_NMS_IOU
    max_scan_images: int = DEFAULT_MAX_SCAN_IMAGES
    pose_concurrency: int = DEFAULT_POSE_CONCURRENCY
    max_set_results: int = DEFAULT_MAX_SET_RESULTS
    max_json_bytes: int = DEFAULT_MAX_JSON_BYTES
    body_timeout_s: float = DEFAULT_BODY_TIMEOUT_S
    # argus-curator's base URL for the age fail-safe (POST /faces?body=true): ARGUS_PUPPET_AGE_URL, else the suite's
    # ARGUS_CURATOR_URL. Empty: the age check cannot run, and nothing is ever sent anywhere. Only ever this configured
    # URL, never one a caller supplies.
    age_url: str = ""

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"ARGUS_PUPPET_PORT {self.port}: must be between 1 and 65535")
        for setting, name, role in (
            ("POSE_MODEL", self.pose_model, "pose"),
            ("DETECTOR_MODEL", self.detector_model, "detector"),
        ):
            try:
                registry.get(name, role)  # type: ignore[arg-type]
            except KeyError as exc:
                raise ConfigError(f"ARGUS_PUPPET_{setting}: {exc.args[0]}") from exc
        if self.device not in DEVICES:
            raise ConfigError(f"ARGUS_PUPPET_DEVICE {self.device!r}: choose from {', '.join(DEVICES)}")
        if self.gpu_mem_limit_mb is not None and self.gpu_mem_limit_mb < 256:
            raise ConfigError(f"ARGUS_PUPPET_GPU_MEM_LIMIT_MB {self.gpu_mem_limit_mb}: must be at least 256")
        if self.max_upload_bytes < 1024:
            raise ConfigError(f"ARGUS_PUPPET_MAX_UPLOAD_BYTES {self.max_upload_bytes}: must be at least 1024")
        if self.max_pixels < 64 * 64:
            raise ConfigError(f"ARGUS_PUPPET_MAX_PIXELS {self.max_pixels}: must be at least {64 * 64}")
        if not 1 <= self.max_people <= MAX_PEOPLE:
            raise ConfigError(f"ARGUS_PUPPET_MAX_PEOPLE {self.max_people}: must be between 1 and {MAX_PEOPLE}")
        for setting, value in (("DET_THRESHOLD", self.det_threshold), ("NMS_IOU", self.nms_iou)):
            if not 0.0 < value < 1.0:
                raise ConfigError(f"ARGUS_PUPPET_{setting} {value}: must be between 0 and 1 (exclusive)")
        if self.max_scan_images < 1:
            raise ConfigError(f"ARGUS_PUPPET_MAX_SCAN_IMAGES {self.max_scan_images}: must be at least 1")
        if not 1 <= self.pose_concurrency <= 64:
            raise ConfigError(f"ARGUS_PUPPET_POSE_CONCURRENCY {self.pose_concurrency}: must be between 1 and 64")
        if self.max_set_results < 1:
            raise ConfigError(f"ARGUS_PUPPET_MAX_SET_RESULTS {self.max_set_results}: must be at least 1")
        if self.max_json_bytes < 1024:
            raise ConfigError(f"ARGUS_PUPPET_MAX_JSON_BYTES {self.max_json_bytes}: must be at least 1024")
        if not 0.0 < self.body_timeout_s <= 3600.0:
            raise ConfigError(f"ARGUS_PUPPET_BODY_TIMEOUT_S {self.body_timeout_s}: must be above 0 and at most 3600")
        if self.age_url and not self.age_url.lower().startswith(("http://", "https://")):
            raise ConfigError(f"ARGUS_PUPPET_AGE_URL {self.age_url!r}: must be an http(s) URL")

    def model_path(self, name: str) -> Path:
        """Where pinned model *name*'s file lives (whether or not it is there yet)."""
        return self.model_dir / registry.get(name).filename

    @classmethod
    def from_env(cls) -> Settings:
        try:
            from argus_cortex.server import env_flag  # noqa: PLC0415 - needs the [server] extra's starlette

            read_only = env_flag("ARGUS_PUPPET_READ_ONLY")
        except ImportError:
            read_only = _env("READ_ONLY").lower() in ("1", "true", "yes", "on")
        model_dir = _env("MODEL_DIR")
        data_root = _env("DATA_ROOT")
        return cls(
            port=_int("PORT", DEFAULT_PORT),  # type: ignore[arg-type]
            model_dir=Path(model_dir).expanduser() if model_dir else DEFAULT_MODEL_DIR.expanduser(),
            pose_model=_env("POSE_MODEL") or registry.DEFAULT_POSE,
            detector_model=_env("DETECTOR_MODEL") or registry.DEFAULT_DETECTOR,
            device=(_env("DEVICE") or "auto").lower(),  # type: ignore[arg-type]
            gpu_mem_limit_mb=_int("GPU_MEM_LIMIT_MB", None),
            data_root=Path(data_root).expanduser().resolve() if data_root else None,
            read_only=read_only,
            cors_origins=tuple(o.rstrip("/") for o in _list("CORS_ORIGINS")),
            trusted_hosts=tuple(h.lower() for h in _list("TRUSTED_HOSTS")),
            max_upload_bytes=_int("MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),  # type: ignore[arg-type]
            max_pixels=_int("MAX_PIXELS", DEFAULT_MAX_PIXELS),  # type: ignore[arg-type]
            max_people=_int("MAX_PEOPLE", DEFAULT_MAX_PEOPLE),  # type: ignore[arg-type]
            det_threshold=_float("DET_THRESHOLD", DEFAULT_DET_THRESHOLD),
            nms_iou=_float("NMS_IOU", DEFAULT_NMS_IOU),
            max_scan_images=_int("MAX_SCAN_IMAGES", DEFAULT_MAX_SCAN_IMAGES),  # type: ignore[arg-type]
            pose_concurrency=_int("POSE_CONCURRENCY", DEFAULT_POSE_CONCURRENCY),  # type: ignore[arg-type]
            max_set_results=_int("MAX_SET_RESULTS", DEFAULT_MAX_SET_RESULTS),  # type: ignore[arg-type]
            max_json_bytes=_int("MAX_JSON_BYTES", DEFAULT_MAX_JSON_BYTES),  # type: ignore[arg-type]
            body_timeout_s=_float("BODY_TIMEOUT_S", DEFAULT_BODY_TIMEOUT_S),
            age_url=_age_url(),
        )
