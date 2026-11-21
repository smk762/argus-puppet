"""Settings, read from ``ARGUS_PUPPET_*``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from argus_puppet import registry

DEFAULT_PORT = 8110
DEFAULT_MODEL_DIR = Path("~/.cache/argus_puppet/models")
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_PIXELS = 40_000_000
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")

Device = Literal["auto", "cpu", "cuda"]


class ConfigError(ValueError):
    """A setting that cannot be used."""


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"ARGUS_PUPPET_{name}", default).strip()


def _int(name: str, default: int | None) -> int | None:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"ARGUS_PUPPET_{name} {raw!r} is not a whole number") from exc


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
    # A cap on ONNX Runtime's CUDA arena, in MB; None leaves it to ORT.
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
        )
