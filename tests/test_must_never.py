"""Checks for rows of CLAUDE.md's "Must never happen" table."""

from __future__ import annotations

import base64
import binascii
import re
import subprocess
from pathlib import Path

from argus_puppet import registry

ROOT = Path(__file__).resolve().parent.parent

# --- only permissive, pinned estimator weights ------------------------------------------------------------------

PERMISSIVE = {"Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_every_model_is_permissive_and_pinned() -> None:
    """OpenPose's CMU weights are non-commercial and YOLO-pose is AGPL-3.0: neither may be listed. Every file is
    pinned by its SHA-256 (and its archive's, when it ships in one), so a result's provenance names exact bytes."""
    for name, m in registry.MODELS.items():
        assert m.license in PERMISSIVE, f"{name}: licence {m.license!r} is not in {sorted(PERMISSIVE)}"
        assert HEX64.match(m.sha256), f"{name}: sha256 is not 64 lower-case hex"
        if m.archive_sha256 is not None:
            assert HEX64.match(m.archive_sha256), f"{name}: archive_sha256 is not 64 lower-case hex"
            assert m.archive_member, f"{name}: an archive needs the member that is the model"


# --- no imagery or model file enters the repo -------------------------------------------------------------------

MEDIA = re.compile(
    r"\.(png|jpe?g|webp|gif|bmp|tiff?|avif|heic|ico|mp4|webm|mov|mkv|avi"
    r"|onnx|safetensors|ckpt|pt|pth|bin|npy|npz|tflite|engine|zip)$",
    re.I,
)
BASE64_RUN = re.compile(rb"[A-Za-z0-9+/]{120,}={0,2}")
IMAGE_MAGIC = (b"\x89PNG", b"\xff\xd8\xff", b"RIFF", b"GIF8", b"BM", b"II*\x00", b"MM\x00*", b"\x00\x00\x00")
TEXT_SUFFIXES = (".py", ".json", ".md", ".toml", ".yaml", ".yml", ".txt", ".html", ".js", ".ts", ".css", ".svg")


def tracked() -> list[str]:
    # NUL-separated and never quoted: a path with a space or a non-ASCII character comes back whole.
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.split("\0") if p and (ROOT / p).is_file()]


def test_no_image_video_or_model_file_is_tracked() -> None:
    """Tests draw their people in code and models are fetched by hash, so no such file has a reason to be in the
    repo, and one that is cannot be reviewed in a diff (or may carry a real person or a non-permissive licence)."""
    assert [p for p in tracked() if MEDIA.search(p)] == []


def test_no_image_is_hidden_in_a_base64_literal() -> None:
    """Nor is an image pasted into source as base64."""
    found = []
    for path in tracked():
        if not path.endswith(TEXT_SUFFIXES):
            continue
        for run in BASE64_RUN.findall((ROOT / path).read_bytes()):
            try:
                head = base64.b64decode(run[:64] + b"=" * (-len(run[:64]) % 4), validate=True)
            except (binascii.Error, ValueError):
                continue
            if head.startswith(IMAGE_MAGIC):
                found.append(path)
    assert found == []
