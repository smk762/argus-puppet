"""The pinned estimator models: what each file is, where it comes from, its hash and its licence.

Data only. Fetching and verifying the files belongs to the estimator; this table is what they are checked against,
and what a result's provenance (:class:`argus_puppet.models.ModelRef`) is built from. Only permissively licensed
weights may be listed (``tests/test_must_never.py`` enforces the licence allow-list and 64-hex hashes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from argus_puppet.models import ModelRef

Role = Literal["pose", "detector"]
# How the estimator feeds and decodes the file.
ModelFormat = Literal["rtmpose-simcc", "yolox-raw"]

_DWPOSE_CAVEAT = (
    "DWPose was trained on COCO-WholeBody (with UBody), whose dataset terms say research and non-commercial use; "
    "the weights themselves are published under Apache-2.0. Check the dataset terms before commercial use."
)
_DWPOSE_DATA = "COCO-WholeBody + UBody (distilled from a DWPose teacher)"
_YOLOX_CAVEAT = "Trained on COCO only; argus-puppet keeps only its 'person' class."
_YOLOX_DATA = "COCO 2017 detection"
_OPENMMLAB_ONNX = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk"
_YOLOX_RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"


@dataclass(frozen=True)
class PinnedModel:
    """One model file, pinned by its SHA-256."""

    name: str  # registry name, also the file's stem
    role: Role
    filename: str  # canonical file name in the model directory
    url: str  # where it is downloaded from (a zip when archive_sha256 is set)
    sha256: str  # of the .onnx file itself
    bytes: int  # size of the .onnx file
    license: str  # SPDX id of the code and weights
    training_data: str
    caveat: str
    input_hw: tuple[int, int]  # network input (height, width)
    format: ModelFormat
    archive_sha256: str | None = None  # when url is an archive: its SHA-256
    archive_member: str | None = None  # and the suffix of the member that is the .onnx


MODELS: dict[str, PinnedModel] = {
    m.name: m
    for m in (
        PinnedModel(
            name="dwpose-l-384x288",
            role="pose",
            filename="dwpose-l-384x288.onnx",
            url=f"{_OPENMMLAB_ONNX}/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip",
            sha256="8cfecfc2226d8e14b510c2fd28442226c518b4d60690535753187877411d4005",
            bytes=134_399_323,
            license="Apache-2.0",
            training_data=_DWPOSE_DATA,
            caveat=_DWPOSE_CAVEAT,
            input_hw=(384, 288),
            format="rtmpose-simcc",
            archive_sha256="b06f1bc598ddbdb5008a61f4a411aa11ed331370b4f7f438126d6d99e168d147",
            archive_member="end2end.onnx",
        ),
        PinnedModel(
            name="dwpose-m-256x192",
            role="pose",
            filename="dwpose-m-256x192.onnx",
            url=f"{_OPENMMLAB_ONNX}/rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip",
            sha256="94ca58fa2d6c4530b6957ac9548084ebc2fa27ed71e4e01f0b73844306ed01a6",
            bytes=71_792_372,
            license="Apache-2.0",
            training_data=_DWPOSE_DATA,
            caveat=_DWPOSE_CAVEAT,
            input_hw=(256, 192),
            format="rtmpose-simcc",
            archive_sha256="a4e4d56e9dc043e2171c73f5918f334509a47fde60182c1961370c7160334ac0",
            archive_member="end2end.onnx",
        ),
        PinnedModel(
            name="yolox_m",
            role="detector",
            filename="yolox_m.onnx",
            url=f"{_YOLOX_RELEASE}/yolox_m.onnx",
            sha256="21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f",
            bytes=101_259_744,
            license="Apache-2.0",
            training_data=_YOLOX_DATA,
            caveat=_YOLOX_CAVEAT,
            input_hw=(640, 640),
            format="yolox-raw",
        ),
        PinnedModel(
            name="yolox_nano",
            role="detector",
            filename="yolox_nano.onnx",
            url=f"{_YOLOX_RELEASE}/yolox_nano.onnx",
            sha256="c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d",
            bytes=3_659_407,
            license="Apache-2.0",
            training_data=_YOLOX_DATA,
            caveat=_YOLOX_CAVEAT,
            input_hw=(416, 416),
            format="yolox-raw",
        ),
    )
}

DEFAULT_POSE = "dwpose-l-384x288"
LIGHT_POSE = "dwpose-m-256x192"
DEFAULT_DETECTOR = "yolox_m"
LIGHT_DETECTOR = "yolox_nano"


def get(name: str, role: Role | None = None) -> PinnedModel:
    """The pinned model *name*; ``KeyError`` naming the choices if unknown or not of *role*."""
    model = MODELS.get(name)
    if model is None or (role is not None and model.role != role):
        choices = ", ".join(n for n, m in MODELS.items() if role is None or m.role == role)
        raise KeyError(f"unknown {role or 'model'} {name!r}: choose from {choices}")
    return model


def names(role: Role | None = None) -> tuple[str, ...]:
    """Registry names, optionally only those of *role*."""
    return tuple(n for n, m in MODELS.items() if role is None or m.role == role)


def model_ref(name: str) -> ModelRef:
    """The provenance record for pinned model *name*."""
    m = get(name)
    return ModelRef(name=m.name, sha256=m.sha256, license=m.license, bytes=m.bytes)
