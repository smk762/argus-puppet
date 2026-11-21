# Models and licences

Puppet is a **top-down** estimator: a person detector finds each person's box, then a pose model places 133
COCO-WholeBody keypoints (17 body joints, 6 foot points, 68 face landmarks, 21 points per hand) inside each box. Both
are ONNX files run by ONNX Runtime. Four files are pinned in `src/argus_puppet/registry.py`: a default and a light
model for each role.

## The pinned models

| Name | Role | Input (h × w) | File size | Default |
|---|---|---|---|---|
| `dwpose-l-384x288` | pose | 384 × 288 | 134,399,323 bytes (134.4 MB) | yes |
| `dwpose-m-256x192` | pose | 256 × 192 | 71,792,372 bytes (71.8 MB) | light pair |
| `yolox_m` | detector | 640 × 640 | 101,259,744 bytes (101.3 MB) | yes |
| `yolox_nano` | detector | 416 × 416 | 3,659,407 bytes (3.7 MB) | light pair |

| Name | SHA-256 of the `.onnx` file |
|---|---|
| `dwpose-l-384x288` | `8cfecfc2226d8e14b510c2fd28442226c518b4d60690535753187877411d4005` |
| `dwpose-m-256x192` | `94ca58fa2d6c4530b6957ac9548084ebc2fa27ed71e4e01f0b73844306ed01a6` |
| `yolox_m` | `21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f` |
| `yolox_nano` | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` |

| Name | Source | Licence (code and weights) | Training data |
|---|---|---|---|
| `dwpose-l-384x288` | OpenMMLab's RTMPose ONNX SDK zip `rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip` from `download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/`; the file is the zip's `end2end.onnx`. Zip SHA-256 `b06f1bc598ddbdb5008a61f4a411aa11ed331370b4f7f438126d6d99e168d147` | Apache-2.0 | COCO-WholeBody + UBody, distilled from a DWPose teacher |
| `dwpose-m-256x192` | The same location, `rtmpose-m_simcc-ucoco_dw-ucoco_270e-256x192-c8b76419_20230728.zip`, member `end2end.onnx`. Zip SHA-256 `a4e4d56e9dc043e2171c73f5918f334509a47fde60182c1961370c7160334ac0` | Apache-2.0 | COCO-WholeBody + UBody, distilled from a DWPose teacher |
| `yolox_m` | Megvii's YOLOX release 0.1.1rc0: `github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.onnx` | Apache-2.0 | COCO 2017 detection; puppet keeps only the `person` class |
| `yolox_nano` | The same release, `yolox_nano.onnx` | Apache-2.0 | COCO 2017 detection; puppet keeps only the `person` class |

`argus-puppet models list` prints the same table with each file's state on this host. DWPose is an RTMPose model
trained by distillation for whole-body keypoints; the pose files here are its RTMPose-l and RTMPose-m students.

## Pinning and verification

A pinned model is a name bound to one exact file: its size and SHA-256.

- **Before a file is ever loaded**, puppet checks its size and then its SHA-256 against the pin. A file that is
  missing, the wrong size, or hashes to anything else is refused with an error naming the fix; it is never run. The
  hash is computed once per process and again only if the file changes on disk.
- **Every result records what ran.** `PoseResult.estimator` names the detector and the pose model with their
  SHA-256, licence and size, together with the ONNX Runtime provider, the puppet version, and the detection
  threshold and NMS IoU. `PoseSetStats.estimator` and `PoseSetComparison.estimator` carry the same record when every
  pose in the set shares it, so a score can always be traced to the exact weights behind it.
- **Downloads are verified before they are written.** `models fetch` downloads only from OpenMMLab's and GitHub's
  hosts (redirects to other hosts are refused). For a zip, the archive must match its pinned hash; only the one
  named member is read, and only if its declared size is the pinned size. The `.onnx` bytes must then match the
  pinned size and hash. The file is written to a temporary name and moved into place, so a failed or interrupted
  download leaves nothing behind.

Changing a pin is a code change to `registry.py`, and a test (`tests/test_must_never.py`) checks every entry has a
licence on the allow-list and a well-formed hash.

## Managing the files

The files live in `ARGUS_PUPPET_MODEL_DIR` (default `~/.cache/argus_puppet/models`; `/models` in the container
image), under the names in the first table.

```bash
argus-puppet models fetch                     # download every pinned model that is missing or does not verify
argus-puppet models fetch yolox_nano          # just one; --force downloads even when a verified file is present
argus-puppet models verify                    # check size and SHA-256 of each; exit 1 if any is missing or wrong
argus-puppet models list                      # role, licence, size, present, verified (--json, --no-verify)
```

The container image ships no weights: see [deploy.md](deploy.md#models-are-not-in-the-image). `GET /health` reports
whether each file is present and verified.

## Choosing models

Pick models by registry name with `ARGUS_PUPPET_POSE_MODEL` and `ARGUS_PUPPET_DETECTOR_MODEL`, or per run with
`argus-puppet pose --pose-model ... --detector ...`. An unknown name, or a name of the wrong role, is refused at
startup.

| Pair | Use |
|---|---|
| `dwpose-l-384x288` + `yolox_m` (default) | GPU hosts, and any run where keypoint accuracy matters more than time |
| `dwpose-m-256x192` + `yolox_nano` (light) | Small jobs on the CPU, CI, and laptops |

The light detector is about 27 times smaller than the default and, by Megvii's published COCO figures, much less
accurate: expect more missed people, small ones especially. The light pose model works on a smaller crop, so fine
detail (fingers, face landmarks) has fewer pixels to go on. Do not mix pairs within a set you will measure or
compare: distances between poses from different models include the models' differences. `PoseSetStats.estimator` is
left empty when a set's poses came from more than one estimator, which is a signal to re-pose.

### Measured cost

These figures are single-host measurements, not guarantees. Expect different numbers on other hardware.

| Setting | Measured |
|---|---|
| CPU, 4 threads, pose model only | `dwpose-l-384x288` about 54 ms per person; `dwpose-m-256x192` about 17 ms per person |
| GPU (RTX 3090), default pair | about 33 ms per image and about 750 MB of VRAM |

The pose model runs every person of an image in one batch, so its cost grows with the number of people (at most
`ARGUS_PUPPET_MAX_PEOPLE`, default 20). The detector runs once per image.

On CUDA, each ONNX Runtime session's memory arena is capped (1024 MB by default, `ARGUS_PUPPET_GPU_MEM_LIMIT_MB` to
change it) and grows only by what an allocation asks for. The argus-iris tenant manifest (`gpu-tenant.yaml`)
reserves 1600 MB for a lease, of which 240 MB is weights, leaving headroom over the measured figure for large
batches of people. See [deploy.md](deploy.md#gpu-sharing-argus-iris).

## Why no rtmlib, mmpose or OpenCV

The estimator (`src/argus_puppet/estimator.py`) uses only numpy, ONNX Runtime and Pillow. mmpose would bring PyTorch
and the OpenMMLab stack, rtmlib brings OpenCV, and Ultralytics is AGPL-3.0; none of them is needed to run two ONNX
files. Puppet instead reimplements the pre- and post-processing those libraries use: the detector's letterbox and
raw-head decoding with non-maximum suppression, and the pose model's box padding, affine crop, normalisation and
SimCC decoding, including their resampling conventions. Given the same boxes, its keypoints match rtmlib's to
0.000 px. The result is a small install and container image, and an estimator whose every step is in this repo.

## The licence position

This section records what the project checked and why it chose these models. It is not legal advice, and it draws
no conclusion about any particular use.

**What is published under which licence.** The code and weights of all four pinned models are published under
Apache-2.0: DWPose's RTMPose students by OpenMMLab (mmpose), and YOLOX by Megvii. Puppet's own code is MIT-licensed.

**What was excluded on licence grounds.**

- OpenPose's CMU weights, whose licence is non-commercial.
- Ultralytics YOLO models (including YOLO-pose), which are AGPL-3.0. The suite reaches AGPL code only as an external
  image, never as an import.

Only weights on a permissive allow-list may be added to the registry, and a test enforces it.

**The training-data caveat.** The registry records this against both DWPose models:

> DWPose was trained on COCO-WholeBody (with UBody), whose dataset terms say research and non-commercial use; the
> weights themselves are published under Apache-2.0. Check the dataset terms before commercial use.

This is not peculiar to DWPose: every whole-body 133-keypoint model the project found was trained on
COCO-WholeBody. Whether a model's licence or its training data's terms govern a given use is a question the
project does not answer. **Anyone with a commercial use should make their own assessment.**

**The detector.** The detector is Megvii's own YOLOX, trained on COCO detection only. It was chosen over the
Human-Art-trained YOLOX detectors that the RTMPose tooling offers, because Human-Art's terms are non-commercial.

**Images and output.** The licences above cover the models. They say nothing about the images you pose. Keypoints
and control images are derived from their source photos; the pose library keeps each pose's source name and
SHA-256 so that a generator can apply its input checks to the source ([conditioning.md](conditioning.md)).
