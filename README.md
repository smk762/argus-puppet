# argus-puppet

Body pose for the Argus suite. Puppet finds every person in an image, places 133 whole-body keypoints on each
(body, feet, hands and face), and turns those keypoints into numbers other packages can act on: how varied a set of
poses is, which poses dominate it, whether outputs sit closer to training frames' poses than those frames sit to
each other, and skeleton control images
that drive a pose ControlNet. It is a library, a CLI and an HTTP service on port 8220. It has no Studio stage of its
own: it is built for argus-proof, argus-curator, argus-darkroom and the others to call, as they call argus-retina's
detectors. None of them does yet.

Part of the [Argus suite](https://github.com/smk762?tab=repositories&q=argus). Shared code (wire-schema versioning,
write guards, GPU leases) comes from argus-cortex.

## What you can do with it

| Stage | What it produces | CLI | HTTP |
|---|---|---|---|
| **Extract** | A `PoseResult` per image: every person's box, 133 keypoints with scores, the primary subject, and the SHA-256 of the models that ran. Saved as a `<stem>.pose.json` sidecar beside the image | `pose` | `POST /pose`, `POST /pose/scan` |
| **Measure** | `PoseSetStats`: a diversity score, clusters of similar poses, and coverage counts by stance, arms and facing | `stats` | `POST /stats` |
| **Compare** | `PoseSetComparison`: each output's nearest reference (training) pose, the share of close pose matches against the training set's own baseline, and how far the pose mix drifted | `compare` | `POST /compare` |
| **Condition** | OpenPose-convention skeleton PNGs for pose ControlNets, and a pose library you can sample diverse poses from | `render`, `library build`, `library sample` | `POST /render` |

## Install

Python 3.11 or newer. Pick one ONNX Runtime extra, and add the interfaces you need:

```bash
pip install "argus-puppet[cpu,cli]"          # library + CLI on the CPU
pip install "argus-puppet[gpu,cli,server]"   # ONNX Runtime CUDA, the CLI and the HTTP service
```

| Extra | Brings |
|---|---|
| `cpu` | `onnxruntime` (CPU) |
| `gpu` | `onnxruntime-gpu` and argus-cortex's GPU lease client. Install `cpu` or `gpu`, not both: they provide the same module |
| `cli` | the `argus-puppet` command (Typer, Rich) and the HTTP client `models fetch` downloads through |
| `server` | FastAPI and uvicorn for the HTTP service. `argus-puppet serve` also needs `cli`; the app alone is `argus_puppet.server.app:app` |

Without `cpu` or `gpu` the package imports, reads sidecars and computes metrics, but cannot extract poses. Container
images (CPU and CUDA) are published to GHCR: see [docs/deploy.md](docs/deploy.md).

## Quickstart

Every command below was run against the real CLI on the CPU. `~/datasets/example` stands for any folder of `.jpg`,
`.jpeg`, `.png` or `.webp` photos of adults.

```bash
# 1. Download the four pinned models (about 310 MB) into ~/.cache/argus_puppet/models, verified by SHA-256
argus-puppet models fetch
argus-puppet models list                      # present and verified?

# 2. Pose a folder: writes photo.pose.json beside each photo.jpg (existing sidecars are kept; --overwrite redoes them)
argus-puppet pose ~/datasets/example --recursive

# 3. How varied are the poses? (reads the sidecars; --json for the PoseSetStats document)
argus-puppet stats ~/datasets/example
argus-puppet stats ~/datasets/example --min-diversity 0.3   # exit 1 below the threshold, for scripts

# 4. Compare a LoRA's outputs with its training set (both posed first, or pass --extract)
argus-puppet compare ~/datasets/outputs ~/datasets/example --extract

# 5. Draw ControlNet skeletons at the generation size: control/<stem>.openpose.png
argus-puppet render ~/datasets/example --out control --width 1024 --height 1024 --stroke-scale auto

# 6. Build a pose library, then pick 8 mutually distant poses and render them
argus-puppet library build ~/datasets/example --out poses.library.json --created-from "CC0 photos of adults"
argus-puppet library sample poses.library.json -n 8 --render-dir batch --size 1024x1024
```

Re-running `pose` skips images that already have a sidecar, so it is cheap to repeat. A set with fewer than two
usable poses scores diversity 0; measure sets of tens of images or more, and read `usable` and `comparable_share`
before trusting the score (see [docs/metrics.md](docs/metrics.md#suggested-gate-for-proof)). In `compare`, a close
pose match means "the same pose", not "a copied frame": most same-stance poses have a near twin in a large training
set. Read the per-output match as a pointer for an image-level check, and `close_share` against the training set's
own `baseline_close_share` (`close_excess`) as the run-level signal
([docs/metrics.md](docs/metrics.md#what-a-close-pose-match-means)). For a faster, lighter model pair on the
CPU, add `--pose-model dwpose-m-256x192 --detector yolox_nano` (see
[docs/models.md](docs/models.md#choosing-models)).

The same from Python:

```python
from pathlib import Path

from argus_puppet.compare import compare_pose_sets
from argus_puppet.extract import pose_image
from argus_puppet.metrics import pose_set_stats
from argus_puppet.render import render_png
from argus_puppet.models import RenderOptions
from argus_puppet.sidecars import load_sidecars

result = pose_image("photo.jpg")                       # a PoseResult; writes nothing
stats = pose_set_stats(load_sidecars(Path("~/datasets/example").expanduser()))
print(stats.diversity, stats.largest_cluster_share)
comparison = compare_pose_sets(load_sidecars(Path("outputs")), load_sidecars(Path("training")))
png = render_png(result, RenderOptions(width=1024, height=1024, stroke_scale="auto"))   # PNG bytes
```

### The service

```bash
ARGUS_PUPPET_DATA_ROOT=~/datasets argus-puppet serve     # binds 127.0.0.1:8220 by default
curl -s -X POST -H 'content-type: image/jpeg' --data-binary @photo.jpg 'http://127.0.0.1:8220/pose?name=photo.jpg'
```

There is **no authentication**: anyone who can reach the port can use the service. Keep it on loopback or behind a
proxy you control. Every route, its limits and the security model are in [docs/server.md](docs/server.md).

## The sidecar

`argus-puppet pose` and `POST /pose/scan` write `photo.pose.json` beside `photo.jpg` (the `.pose.json` suffix keeps
it apart from other packages' `<stem>.json` sidecars). It is a `PoseResult`, the same document `POST /pose` returns:

```jsonc
{
  "puppet_version": "1.1",
  "image": "woman_yoga.jpg",
  "image_sha256": "…",
  "width": 480, "height": 720,
  "layout": "coco_wholebody_133",
  "primary": 0,
  "people": [
    {"bbox": [151.2, 38.6, 190.4, 662.0], "det_score": 0.9, "primary": true,
     "keypoints": [[243.1, 101.7, 0.95], …],          // exactly 133 (x, y, score) triples
     "part_scores": {"body": 0.78, "feet": 0.45, "face": 0.99, "left_hand": 0.69, "right_hand": 0.71},
     "labels": {"stance": "standing", "arms": "down", "facing": "front"}}  // since 1.1; absent in 1.0 sidecars
  ],
  "estimator": {"name": "topdown", "provider": "CPUExecutionProvider",
                "detector": {"name": "yolox_m", "sha256": "21ff6c…", "license": "Apache-2.0", "bytes": 101259744},
                "pose": {"name": "dwpose-l-384x288", "sha256": "8cfecf…", "license": "Apache-2.0", "bytes": 134399323},
                "package_version": "0.1.0", "det_threshold": 0.3, "nms_iou": 0.45}
}
```

Keypoints are `(x, y, score)` in the image's pixels after EXIF orientation; a point the model could not place has
score 0. Boxes are `[x, y, w, h]`. People are listed largest first, and `primary` picks the main subject by the same
rule argus-curator uses for its primary face. Each person's `labels` are the coarse stance, arms and facing
of [docs/metrics.md](docs/metrics.md#labels). The authoritative field list is `src/argus_puppet/models.py`, rendered
to [`schema/puppet-wire.schema.json`](schema/puppet-wire.schema.json) (`argus-puppet schema`); CI fails if the two
drift. A 1.x reader ignores unknown fields, so it accepts any later 1.y document.

## Documentation

| Page | What it covers |
|---|---|
| [docs/index.md](docs/index.md) | Overview, the pipeline, which page you need, privacy and the age fail-safe |
| [docs/cli.md](docs/cli.md) | Every command and option (generated from the CLI) |
| [docs/server.md](docs/server.md) | HTTP routes, request and response shapes, status codes, limits, security model |
| [docs/pose-vector.md](docs/pose-vector.md) | The normalised pose vector and the mirror-invariant pose distance |
| [docs/metrics.md](docs/metrics.md) | Diversity, clustering, coverage labels, set-vs-set comparison and their limits |
| [docs/conditioning.md](docs/conditioning.md) | Skeleton control images, the pose library and diverse sampling |
| [docs/models.md](docs/models.md) | The pinned models, verification, cost, and the licence position |
| [docs/deploy.md](docs/deploy.md) | Container images, environment variables, the data root, GPU sharing |

## Models and licences

Puppet runs a YOLOX person detector and a DWPose whole-body pose model, both published under Apache-2.0, on ONNX
Runtime. Every model file is pinned by SHA-256: a file that does not match is refused, and the hashes of the models
that ran are recorded in every result. OpenPose's CMU weights (non-commercial) and Ultralytics YOLO (AGPL-3.0) were
excluded on licence grounds. DWPose, like every whole-body 133-keypoint model the project found, was trained on
COCO-WholeBody (with UBody), whose dataset terms say research and non-commercial use; anyone with a commercial use
should make their own assessment. Details in [docs/models.md](docs/models.md). Puppet's own code is MIT-licensed
([LICENSE](LICENSE)).

## Develop

```bash
make install   # uv venv + editable install with the dev, server, cli and cpu extras
make test      # pytest
make lint      # ruff check + format check (pinned ruff)
make format    # ruff format + fix
make schema    # regenerate schema/puppet-wire.schema.json after changing models.py
make docs      # regenerate docs/cli.md after changing a command
```

Tests never load a model or touch the network. Fixtures and demos use images of adults only.

## CI / Release

- **CI** runs via the shared [`argus-ci`](https://github.com/smk762/argus-ci) reusable workflow: lint, tests,
  `argus-puppet schema --check`, the suite rules, and a docs check (no private addresses or paths in `README.md` and
  `docs/`, and `docs/cli.md` matching the CLI).
- **Release** publishes to PyPI (OIDC trusted publishing) and the CPU and CUDA images to GHCR on `v*` tags.
- Versioning is derived from git tags via `hatch-vcs`: tag `vX.Y.Z` to cut a release. Never hand-edit a version.

This repo was scaffolded from [`argus-pkg-template`](https://github.com/smk762/argus-pkg-template).
Run `copier update` to pull template changes (CI, release, tooling).
