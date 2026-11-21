# The HTTP service

`argus-puppet serve` runs a FastAPI app on port **8220** (`ARGUS_PUPPET_PORT`, or `--port`). It binds `127.0.0.1`
unless `--host` says otherwise; the container image binds `0.0.0.0` inside the container and `compose.yaml`
publishes it on the host's loopback (see [deploy.md](deploy.md)). The app is `argus_puppet.server.app:app`, and
`create_app(settings)` builds one from explicit settings. FastAPI's own `/docs` and `/openapi.json` are served too.

Every request and response body is a type from `src/argus_puppet/models.py`, the wire contract. Its field
descriptions are authoritative and are rendered to `schema/puppet-wire.schema.json`. Top-level documents carry
`puppet_version` (currently `"1.1"`); a 1.x reader ignores fields it does not know, and every 1.x reader accepts a
1.0 document. 1.1 added `PersonPose.labels` (see [`POST /pose`](#post-pose)).

## Routes

| Route | Body in | Body out | Writes |
|---|---|---|---|
| `GET /health` | none | `Health` | nothing |
| `POST /pose` | one image (raw bytes or multipart) | `PoseResult` | nothing |
| `POST /pose/scan` | `ScanRequest` (JSON) | `ScanSummary` | `<stem>.pose.json` sidecars under the data root |
| `POST /stats` | `StatsRequest` (JSON) | `PoseSetStats` | nothing |
| `POST /compare` | `CompareRequest` (JSON) | `PoseSetComparison` | nothing |
| `POST /render` | `RenderRequest` (JSON) | `image/png` | nothing |

Errors are FastAPI's `{"detail": ...}`: a string, or for validation errors (422) a list of
`{type, loc, msg}` entries. Puppet's JSON routes (`/pose/scan`, `/stats`, `/compare`, `/render`) never echo the
input back in a 422.

### `GET /health`

Whether each pinned model file is present and verified, which models requests use, where they run, free VRAM and
read-only mode.

```bash
curl -s http://<PUPPET_HOST>:8220/health
```

```json
{"puppet_version": "1.1", "version": "0.2.0", "read_only": false,
 "models": {"dwpose-l-384x288": true, "dwpose-m-256x192": true, "yolox_m": true, "yolox_nano": true},
 "pose_model": "dwpose-l-384x288", "detector_model": "yolox_m",
 "provider": null, "device": "cpu", "gpu_coordinator": "none", "free_vram_mb": null}
```

| Field | Meaning |
|---|---|
| `models` | Registry name to whether its file is present and matches its pinned size and SHA-256. The first call hashes the files (about 310 MB), so it can take a few seconds; later calls use a cache |
| `pose_model`, `detector_model` | The models `POST /pose` and `/pose/scan` run (`ARGUS_PUPPET_POSE_MODEL`, `ARGUS_PUPPET_DETECTOR_MODEL`) |
| `provider` | The ONNX Runtime execution provider, or `null` until the first pose request loads a model |
| `device` | `cuda` when ONNX Runtime has the CUDA provider and the device setting allows it, else `cpu` |
| `gpu_coordinator` | `none`, `lease` or `broker` (`ARGUS_GPU_COORDINATOR`), or `invalid` for an unknown value |
| `free_vram_mb` | Free GPU memory on CUDA when measurable, else `null` |

Always 200.

### `POST /pose`

Pose one image held in memory. Nothing is written to disk.

Send the image either as the raw body with `Content-Type: application/octet-stream` or `image/*` (or no content
type), or as `multipart/form-data`, where the first file part is used.

| Query | Meaning |
|---|---|
| `name` | The image's name in the result (`PoseResult.image`). A file name, not a path. Default: the multipart file name, else `upload` |
| `face` | A primary-face hint `x,y,w,h` in pixels, for example argus-curator's primary face. The person whose body best matches that face becomes the primary subject; with no match the usual rule decides (see [the primary subject](#the-primary-subject)) |

| Header | Meaning |
|---|---|
| `X-Argus-Priority` | The GPU lease priority for this call: `realtime`, `interactive`, `batch` or `background`. Default `interactive`. `realtime` is lowered to `interactive`, the highest class puppet leases at. Has no effect on the CPU, where no lease is taken |

```bash
curl -s -X POST -H 'content-type: image/jpeg' --data-binary @photo.jpg \
  'http://<PUPPET_HOST>:8220/pose?name=photo.jpg'
curl -s -X POST -F file=@photo.jpg 'http://<PUPPET_HOST>:8220/pose'
curl -s -X POST -H 'X-Argus-Priority: background' -H 'content-type: image/jpeg' --data-binary @photo.jpg \
  'http://<PUPPET_HOST>:8220/pose?face=212,80,96,110'
```

The answer is a `PoseResult`: the same document as a sidecar. The README's [sidecar section](https://github.com/smk762/argus-puppet#the-sidecar)
shows one; every field is described in `models.py`. At most `ARGUS_PUPPET_MAX_PEOPLE` people (default 20) are kept
per image, those with the highest detector scores; the rest are dropped, not refused.

Every person carries `labels`, the coarse `stance`, `arms` and `facing` of their keypoints
([metrics.md](metrics.md#labels); `pose_labels` at the default joint score of 0.3), so a caller needs no second
`/stats` call for them. `labels` is new in wire 1.1: a 1.0 document (an older sidecar) has none, and readers must
treat it as optional. It is informational: `/stats`, `/compare` and library builds recompute labels from the
keypoints with their own `params` and never read the stored ones.

```json
{"bbox": [151.2, 38.6, 190.4, 662.0], "det_score": 0.9, "primary": true,
 "keypoints": [[243.1, 101.7, 0.95], "... 133 in all"],
 "part_scores": {"body": 0.78, "feet": 0.45, "face": 0.99, "left_hand": 0.69, "right_hand": 0.71},
 "labels": {"stance": "standing", "arms": "down", "facing": "front"}}
```

| Status | When |
|---|---|
| 200 | Posed. An image with nobody in it is still 200, with `people: []` and `primary: null` |
| 400 | The body is empty; the multipart form has no file; `face` is not four finite numbers with `w, h > 0`; `name` is not a plain file name; `X-Argus-Priority` is not a priority class; the image could not be decoded |
| 408 | The body did not arrive within `ARGUS_PUPPET_BODY_TIMEOUT_S` (default 60 s) |
| 413 | The image is larger than `ARGUS_PUPPET_MAX_UPLOAD_BYTES` (default 50 MiB) or `ARGUS_PUPPET_MAX_PIXELS` (default 40,000,000 pixels) |
| 415 | The content type is neither raw nor multipart, or the bytes are not an image |
| 503 | Busy (see [concurrency](#concurrency-and-503)), a model is missing or does not verify, ONNX Runtime is not installed, or the GPU lease was not granted |

### `POST /pose/scan`

Pose a folder under the data root and write a `<stem>.pose.json` sidecar beside each image. This is the one route
that changes files.

```json
{"path": "shoots/2026-10", "recursive": false, "overwrite": false}
```

| Field | Default | Meaning |
|---|---|---|
| `path` | required | A folder, or one image, relative to `ARGUS_PUPPET_DATA_ROOT` |
| `recursive` | `false` | Include subfolders |
| `overwrite` | `false` | Re-pose images that already have a sidecar |

```bash
curl -s -X POST -H 'content-type: application/json' \
  -d '{"path": "shoots/2026-10", "recursive": true}' http://<PUPPET_HOST>:8220/pose/scan
```

The answer is a `ScanSummary`: totals (`images`, `written`, `people`, `skipped` and `failed` by reason, `truncated`)
and one `ImageOutcome` per image. Images are `.jpg`, `.jpeg`, `.png` and `.webp`; hidden names are left out.

| Outcome reason | Status | Why |
|---|---|---|
| `exists` | skipped | A sidecar is already there and `overwrite` is false. An older sidecar (a wire 1.0 one, without `labels`) is kept as it is too: pass `overwrite` to re-pose it |
| `sidecar_collision` | skipped | Two images share a stem (`a.jpg`, `a.png`) and so a sidecar; only the first in sorted order is posed |
| `max_images` | skipped | The run reached `ARGUS_PUPPET_MAX_SCAN_IMAGES` (default 2000) images; `truncated` is true. Call again to continue: posed images are then skipped as `exists` |
| `outside_root` | skipped | The image resolves outside the data root (a symlink) |
| `unreadable` | failed | The file could not be read or decoded |
| `too_large` | failed | Over the byte or pixel cap |
| `unwritable` | failed | The sidecar could not be written: its folder is not writable (the image is then not posed), or the write failed (a directory in its place, a full disk) |

A failed image never stops the run. A missing model or a refused GPU lease does (503).

| Status | When |
|---|---|
| 200 | The run finished, including runs where some images failed (some `unwritable` among written ones too) or none were found |
| 400 | `path` falls outside the data root |
| 404 | `path` does not exist under the data root |
| 409 | No data root is configured: the route is off (answered before the body is read). Or no sidecar at all could be written (every attempt failed `unwritable`, typically a read-only data root): `detail` is then `{"message", "summary"}`, the `ScanSummary` of the run. Retrying will not help until the folder is writable |
| 413 | The body is larger than 16 KiB: a `ScanRequest` is a path and two flags |
| 415 | `Content-Type` is present and is not `application/json` |
| 422 | The body is not a valid `ScanRequest` |
| 503 | Another scan is running (one at a time), or a model or GPU lease problem as for `/pose` |

The scan runs to the end inside the request, so a large folder is a long request. On CUDA it holds one GPU lease
for the whole folder at `batch` priority, handing it back between images when argus-iris asks.

### Pose sets: `results` or `path`

`/stats` and `/compare` take sets of poses. Each set (`PoseSetSource`) is given exactly one way:

| Field | Meaning |
|---|---|
| `results` | A list of `PoseResult` documents, for example sidecars the client read |
| `path` | A folder relative to `ARGUS_PUPPET_DATA_ROOT`; its `*.pose.json` sidecars are the set |
| `recursive` | With `path`: include subfolders. Default `true` |

A `path` is fenced as `/pose/scan` fences its path: 409 without a data root, 400 outside it or not a folder, 404
missing. A sidecar that resolves outside the data root is skipped; one that is not a valid `PoseResult` makes the
request 422, naming the file relative to the data root. Giving both `results` and `path`, or neither, is 422.

A set may hold at most `ARGUS_PUPPET_MAX_SET_RESULTS` poses (default 5000; 413 past it), because the distance
matrix grows with the square of the set (5000 poses is a 100 MB matrix). An inline set is counted on the raw body
before it is validated, so an oversized one is refused cheaply: a body may hold at most that many results per set
(two sets for `/compare`) and at most `ARGUS_PUPPET_MAX_JSON_BYTES / 2048` people in all (a real person is about
3.5 KB of JSON; validating JSON costs many times its size in memory). A `PoseResult` holds at most 500 people.

Every metric is computed on each image's **primary** person. What the numbers mean, and their limits, is in
[metrics.md](metrics.md).

### `POST /stats`

Diversity, clusters and coverage of one set.

```json
{"path": "lora-outputs/run-12", "params": {"cluster_threshold": 0.25}}
```

`params` is a `MetricParams`: any subset of `min_joint_score`, `min_usable_joints`, `cluster_threshold`,
`min_shared_joints`, `min_shared_limb_joints` and `diversity_scale`, each defaulting to the suite value
([metrics.md](metrics.md#parameters)). Leave it out for the defaults.

```bash
curl -s -X POST -H 'content-type: application/json' \
  -d '{"path": "lora-outputs/run-12"}' http://<PUPPET_HOST>:8220/stats
```

The answer is a `PoseSetStats`: `count`, `usable`, `comparable_pairs` and `comparable_share` (how many usable pairs
the scores could compare), `diversity` (0 to 1; argus-proof's pose-diversity gate reads this, beside
`largest_cluster_share`), `mean_pairwise_distance`, `cluster_count`, `largest_cluster_share`, `clusters` (largest first, each with a
`medoid` image), `coverage` (counts per stance, arms and facing label), `items` (one per image, with its cluster,
labels and a 2D point for a scatter plot), the `params` used, and the shared `estimator` when every pose came from
the same one.

### `POST /compare`

An output set's poses against a reference (training) set: for each output, the nearest reference pose.

```json
{"outputs": {"path": "lora-outputs/run-12"}, "reference": {"path": "training/concept-a"}, "threshold": 0.04}
```

| Field | Default | Meaning |
|---|---|---|
| `outputs`, `reference` | required | Two pose sets, each `results` or `path` |
| `threshold` | 0.04 | Pose distance (torso units) below which a match over the whole limb skeleton is a close pose match |
| `params` | suite defaults | `MetricParams`, as for `/stats` |

```bash
curl -s -X POST -H 'content-type: application/json' \
  -d '{"outputs": {"path": "lora-outputs/run-12"}, "reference": {"path": "training/concept-a"}}' \
  http://<PUPPET_HOST>:8220/compare
```

The answer is a `PoseSetComparison`: one `matches` entry per output (`nearest`, `distance`, `shared_joints`,
`close`), `output_count`, `matched_count`, `reference_count`, `close_eligible` and `close_share`, the reference set's
own leave-one-out `baseline_close_share` and `close_excess`, nearest-distance quantiles for both sides, the
`coverage_divergence` between the two sets' label mixes, both coverages, the threshold and params used.

A close pose match means "the same pose", not "a copied frame": different people strike the same pose, and a
pose-conditioned generator reproduces reference poses by design. The per-output flag points an image-level check at
a training frame; the run-level signal is `close_excess`
([metrics.md](metrics.md#what-a-close-pose-match-means)).

Each set counts against the set cap separately, so one request can carry two sets of 5000.

### `POST /render`

One pose to an OpenPose-convention skeleton PNG for a pose ControlNet.

```json
{"pose": {"puppet_version": "1.1", "image": "photo.jpg", "...": "a whole PoseResult"},
 "options": {"width": 1024, "height": 1024, "face": false, "stroke_scale": "auto"}}
```

`pose` is either a whole `PoseResult` (a sidecar, or a `/pose` answer) or bare keypoints with their frame:
`{"keypoints": [[x, y, score], ...], "width": 853, "height": 1280}`, where `keypoints` is one person's 133 points or
a list of people. `options` is a `RenderOptions`: `body`, `hands`, `face`, `width`, `height`, `min_score`, `fit`,
`primary_only`, `stroke_scale` and `margin`, documented in [conditioning.md](conditioning.md#options).

```bash
jq -n --slurpfile p photo.pose.json '{pose: $p[0], options: {width: 1024, height: 1024, stroke_scale: "auto"}}' |
  curl -s -X POST -H 'content-type: application/json' --data-binary @- \
    -o photo.openpose.png http://<PUPPET_HOST>:8220/render
```

| Status | When |
|---|---|
| 200 | `image/png` |
| 413 | The body is over 4 MiB (one pose of 500 people fits in 2 MB), the canvas is over `ARGUS_PUPPET_MAX_PIXELS`, or more people would be drawn than `ARGUS_PUPPET_MAX_PEOPLE` |
| 422 | The body is not a valid `RenderRequest`, or the pose cannot be drawn |

### Errors common to the JSON routes

`/stats`, `/compare` and `/render` read their body themselves (`/pose/scan` too, with its own 16 KiB cap):

| Status | When |
|---|---|
| 400 | The body could not be read (for example a malformed `Content-Length`) |
| 408 | The body did not arrive within `ARGUS_PUPPET_BODY_TIMEOUT_S` (default 60 s) |
| 413 | The body is larger than `ARGUS_PUPPET_MAX_JSON_BYTES` (default 128 MiB; 4 MiB for `/render`), a set is over the set cap, or the body holds more people than its size allows |
| 415 | `Content-Type` is present and is not `application/json` |
| 422 | The body is not valid JSON for the route's model |
| 503 | Busy (see below) |

## The primary subject

Every image with anyone in it has exactly one primary person (`PoseResult.primary`, and `primary: true` on that
person). Puppet uses argus-curator's rule for its primary face, so the two packages agree on who an image is about:

- Without a hint, the person with the largest `box width × box height × (0.5 + 0.5 × detector score)`; the first
  wins a tie.
- With a `face` hint, the person whose box best matches the face (the mean of their IoU and the share of the face
  inside the body box). If no body overlaps the face, the rule above decides.

## Concurrency and 503

Requests are admitted, never queued. A request past its bound is answered at once with 503 and `Retry-After: 1`,
and the client should retry.

| Routes | Bound |
|---|---|
| `POST /pose` | `ARGUS_PUPPET_POSE_CONCURRENCY` calls at once (default 2) |
| `POST /stats`, `/compare`, `/render` | A separate pool of the same size, shared by the three |
| `POST /pose/scan` | One scan at a time |

A call holds its slot while its body arrives, so the body read has a deadline: `ARGUS_PUPPET_BODY_TIMEOUT_S`
(default 60 s, enough for a 50 MiB upload at about 1 MB/s). A body still arriving past it is answered 408, the
connection is closed and the slot is freed. A body is counted against its cap as it streams, whether or not it
declares a `Content-Length`, so an oversized one is refused without being read to the end.

Model inference and set metrics run in worker threads, so `GET /health` keeps answering while they work.

Other 503s carry a different meaning:

| 503 detail | Meaning | What to do |
|---|---|---|
| `busy: try again` (with `Retry-After: 1`) | Over the admission bound | Retry |
| `GPU busy: ...` (with `Retry-After: 5`) | The GPU lease was refused or timed out (argus-iris or the host lease) | Retry later |
| `Model ... not found ...`, `... does not match its pinned SHA-256 ...`, `onnxruntime is not installed ...` | The service cannot run a model | Operator: `argus-puppet models fetch` / `verify`, or install an ONNX Runtime extra. Retrying will not help |
| `the GPU coordinator is misconfigured (ARGUS_GPU_COORDINATOR) ...` | `ARGUS_GPU_COORDINATOR` names no coordinator; `GET /health` reports `gpu_coordinator: invalid`. Only on CUDA, where a lease is needed | Operator: fix the variable (`none`, `lease` or `broker`). Retrying will not help |

## Security model

There is **no authentication**. Anyone who can reach the port can use every route: treat reaching it as access to
the host's models, its GPU time and, through the data root, its files. Keep the service on loopback (the default for
both `serve` and `compose.yaml`) or behind a proxy you control.

Three guards stand in front of the routes (a request meets the read-only guard before the origin guard):

1. **Origin guard.** Every state-changing request (every `POST` here) with a browser `Origin` header is refused with
   403 unless the origin is listed in `ARGUS_PUPPET_CORS_ORIGINS`, or the page is served from the same host and port
   and that host is an IP address, `localhost`, or listed in `ARGUS_PUPPET_TRUSTED_HOSTS`. This stops another
   website's page from driving the service through a visitor's browser. Clients that send no `Origin` (curl,
   server-to-server calls, the sibling packages) pass. A `*` in `ARGUS_PUPPET_CORS_ORIGINS` lets any page read
   answers but never write; name an origin to let a page post.
2. **Read-only mode.** With `ARGUS_PUPPET_READ_ONLY=1` every `POST` is refused with 403, including `POST /pose`,
   `/stats`, `/compare` and `/render`, which write nothing. Only `GET /health` (and the OpenAPI pages) answer.
3. **The data root.** Routes that take a path (`/pose/scan`, and `/stats` and `/compare` given a `path`) resolve it
   under `ARGUS_PUPPET_DATA_ROOT` and refuse anything outside it, including `..` and absolute paths. Files that
   resolve outside the root through a symlink are skipped. With no data root these routes refuse every path (409).
   No route takes a URL to fetch.

CORS headers are added outside the guards, so a browser can read a refusal's reason. Limits on every input (upload
bytes, pixels, people, JSON bytes, set sizes, images per scan) are listed with the route that applies them above and
in [deploy.md](deploy.md#environment-variables).

## Settings

All `ARGUS_PUPPET_*` variables are listed in [deploy.md](deploy.md#environment-variables). The ones the routes above
depend on:

| Variable | Default | Used by |
|---|---|---|
| `ARGUS_PUPPET_DATA_ROOT` | unset (path routes off) | `/pose/scan`, `path` sets |
| `ARGUS_PUPPET_READ_ONLY` | off | every `POST` |
| `ARGUS_PUPPET_CORS_ORIGINS`, `ARGUS_PUPPET_TRUSTED_HOSTS` | none | the origin guard |
| `ARGUS_PUPPET_MAX_UPLOAD_BYTES` | 52428800 (50 MiB) | `/pose`, and each image of a scan |
| `ARGUS_PUPPET_MAX_PIXELS` | 40000000 | `/pose`, scans, `/render` canvases |
| `ARGUS_PUPPET_MAX_PEOPLE` | 20 | people kept per image; people drawn by `/render` |
| `ARGUS_PUPPET_MAX_SCAN_IMAGES` | 2000 | `/pose/scan` |
| `ARGUS_PUPPET_POSE_CONCURRENCY` | 2 | `/pose`; separately, `/stats` + `/compare` + `/render` |
| `ARGUS_PUPPET_MAX_SET_RESULTS` | 5000 | each set of `/stats` and `/compare` |
| `ARGUS_PUPPET_MAX_JSON_BYTES` | 134217728 (128 MiB) | `/stats`, `/compare` bodies (and people per body: this / 2048); `/render` bodies up to 4 MiB |
| `ARGUS_PUPPET_BODY_TIMEOUT_S` | 60 | every `POST` body (408 past it) |
