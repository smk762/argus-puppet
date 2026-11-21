# Deploying argus-puppet

argus-puppet ships as two container images built from one `Dockerfile`:

| Image | Build | Runtime |
|---|---|---|
| `ghcr.io/smk762/argus-puppet:<version>` (also `:<major>.<minor>`, `:latest`) | `GPU=0` (default) | ONNX Runtime on the CPU |
| `ghcr.io/smk762/argus-puppet:<version>-gpu` (also `:<major>.<minor>-gpu`, `:latest-gpu`) | `GPU=1` | ONNX Runtime CUDA |

Both serve the HTTP API on port **8220** and run as an unprivileged user (uid 10001).

The GPU image brings the CUDA and cuDNN runtimes as NVIDIA's pip wheels (`onnxruntime-gpu[cuda,cudnn]`), so it
needs no CUDA base image. It does need the NVIDIA container runtime on the host and a driver new enough for the
CUDA major that the image's onnxruntime-gpu targets. The build fails unless `CUDAExecutionProvider` is available.

There is **no auth**. Anyone who can reach the port can use the service, so publish it on loopback (the default
in `compose.yaml`) or behind a proxy you control.

## Models are not in the image

The image ships no model weights. They live on a volume mounted at `/models` (`ARGUS_PUPPET_MODEL_DIR=/models` is
set in the image), and you download them into it once with `argus-puppet models fetch`. Each file is checked
against its pinned SHA-256 before it is written.

```bash
docker volume create puppet-models
docker run --rm -v puppet-models:/models ghcr.io/smk762/argus-puppet:latest argus-puppet models fetch
docker run --rm -v puppet-models:/models ghcr.io/smk762/argus-puppet:latest argus-puppet models list
```

A named volume takes on the image's ownership of `/models`, so the unprivileged user can write to it. If you bind a
host directory instead, it must be writable by uid 10001 for `models fetch`. To only serve from it, mount it
read-only (`-v <MODEL_DIR>:/models:ro`).

## Running

### CPU

```bash
docker run -d --name argus-puppet \
  -p 127.0.0.1:8220:8220 \
  -v puppet-models:/models \
  ghcr.io/smk762/argus-puppet:latest
```

### GPU

```bash
docker run -d --name argus-puppet --gpus all \
  -p 127.0.0.1:8220:8220 \
  -v puppet-models:/models \
  ghcr.io/smk762/argus-puppet:latest-gpu
```

`GET /health` reports `"device": "cuda"`. After the first pose request it also reports
`"provider": "CUDAExecutionProvider"`.

### Compose

`compose.yaml` has a `puppet` service (the CPU image) and a `puppet-gpu` service (the CUDA image) behind the `gpu`
profile. Both use the `puppet-models` volume. Run one of them, not both: they publish the same host port.

```bash
docker compose run --rm puppet argus-puppet models fetch   # once
docker compose up -d                                       # CPU
docker compose --profile gpu up -d puppet-gpu              # or the GPU image
```

The compose file reads these variables from the shell or a `.env` file:

| Variable | Default | Meaning |
|---|---|---|
| `PUPPET_PUBLISH` | `127.0.0.1:8220` | The host address and port to publish on |
| `PUPPET_TAG` | `latest` | The image tag. The GPU service appends `-gpu` |
| `PUPPET_DATA_DIR` | `./data` | The host directory for the optional data-root bind (commented out by default) |

Build locally instead of pulling with `docker compose build` (or `docker build --build-arg GPU=1 --build-arg
VERSION=<version> .`). Without `VERSION` the image reports `0.0.0.dev0`.

## Health check

`GET /health` answers with the package and wire versions and whether each pinned model file is present and
verified. It also gives the provider and device, free VRAM on a GPU, and read-only mode:

```json
{"puppet_version": "1.0", "version": "0.1.0", "read_only": false,
 "models": {"dwpose-l-384x288": true, "dwpose-m-256x192": true, "yolox_m": true, "yolox_nano": true},
 "pose_model": "dwpose-l-384x288", "detector_model": "yolox_m",
 "provider": "CUDAExecutionProvider", "device": "cuda", "gpu_coordinator": "none", "free_vram_mb": 22544}
```

The image's `HEALTHCHECK` calls it with Python (the slim image has no curl). The first call can hash the model
files, so the check allows a 60 s start period. `provider` is `null` until the first pose request loads a session.

A quick end-to-end test:

```bash
curl -s -X POST -H 'content-type: image/jpeg' --data-binary @photo.jpg 'http://127.0.0.1:8220/pose?name=photo.jpg'
```

## Environment variables

All `ARGUS_PUPPET_*` settings are read in `src/argus_puppet/config.py`:

| Variable | Default | Meaning |
|---|---|---|
| `ARGUS_PUPPET_MODEL_DIR` | `/models` in the image (`~/.cache/argus_puppet/models` outside it) | Where the pinned model files live |
| `ARGUS_PUPPET_PORT` | `8220` | Port, when `serve` gets no `--port`. The image passes `--port 8220` |
| `ARGUS_PUPPET_DEVICE` | `auto` | `auto`, `cpu` or `cuda`. `cuda` on the CPU image falls back to the CPU |
| `ARGUS_PUPPET_GPU_MEM_LIMIT_MB` | runtime default | A cap on each ONNX Runtime CUDA session's arena, in MB (at least 256) |
| `ARGUS_PUPPET_POSE_MODEL` | registry default | The pose model, by registry name (`argus-puppet models list`) |
| `ARGUS_PUPPET_DETECTOR_MODEL` | registry default | The person detector, by registry name |
| `ARGUS_PUPPET_DATA_ROOT` | unset | The only root a caller-supplied path may fall under. Unset: `POST /pose/scan`, and `/stats` or `/compare` given a `path`, refuse (409) |
| `ARGUS_PUPPET_READ_ONLY` | off | Refuse every state-changing request (see below) |
| `ARGUS_PUPPET_CORS_ORIGINS` | none | Browser origins allowed to call the API, comma-separated |
| `ARGUS_PUPPET_TRUSTED_HOSTS` | none | Host names (no port) a page may be served from and still write to its own server. IPs and `localhost` need none |
| `ARGUS_PUPPET_MAX_UPLOAD_BYTES` | 52428800 (50 MiB) | Largest accepted upload |
| `ARGUS_PUPPET_MAX_PIXELS` | 40000000 | Largest accepted image, in pixels |
| `ARGUS_PUPPET_MAX_PEOPLE` | 20 | People kept per image, highest detector score first (1 to 500) |
| `ARGUS_PUPPET_DET_THRESHOLD` | 0.3 | Person-detector score threshold |
| `ARGUS_PUPPET_NMS_IOU` | 0.45 | Detector non-maximum-suppression IoU |
| `ARGUS_PUPPET_MAX_SCAN_IMAGES` | 2000 | Images one `POST /pose/scan` processes. The rest are skipped |
| `ARGUS_PUPPET_POSE_CONCURRENCY` | 2 | `POST /pose` calls in flight at once, and separately `POST /stats`, `/compare` and `/render` calls together (1 to 64). Past it a call gets 503 with `Retry-After` |
| `ARGUS_PUPPET_MAX_SET_RESULTS` | 5000 | Poses in one set of `POST /stats` or `/compare` (the distance matrix grows with the square). Past it: 413 |
| `ARGUS_PUPPET_MAX_JSON_BYTES` | 134217728 (128 MiB) | Largest JSON body `POST /stats`, `/compare` and `/render` accept. Past it: 413 |
| `ARGUS_PUPPET_AGE_URL` | unset | argus-curator's base URL, for `argus-puppet library build --age-check` only. Unset: the check cannot run and nothing is sent ([index.md](index.md#privacy-and-the-age-fail-safe)) |

The image also sets `ARGUS_GPU_TENANT_MANIFEST=/app/gpu-tenant.yaml` (see [GPU sharing](#gpu-sharing-argus-iris)).

## Read-only mode

`ARGUS_PUPPET_READ_ONLY=1` adds a guard that refuses every state-changing request with 403. In the current release
that covers every `POST`, including `POST /pose` (which writes nothing), so only `GET /health` still answers.
Independently of read-only mode, every unsafe request passes an origin guard: a page from another site is refused
unless its origin is in `ARGUS_PUPPET_CORS_ORIGINS`, or its host is in `ARGUS_PUPPET_TRUSTED_HOSTS`. The full
security model is in [server.md](server.md#security-model).

## The data root

`POST /pose/scan {path, recursive, overwrite}` poses a folder of images and writes a `<stem>.pose.json` sidecar
beside each one. `path` is resolved under `ARGUS_PUPPET_DATA_ROOT` and refused if it falls outside. With no data
root the route refuses everything. `POST /stats` and `POST /compare` can also name a set by a folder of sidecars
under the same root. `POST /pose`, which takes the image in the request body, needs no data root, nor do sets sent
inline.

To enable scans, bind the dataset into the container and point the data root at it:

```bash
docker run -d --name argus-puppet \
  -p 127.0.0.1:8220:8220 \
  -v puppet-models:/models \
  -v <DATA_DIR>:/data \
  -e ARGUS_PUPPET_DATA_ROOT=/data \
  ghcr.io/smk762/argus-puppet:latest
```

The container runs as uid 10001, which must be able to read the images and write the sidecars. Mounted `:ro`, a
scan can read the images, but every sidecar write fails. Run with `--user <uid>:<gid>` to match the dataset's owner
if needed.

## GPU sharing (argus-iris)

On a GPU shared with other Argus services, the GPU image takes a lease for each batch of pose work through
argus-cortex's coordinator. The CPU image never takes one. These host-wide variables, shared by every service on the
host, select the coordinator:

| Variable | Meaning |
|---|---|
| `ARGUS_GPU_COORDINATOR` | `none` (default, no gating), `lease` (a host-wide file lock plus a free-VRAM wait) or `broker` (argus-iris) |
| `ARGUS_GPU_BROKER_URL` | argus-iris's URL, e.g. `http://<GPU_HOST>:<IRIS_PORT>` |
| `ARGUS_GPU_BROKER_TOKEN` | The broker token. Keep it in the host's `.env`, never in a committed file |
| `ARGUS_GPU_TENANT_MANIFEST` | The tenant manifest. The image sets `/app/gpu-tenant.yaml` (this repo's `gpu-tenant.yaml`) |
| `ARGUS_GPU_TENANT` | Overrides the tenant name |
| `ARGUS_GPU_BROKER_FALLBACK` | What to use while the broker cannot be reached: `lease` (default) or `none` |
| `ARGUS_GPU_LEASE_PATH` | The `lease` coordinator's lock file. Mount one host directory into every GPU service and name the same file in it |
| `ARGUS_GPU_LEASE_TIMEOUT_S` | How long a `lease` waits |
| `ARGUS_GPU_DEVICE` | Which card NVML probes for free VRAM |

The manifest declares one kind, `pose` (the person detector and the pose model loaded together; about 1600 MB VRAM,
240 MB of it weights, batch priority, not preemptible). `POST /pose` leases at `interactive` priority unless the
`X-Argus-Priority` header asks for another class. Folder scans lease at `batch` priority.
`compose.yaml` lists these variables, commented out, on the `puppet-gpu` service.
