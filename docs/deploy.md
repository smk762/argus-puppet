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
`"provider": "CUDAExecutionProvider"`. Started without `--gpus` (or without the NVIDIA runtime), the GPU image runs
on the CPU under the default `ARGUS_PUPPET_DEVICE=auto`: it logs a `cuda_unavailable` warning, takes no GPU lease, and
`/health` reports `"device": "cpu"` with the reason in `cuda_unavailable`. With `ARGUS_PUPPET_DEVICE=cuda` it refuses
pose requests (503) instead of running them on the CPU. See [Device](server.md#device).

Name the container `argus-puppet`, as above, on a host whose GPU argus-iris brokers: see
[GPU sharing](#gpu-sharing-argus-iris).

### Compose

`compose.yaml` has a `puppet` service (the CPU image) and an `argus-puppet` service (the CUDA image, named after its
argus-iris tenant) behind the `gpu` profile. Both use the `puppet-models` volume. Run one of them, not both: they publish the same host port.

```bash
docker compose run --rm puppet argus-puppet models fetch   # once
docker compose up -d                                       # CPU
docker compose --profile gpu up -d argus-puppet            # or the GPU image
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
verified. It also gives the provider and device, free VRAM on a GPU, and read-only mode. The body is the suite's
`ServiceHealth` (argus-cortex 0.16) with puppet's own fields beside it ([server.md](server.md#get-health)):

```json
{"service_version": "1.0", "service": "argus-puppet", "version": "0.2.0", "status": "ok",
 "started_at": "2026-11-23T09:00:00Z", "wire": {"puppet_version": "1.3"}, "capabilities": [], "checks": {},
 "read_only": false, "build": null, "peer": "puppet", "puppet_version": "1.3",
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
| `ARGUS_PUPPET_DEVICE` | `auto` (`cuda` on the compose GPU service) | `auto`, `cpu` or `cuda`. `auto` uses CUDA when a session can really get it, else the CPU (with a `cuda_unavailable` warning). `cuda` refuses pose work (503) rather than run it on the CPU, for example on the CPU image or without `--gpus`. See [Device](server.md#device) |
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
| `ARGUS_PUPPET_MAX_JSON_BYTES` | 134217728 (128 MiB) | Largest JSON body `POST /stats` and `/compare` accept (`/render` takes at most 4 MiB), and with it the people one body may carry (this / 2048). Past it: 413 |
| `ARGUS_PUPPET_BODY_TIMEOUT_S` | 60 | Seconds any `POST` body may take to arrive, upload or JSON. Past it: 408, the connection is closed and the call's admission slot is freed, so a slow client cannot hold one |
| `ARGUS_PUPPET_AGE_URL` | unset | argus-curator's base URL, for `argus-puppet library build --age-check` only. Unset: the suite-wide `ARGUS_CURATOR_URL` is used (environment, then the [suite settings file](#the-suite-settings-file)); `off` ignores that too. Neither set: the check cannot run and nothing is sent ([index.md](index.md#privacy-and-the-age-fail-safe)). A value that is not an absolute `http(s)` URL is a configuration error at startup, naming the variable |

The image also sets `ARGUS_GPU_TENANT_MANIFEST=/app/gpu-tenant.yaml` (see [GPU sharing](#gpu-sharing-argus-iris)).

### The suite settings file

Values every Argus service on a host shares (`ARGUS_GPU_BROKER_URL`, `ARGUS_GPU_BROKER_TOKEN`, `ARGUS_CURATOR_URL`)
can live once in argus-cortex's suite file instead of in each service's `.env`: `~/.config/argus/suite.env` for the
CLI (`python -m argus_cortex.settings --init` creates it with mode 600; `ARGUS_ENV_FILE` names another, `off`
disables it), `/etc/argus/suite.env` in a container. Anything set in the service's own environment wins, and a blank
value counts as unset. Puppet's own `ARGUS_PUPPET_*` settings are read from the environment only.

`compose.yaml` has the mount as a commented-out line, because compose cannot make a bind mount optional (with no file
on the host, Docker creates a directory in its place):

```yaml
    volumes:
      - ${ARGUS_SUITE_FILE:-/etc/argus/suite.env}:/etc/argus/suite.env:ro
```

The file is read only when it belongs to the reading user or to root, nobody else can write it, and nobody else can
read it unless root owns it. The container runs as uid 10001, so a file owned by your own user with mode 600 is not
readable there: give the container a copy owned by root (mode 444, kept in a directory only root can enter) or owned
by uid 10001 (mode 600). A file that is missing, unreadable or refused is a warning in the log and no settings, never
a crash. Values are never logged.

### Discovery labels

Both compose services carry the labels argus-iris's agents read to list the suite's services, and `ARGUS_PEER`
repeats the first so `GET /health` can report it:

| Label | Value |
|---|---|
| `argus.peer` | `puppet`: the name callers ask for (their `ARGUS_PUPPET_URL`) |
| `argus.service` | `argus-puppet` |
| `argus.health` | `/health` |
| `argus.port` | `8220`, the container port |

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
| `ARGUS_GPU_COORDINATOR` | Unset: `broker` when `ARGUS_GPU_BROKER_URL` is set, else `none` (no gating). Or name one: `none`, `lease` (a host-wide file lock plus a free-VRAM wait) or `broker` (argus-iris) |
| `ARGUS_GPU_BROKER_URL` | argus-iris's URL, e.g. `http://<GPU_HOST>:<IRIS_PORT>` |
| `ARGUS_GPU_BROKER_TOKEN` | The broker token. Keep it in the host's `.env` or the [suite settings file](#the-suite-settings-file), never in a committed file |
| `ARGUS_GPU_TENANT_MANIFEST` | The tenant manifest. The image sets `/app/gpu-tenant.yaml` (this repo's `gpu-tenant.yaml`); unset, `./gpu-tenant.yaml` then `/app/gpu-tenant.yaml` is used |
| `ARGUS_GPU_TENANT` | Overrides the tenant name |
| `ARGUS_GPU_BROKER_FALLBACK` | What to use while the broker cannot be reached: `lease` (default) or `none` |
| `ARGUS_GPU_LEASE_PATH` | The `lease` coordinator's lock file. Mount one host directory into every GPU service and name the same file in it |
| `ARGUS_GPU_LEASE_TIMEOUT_S` | How long a `lease` waits |
| `ARGUS_GPU_DEVICE` | Which card NVML probes for free VRAM |

The manifest declares one kind, `pose` (the person detector and the pose model loaded together; about 1600 MB VRAM,
240 MB of it weights, batch priority, not preemptible). `POST /pose` leases at `interactive` priority unless the
`X-Argus-Priority` header asks for another class. Folder scans lease at `batch` priority.
`compose.yaml` lists the main ones, commented out, on the `argus-puppet` service. They are read from the environment,
then from the [suite settings file](#the-suite-settings-file).

Since argus-cortex 0.15 the broker's URL alone selects it: a host that sets `ARGUS_GPU_BROKER_URL` (in the
environment or the suite file) and leaves `ARGUS_GPU_COORDINATOR` unset now leases through argus-iris, where it used
to get no gating. Set `ARGUS_GPU_COORDINATOR=none` to keep puppet out of a suite-wide broker.

### A working broker configuration

In the host's `.env`, beside `compose.yaml`, or as `-e` flags to `docker run`:

```bash
ARGUS_GPU_BROKER_URL=http://<IRIS_HOST>:8120
ARGUS_GPU_BROKER_TOKEN=<IRIS_TOKEN>   # only when argus-iris is run with a token
```

The URL and the token are enough: the coordinator defaults to `broker` once the URL is set, the fallback to `lease`,
and the image already sets `ARGUS_GPU_TENANT_MANIFEST=/app/gpu-tenant.yaml`. Puppet registers its tenant (`argus-puppet`)
with the broker on its first lease, not at start, so iris lists it only after the first pose request on CUDA. No lease
is taken while puppet runs on the CPU.

If argus-iris cannot be reached, leases come from `ARGUS_GPU_BROKER_FALLBACK` (`lease`: the host-wide file lock).
`GET /health` keeps reporting `gpu_coordinator: broker` and shows `last_lease_via: fallback`. If iris answers but
refuses the request (a wrong or missing token, a rejected request), `POST /pose` answers 503 with no `Retry-After` and
says so, and the CLI prints the same message. The token is never echoed.

### Counting puppet's VRAM against its lease

argus-iris counts a container's VRAM as the tenant's own, inside its lease, only when it can link the container to
the tenant. Otherwise the memory counts as unattributed, and puppet's VRAM is counted twice: once as unattributed
and once in its lease. Iris links a container to the tenant in two ways:

* Through its name. Iris matches the compose service name, or the container name outside compose, against the
  tenant. The compose GPU service is `argus-puppet` for this reason, and the `docker run` examples use
  `--name argus-puppet`.
* Through `iris.yaml`, for any other name (a renamed service, a compose project prefix you match on, several
  replicas):

  ```yaml
  tenants:
    argus-puppet:
      containers: ["<CONTAINER_NAME>", "<PROJECT>-argus-puppet-*"]   # fnmatch patterns
  ```
