# syntax=docker/dockerfile:1
#
# argus-puppet: the pose service on :8110. One file, two images:
#
#   docker build --build-arg VERSION=0.1.0 -t argus-puppet .                     # CPU  (:<version>)
#   docker build --build-arg VERSION=0.1.0 --build-arg GPU=1 -t argus-puppet:gpu .  # CUDA (:<version>-gpu)
#
# Models are not baked in: mount a volume at /models and fill it with `argus-puppet models fetch` (docs/deploy.md).
FROM python:3.11-slim

# Pin past the eMudhra/emSign incident (discuss.python.org/t/109576). A floating
# :latest tag can stay cached on a pre-0.13.4 digest.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

# Distrust emSign ECC Root CA - G3 and the Comodo AAA cross-sign root.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i \
        -e 's|^mozilla/emSign_ECC_Root_CA_-_G3|!mozilla/emSign_ECC_Root_CA_-_G3|' \
        -e 's|^mozilla/Comodo_AAA|!mozilla/Comodo_AAA|' \
        /etc/ca-certificates.conf \
    && update-ca-certificates --fresh >/dev/null \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The version is handed in (the release tag without the "v"), so the build context carries no .git and the image
# needs no git binary: hatch-vcs (setuptools-scm) reads SETUPTOOLS_SCM_PRETEND_VERSION instead of walking history.
# It must be the plain name -- hatch-vcs ignores the ..._FOR_ARGUS_PUPPET spelling. It is exported only for the
# install below, not baked into the image's environment. The default keeps a bare `docker build .` working.
ARG VERSION=0.0.0.dev0
# GPU=1 builds the CUDA image: the [gpu] extra (onnxruntime-gpu, NVML and the broker client) instead of [cpu], with
# the CUDA and cuDNN runtimes as NVIDIA's pip wheels (onnxruntime-gpu[cuda,cudnn]) so the base stays python-slim;
# argus_puppet.runtime preloads them through onnxruntime.preload_dlls(). It needs the NVIDIA container runtime and a
# driver new enough for the CUDA major that onnxruntime-gpu targets. The default (0) is the CPU image.
ARG GPU=0

COPY . .

RUN export SETUPTOOLS_SCM_PRETEND_VERSION="$VERSION" \
    && if [ "$GPU" = "1" ]; then \
        uv pip install --system --no-cache "certifi>=2026.10.19" ".[server,cli,gpu]" "onnxruntime-gpu[cuda,cudnn]" \
        # The CPU onnxruntime wheel shares onnxruntime-gpu's module directory and wins if anything pulled it in: \
        # drop it and lay the GPU build back down, then fail the build unless the CUDA provider is really there. \
        && if uv pip show --system onnxruntime >/dev/null 2>&1; then \
            ort_gpu="$(uv pip show --system onnxruntime-gpu | sed -n 's/^Version: //p')" \
            && uv pip uninstall --system onnxruntime \
            && uv pip install --system --no-cache --reinstall --no-deps "onnxruntime-gpu==$ort_gpu"; \
        fi \
        && python -c "import onnxruntime as o, sys; p = o.get_available_providers(); print('providers:', p); sys.exit(0 if 'CUDAExecutionProvider' in p else 'CUDAExecutionProvider missing')"; \
    else \
        uv pip install --system --no-cache "certifi>=2026.10.19" ".[server,cli,cpu]"; \
    fi \
    && argus-puppet version

# Unprivileged at run time. /models is the model volume (the image ships none); a named volume inherits this
# ownership, so `argus-puppet models fetch` can write into it. The home directory is where onnxruntime keeps its
# small per-user state; without a writable one it warns on every start.
RUN useradd --uid 10001 --user-group --create-home --home-dir /home/puppet --shell /usr/sbin/nologin puppet \
    && mkdir -p /models \
    && chown puppet:puppet /models
ENV ARGUS_PUPPET_MODEL_DIR=/models \
    ARGUS_GPU_TENANT_MANIFEST=/app/gpu-tenant.yaml \
    PYTHONUNBUFFERED=1
VOLUME /models
USER puppet

EXPOSE 8110
# GET /health may hash the model files once on its first call, hence the generous start period. No curl in slim.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8110/health', timeout=8)"]
# 0.0.0.0 inside the container; which host address it is published on is compose's (or `docker run -p`'s) choice.
CMD ["argus-puppet", "serve", "--host", "0.0.0.0", "--port", "8110"]
