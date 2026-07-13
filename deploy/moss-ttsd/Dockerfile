# syntax=docker/dockerfile:1

# MOSS-TTSD v1.0 HTTP inference server.
#
# The container clones the upstream MOSS-TTSD repository (which bundles the
# custom SGLang fork used for end-to-end serving), installs its dependencies,
# and exposes the native SGLang HTTP server on port 30000.
#
# Model weights are NOT baked into the image. They are downloaded and fused at
# first start by entrypoint.sh into a persistent volume (see the Helm chart).
# This keeps the image reasonably sized (CUDA + torch + deps only) and lets the
# multi-GB checkpoints live on a PVC that survives restarts and rollouts.
#
# GPU: an NVIDIA CUDA GPU is required. flash-attn is compiled against nvcc, so
# the build uses a CUDA *devel* base. There is no realistic CPU path.

# --- Build arguments -------------------------------------------------------
# CUDA devel base carries nvcc, needed to compile flash-attn. The CUDA minor
# MUST match the toolkit torch was built against: requirements.txt pins
# torch==2.9.1+cu128, i.e. CUDA 12.8. A mismatch (e.g. 12.4) makes flash-attn
# compile against the wrong nvcc. Bump this in lockstep with the torch pin.
# Ubuntu 22.04 is a comfort default; torch wheels are manylinux, so 24.04
# works too. Override the whole base with --build-arg CUDA_IMAGE=...
# Ubuntu 24.04 LTS: supported to 2029, and ships python3.12 natively (no PPA).
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

FROM ${CUDA_IMAGE}

# Upstream ref to clone. Pin to a tag/commit for reproducible builds; override
# at build time with --build-arg MOSS_TTSD_REF=<tag-or-sha>.
ARG MOSS_TTSD_REF=main
ARG MOSS_TTSD_REPO=https://github.com/OpenMOSS/MOSS-TTSD.git
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Keep apt's downloaded packages so the BuildKit cache mount below is effective.
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# System dependencies (python3.12 is the Ubuntu 24.04 system interpreter):
#   - python3.12 + venv/dev: runtime and headers for building extensions
#   - build-essential, ninja-build, git: compile flash-attn / clone the repo
#   - ffmpeg, libsndfile1: audio I/O for the TTS pipeline
#   - ca-certificates, curl: TLS + HEALTHCHECK probe
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        build-essential ninja-build \
        ffmpeg libsndfile1 \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev

# uv: the project's package manager. Faster resolver + install, and the
# repo standard. Used for every Python install below via `uv pip`.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Isolated virtualenv driven by uv, on PATH so later RUN steps and the
# entrypoint resolve the right python. VIRTUAL_ENV lets `uv pip` target it
# without an explicit --python flag.
ENV VENV=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    UV_LINK_MODE=copy
RUN uv venv --python python${PYTHON_VERSION} ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

WORKDIR /app

# Clone the upstream repo at the pinned ref. --depth 1 keeps the layer small;
# the bundled SGLang fork lives under ./sglang and the fuse script under
# ./scripts, both needed at build (install) and runtime (fusion).
RUN git clone --depth 1 --branch ${MOSS_TTSD_REF} ${MOSS_TTSD_REPO} /app || \
    (git clone ${MOSS_TTSD_REPO} /app && git -C /app checkout ${MOSS_TTSD_REF})

# Python dependencies. torch must land before flash-attn (flash-attn's build
# imports torch). uv cache mount keeps wheels across rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# SGLang fork bundled in the repo (provides the `sglang serve` entrypoint used
# for MOSS-TTSD end-to-end inference). [all] pulls the serving extras.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install './sglang/python[all]'

# flash-attn compiles CUDA kernels; --no-build-isolation lets it see the torch
# already installed above. MAX_JOBS bounds parallel nvcc to avoid OOM on the
# builder.
RUN --mount=type=cache,target=/root/.cache/uv \
    MAX_JOBS=4 uv pip install flash-attn --no-build-isolation

# huggingface_hub CLI for weight download in the entrypoint.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install 'huggingface_hub[cli]'

# --- Runtime configuration -------------------------------------------------
# MODEL_DIR: persistent location for the fused model (mounted from a PVC).
# HF_HOME:   HuggingFace cache, also on the volume so downloads survive.
ENV MODEL_DIR=/models/moss-ttsd-fused \
    HF_HOME=/models/hf-cache \
    MOSS_TTSD_HF_REPO=OpenMOSS-Team/MOSS-TTSD-v1.0 \
    MOSS_AUDIO_TOKENIZER_HF_REPO=OpenMOSS-Team/MOSS-Audio-Tokenizer \
    SERVE_HOST=0.0.0.0 \
    SERVE_PORT=30000

COPY --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh

# Non-root runtime user (UID/GID > 10000 per policy). The /models volume is
# chowned here so the downloaded/fused weights are writable at runtime.
RUN groupadd -r -g 10001 moss && useradd -r -u 10001 -g moss -m -d /home/moss moss && \
    mkdir -p /models && chown -R moss:moss /models /app
USER moss

LABEL org.opencontainers.image.title="moss-ttsd-server" \
      org.opencontainers.image.description="MOSS-TTSD v1.0 end-to-end TTS HTTP server (SGLang)" \
      org.opencontainers.image.source="https://github.com/obeone/tts-podcast" \
      org.opencontainers.image.licenses="Apache-2.0"

EXPOSE 30000

# SGLang exposes /health once the model is loaded. Long start-period: first
# boot downloads and fuses multi-GB weights before the server comes up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=5 \
    CMD curl -fsS "http://localhost:${SERVE_PORT}/health" || exit 1

STOPSIGNAL SIGINT

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
