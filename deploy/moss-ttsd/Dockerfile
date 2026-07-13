# syntax=docker/dockerfile:1

# MOSS-TTSD v1.0 HTTP inference server.
#
# The container clones the upstream MOSS-TTSD repo (fuse script + weights glue)
# and, separately, the OpenMOSS SGLang fork used for end-to-end serving, then
# exposes the native SGLang HTTP server on port 30000.
#
# Two virtualenvs on purpose: upstream ships MOSS-TTSD and its SGLang fork with
# conflicting dependency pins (different torch), so they are installed side by
# side rather than resolved together:
#   * VENV_MOSS   -> requirements.txt + flash-attn + hf CLI  (download + fusion)
#   * VENV_SGLANG -> ./sglang/python[all]                    (serving)
# entrypoint.sh calls each venv by absolute path.
#
# Model weights are NOT baked into the image. They are downloaded and fused at
# first start by entrypoint.sh into a persistent volume (see the Helm chart).
# This keeps the image to CUDA + deps only and lets the multi-GB checkpoints
# live on a PVC that survives restarts and rollouts.
#
# GPU: an NVIDIA CUDA GPU is required. flash-attn is compiled against nvcc, so
# the build uses a CUDA *devel* base. There is no realistic CPU path.

# --- Build arguments -------------------------------------------------------
# CUDA devel base carries nvcc, needed to compile flash-attn. The CUDA minor
# MUST match the toolkit torch was built against: requirements.txt pins
# torch==2.9.1+cu128, i.e. CUDA 12.8. A mismatch (e.g. 12.4) makes flash-attn
# compile against the wrong nvcc. Bump this in lockstep with the torch pin.
# Ubuntu 24.04 LTS: supported to 2029, and ships python3.12 natively (no PPA).
# Override the whole base with --build-arg CUDA_IMAGE=...
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

FROM ${CUDA_IMAGE}

# Upstream refs to clone. Pin to tags/commits for reproducible builds; override
# at build time with --build-arg MOSS_TTSD_REF=<tag-or-sha> etc.
ARG MOSS_TTSD_REF=main
ARG MOSS_TTSD_REPO=https://github.com/OpenMOSS/MOSS-TTSD.git
# SGLang is a separate OpenMOSS fork on a MOSS-TTSD-specific branch, not part of
# the MOSS-TTSD repo and not a submodule. It must be cloned explicitly.
ARG SGLANG_REPO=https://github.com/OpenMOSS/sglang.git
ARG SGLANG_REF=moss-ttsd-v1.0-with-cat
ARG PYTHON_VERSION=3.12
# CUDA wheel variant for torch. The `+cu128` build lives on PyTorch's index,
# not PyPI, so uv is pointed there below. Keep in sync with CUDA_IMAGE.
ARG TORCH_CUDA=cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Keep apt's downloaded packages so the BuildKit cache mount below is effective.
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# System dependencies (python3.12 is the Ubuntu 24.04 system interpreter):
#   - python3.12 + venv/dev: runtime and headers for building extensions
#   - build-essential, ninja-build, git: compile flash-attn / clone the repos
#   - ffmpeg, libsndfile1: audio I/O for the TTS pipeline
#   - ca-certificates, curl: TLS + HEALTHCHECK probe
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git \
        build-essential ninja-build \
        ffmpeg libsndfile1 \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev

# uv: the project's package manager. Faster resolver + install, and the repo
# standard. Used for every Python install below via `uv pip --python <venv>`.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Two separate virtualenvs (see header). Neither is put on PATH: the entrypoint
# and the RUN steps below address them by absolute path to avoid cross-talk.
ENV VENV_MOSS=/opt/venv-moss \
    VENV_SGLANG=/opt/venv-sglang \
    UV_LINK_MODE=copy \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/${TORCH_CUDA} \
    UV_INDEX_STRATEGY=unsafe-best-match
RUN uv venv --python python${PYTHON_VERSION} ${VENV_MOSS} && \
    uv venv --python python${PYTHON_VERSION} ${VENV_SGLANG}

WORKDIR /app

# Clone MOSS-TTSD (fuse script under ./scripts) and the SGLang fork (./sglang).
# --depth 1 keeps layers small; fall back to a full clone + checkout when the
# ref is a bare commit sha that --branch can't take.
RUN git clone --depth 1 --branch ${MOSS_TTSD_REF} ${MOSS_TTSD_REPO} /app || \
    (git clone ${MOSS_TTSD_REPO} /app && git -C /app checkout ${MOSS_TTSD_REF})
RUN git clone --depth 1 --branch ${SGLANG_REF} ${SGLANG_REPO} /app/sglang || \
    (git clone ${SGLANG_REPO} /app/sglang && git -C /app/sglang checkout ${SGLANG_REF})

# --- VENV_MOSS: inference / download / fusion deps -------------------------
# torch must land before flash-attn (flash-attn's build imports torch). uv
# cache mount keeps wheels across rebuilds.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_MOSS} -r requirements.txt

# flash-attn compiles CUDA kernels; --no-build-isolation lets it see the torch
# already in VENV_MOSS. MAX_JOBS bounds parallel nvcc to avoid OOM on the builder.
RUN --mount=type=cache,target=/root/.cache/uv \
    MAX_JOBS=4 uv pip install --python ${VENV_MOSS} flash-attn --no-build-isolation

# huggingface_hub CLI for weight download in the entrypoint.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_MOSS} 'huggingface_hub[cli]'

# --- VENV_SGLANG: the serving engine ---------------------------------------
# OpenMOSS SGLang fork; [all] pulls the serving extras and provides the
# `sglang serve` entrypoint used for MOSS-TTSD end-to-end inference.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python ${VENV_SGLANG} './sglang/python[all]'

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
