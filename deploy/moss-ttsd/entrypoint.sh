#!/usr/bin/env bash
#
# MOSS-TTSD server entrypoint.
#
# On first start the multi-GB weights are not yet on the persistent volume, so
# this script:
#   1. downloads MOSS-TTSD v1.0 + MOSS-Audio-Tokenizer from HuggingFace,
#   2. fuses them into a single SGLang-loadable directory ($MODEL_DIR),
#   3. launches the native SGLang HTTP server.
#
# On subsequent starts the fused model already exists on the volume, so steps
# 1-2 are skipped and the server comes up quickly.
#
# All heavy paths (weights, HF cache, fused model) live under /models, which is
# expected to be a persistent volume.
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*" >&2; }

# --- Configuration (overridable via env) -----------------------------------
: "${MODEL_DIR:=/models/moss-ttsd-fused}"
: "${HF_HOME:=/models/hf-cache}"
: "${MOSS_TTSD_HF_REPO:=OpenMOSS-Team/MOSS-TTSD-v1.0}"
: "${MOSS_AUDIO_TOKENIZER_HF_REPO:=OpenMOSS-Team/MOSS-Audio-Tokenizer}"
: "${SERVE_HOST:=0.0.0.0}"
: "${SERVE_PORT:=30000}"

# The two virtualenvs baked into the image (see Dockerfile header). Download +
# fusion run in VENV_MOSS; serving runs in VENV_SGLANG.
: "${VENV_MOSS:=/opt/venv-moss}"
: "${VENV_SGLANG:=/opt/venv-sglang}"

# Staging dirs for the raw (un-fused) downloads.
RAW_DIR="${HF_HOME}/raw"
TTSD_DIR="${RAW_DIR}/moss-ttsd-v1.0"
CODEC_DIR="${RAW_DIR}/moss-audio-tokenizer"

# Extra flags forwarded verbatim to `sglang serve` (e.g. --mem-fraction-static
# to curb VRAM fragmentation on tight GPUs). Split on whitespace intentionally.
: "${SGLANG_EXTRA_ARGS:=}"

export HF_HOME
mkdir -p "${HF_HOME}"

# --- Fusion (first boot only) -----------------------------------------------
# The fused model is considered ready when its directory holds a config.json.
if [[ -f "${MODEL_DIR}/config.json" ]]; then
    log "Fused model already present at ${MODEL_DIR}, skipping download/fusion."
else
    log "Fused model not found. Downloading weights (this takes a while)..."

    # HF_TOKEN, when provided (K8s secret), authenticates gated/private repos.
    # hf CLI picks it up from the environment automatically.
    hf_download() {
        local repo="$1" dest="$2"
        log "Downloading ${repo} -> ${dest}"
        "${VENV_MOSS}/bin/huggingface-cli" download "${repo}" --local-dir "${dest}" --quiet
    }

    hf_download "${MOSS_TTSD_HF_REPO}" "${TTSD_DIR}"
    hf_download "${MOSS_AUDIO_TOKENIZER_HF_REPO}" "${CODEC_DIR}"

    log "Fusing MOSS-TTSD + audio tokenizer into ${MODEL_DIR}..."
    # Fuse into a temp dir, then atomically move into place so an interrupted
    # fusion never leaves a half-written model that the readiness check trusts.
    tmp_fused="${MODEL_DIR}.tmp.$$"
    rm -rf "${tmp_fused}"
    "${VENV_MOSS}/bin/python" /app/scripts/fuse_moss_tts_delay_with_codec.py \
        --model-path "${TTSD_DIR}" \
        --codec-model-path "${CODEC_DIR}" \
        --save-path "${tmp_fused}"
    rm -rf "${MODEL_DIR}"
    mv "${tmp_fused}" "${MODEL_DIR}"
    log "Fusion complete."
fi

# --- Serve ------------------------------------------------------------------
log "Starting SGLang server on ${SERVE_HOST}:${SERVE_PORT} for ${MODEL_DIR}"
# exec so SGLang becomes PID 1 and receives STOPSIGNAL (SIGINT) directly.
# --delay-pattern and --trust-remote-code are required by the fused MOSS model.
# shellcheck disable=SC2086
exec "${VENV_SGLANG}/bin/sglang" serve \
    --model-path "${MODEL_DIR}" \
    --delay-pattern \
    --trust-remote-code \
    --host "${SERVE_HOST}" \
    --port "${SERVE_PORT}" \
    ${SGLANG_EXTRA_ARGS}
