# MOSS-TTSD server

Container image and Helm chart to run [MOSS-TTSD](https://github.com/OpenMOSS/MOSS-TTSD)
v1.0 as an HTTP TTS service, using the project's native SGLang end-to-end
serving. This is an optional self-hosted TTS backend, deployed separately from
the `tts-podcast` pipeline (which uses Gemini TTS by default).

## Layout

```text
deploy/moss-ttsd/
├── Dockerfile        # CUDA image: clones MOSS-TTSD, installs deps (uv), runs SGLang
├── entrypoint.sh     # first-boot weight download + fusion, then `sglang serve`
├── .dockerignore
└── chart/            # bjw-s common 5.x Helm chart
```

## How it works

- **GPU only.** MOSS-TTSD needs an NVIDIA CUDA GPU; there is no CPU path. The
  image builds on `nvidia/cuda:*-devel` (nvcc is required to compile
  flash-attn).
- **Weights are not baked into the image.** On first start `entrypoint.sh`
  downloads `OpenMOSS-Team/MOSS-TTSD-v1.0` and `OpenMOSS-Team/MOSS-Audio-Tokenizer`
  from HuggingFace, fuses them with `scripts/fuse_moss_tts_delay_with_codec.py`
  into `$MODEL_DIR`, then launches SGLang. The fused model lives on a persistent
  volume, so this one-time step is skipped on later starts.
- **Server.** SGLang's native HTTP server on port `30000`
  (`sglang serve --model-path <fused> --delay-pattern --trust-remote-code`).
  `/health` reports readiness once the model is loaded.
- **Two virtualenvs.** SGLang is a separate OpenMOSS fork
  (`OpenMOSS/sglang`, branch `moss-ttsd-v1.0-with-cat`), cloned on its own and
  installed in `/opt/venv-sglang`. MOSS-TTSD's own deps (download + fusion) live
  in `/opt/venv-moss`. They are kept apart because upstream pins conflicting
  torch versions; `entrypoint.sh` calls each venv by absolute path.

## Build

The build context is this directory; the Dockerfile clones MOSS-TTSD itself.

```bash
docker buildx build \
  -t <registry>/moss-ttsd:1.0.0 \
  --build-arg MOSS_TTSD_REF=main \
  deploy/moss-ttsd
docker push <registry>/moss-ttsd:1.0.0
```

Pin `MOSS_TTSD_REF` (and `SGLANG_REF`) to tags or commits for reproducible
builds. `CUDA_IMAGE`, `PYTHON_VERSION`, `TORCH_CUDA`, `SGLANG_REPO` and
`SGLANG_REF` are also overridable build args. `TORCH_CUDA` (default `cu128`)
selects the PyTorch CUDA wheel index; keep it in sync with `CUDA_IMAGE` and the
`torch` pin in the upstream `requirements.txt`.

## Deploy

Set `controllers.main.containers.main.image.repository` to wherever you pushed
the image. If that registry is private, create a pull secret in the target
namespace first:

```bash
kubectl create secret docker-registry <name> \
  --docker-server=<registry> \
  --docker-username=<user> --docker-password=<token> -n <namespace>
```

Then reference it via `defaultPodOptions.imagePullSecrets` in `values.yaml`
(commented example provided) and install:

```bash
cd deploy/moss-ttsd/chart
helm dependency update
helm install moss-ttsd . \
  --set controllers.main.containers.main.image.tag=1.0.0
```

First boot is slow (tens of GB downloaded + fused): the startup probe tolerates
up to ~90 minutes. Watch `kubectl logs -f deploy/moss-ttsd`.

### Key values

- `persistence.models` — the model store (RWO PVC, 100Gi default). Set
  `storageClass` to a fast class. Keep it enabled; otherwise weights
  re-download every restart.
- `controllers.main.containers.main.resources` — GPU (`nvidia.com/gpu: 1`) and
  memory. Adjust to your hardware.
- `controllers.main.pod.runtimeClassName` — `nvidia` by default; add
  `nodeSelector`/`tolerations` (commented in `values.yaml`) to pin GPU nodes.
- `SGLANG_EXTRA_ARGS` env — forwarded to `sglang serve`, e.g.
  `--mem-fraction-static 0.85` to curb VRAM fragmentation on tight GPUs.

### Gated/private weight repos

The MOSS repos are public, so no token is needed by default. For gated repos:

1. Set `secrets.hf-token.enabled: true` and its `HF_TOKEN` value (or point at an
   existing secret).
2. Uncomment the `envFrom` block on the main container in `values.yaml`.

## Caveats / assumptions to verify

- Upstream install commands (`uv pip install ./sglang/python[all]`,
  `sglang serve ...`, the fuse script path) are taken from the MOSS-TTSD README.
  Re-check them against the ref you pin; the project moves fast.
- The CUDA base defaults to 12.8 (`nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`)
  because `requirements.txt` pins `torch==2.9.1+cu128`. Keep `CUDA_IMAGE` in
  lockstep with the torch pin: flash-attn compiles against the base image's
  nvcc, so a mismatched toolkit breaks the build. Ubuntu version is not
  significant (torch wheels are manylinux). The host NVIDIA driver must support
  CUDA 12.8.
- Not wired into the `tts-podcast` CLI. Consuming this backend from the pipeline
  is a separate change.
