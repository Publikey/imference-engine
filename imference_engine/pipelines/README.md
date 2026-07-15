# SDXL backend & shared image-side configuration

This documents the **image side** of the engine — the SDXL backend plus the
shared `RuntimeConfig` consumed by **both** SDXL and Z-Image (Z-Image specifics
live in [`../zimage/README.md`](../zimage/README.md)).

The engine is driven **identically by an env var OR a constructor param**, each
with a default. A launcher (a worker's `start.sh`, the desktop sidecar, a batch
script) only has to populate this contract — no engine logic lives in the
launcher.

- **Worker path:** `start.sh` exports the env vars below → `main.py` calls
  `RuntimeConfig.from_env()` → overrides `max_*_models` with hardware detection.
- **Desktop path:** build `RuntimeConfig(...)` directly with params, no env.

## `RuntimeConfig` — env ↔ param ↔ default

| Env var | `RuntimeConfig` param | Default | Effect |
|---|---|---|---|
| `IMAGE_DEVICE` | `device` | `auto` | `auto` \| `cuda` \| `cuda:N` \| `mps` \| `cpu`. |
| `IMAGE_MODEL_CACHE` | `model_cache_dir` | `None` → `$HF_HOME/image` or `~/.cache/image` | Root of the flat, symlink-free offline model tree (`<root>/<repo>/<file>`). |
| `IMAGE_MODEL_CDN` | `model_cdn` | `None` | Base URL of an R2/CDN mirror of the same `<repo>/<file>` layout. When set, base-components fetch from the CDN instead of HuggingFace. |
| `MAX_GPU_MODELS` | `max_gpu_models` | `None` (=1) | Pipes concurrently resident in VRAM. `auto`/unset → `None`; a worker resolves `auto` → number and overrides. |
| `MAX_CPU_MODELS` | `max_cpu_models` | `None` (=0) | Demoted-but-warm pipes kept in CPU RAM for fast GPU re-promotion. `auto`/unset → `None` (worker fills). |
| `IMAGE_USE_TINY_VAE` | `use_tiny_vae` | `false` | SDXL → TAESDxl, SD 1.5 → TAESD (~5 MB, ~10× faster decode, slight quality loss). No effect on Z-Image / FLUX / Chroma. |
| `IMAGE_ENABLE_CPU_OFFLOAD` | `enable_offload` | `false` | `enable_model_cpu_offload()` — peak VRAM drops to the largest submodel; ~10–30 % slower. Forces `max_cpu_models=0`. |

> **`auto` resolution is worker-side, not engine-side.** `from_env()` maps
> `MAX_*_MODELS=auto` (or unset) to `None` so the engine keeps a safe default;
> `config/resource_detection.py` in the worker computes the real number from
> RAM/VRAM (cgroup-aware) and overrides the field. This is the decoupling seam.

## Batch sizing (image side)

| Env var | Default | Effect |
|---|---|---|
| `BATCH_VRAM_RESERVE_MB` | `512` | Headroom kept free when sizing a batch from measured VRAM. |
| `MAX_BATCH_SIZE` | `8` | Hard ceiling on images per forward, even if VRAM allows more. |

## Offline / HuggingFace env (honoured implicitly)

| Env var | Effect |
|---|---|
| `HF_HUB_OFFLINE=1` | Hard guardrail: any stray HuggingFace call fails loudly instead of silently fetching. Set it whenever you rely on the flat tree / CDN. |
| `HF_HOME` | Fallback root for the flat tree when `IMAGE_MODEL_CACHE` is unset (the engine uses `$HF_HOME/image`). |
| `IMAGE_CDN_THREADS` (alias of `WAN_CDN_THREADS`) | Parallel HTTP streams for CDN downloads. Default `8`. |

## Not configurable (auto / optimal, hard-coded)

Listed so the desktop side knows what to expect — these are intentionally not
exposed:

- **dtype:** SDXL loads `float16` (Z-Image `bfloat16`).
- **attention:** flash + memory-efficient SDPA enabled on CUDA (and on AMD
  ROCm builds, where the same toggles drive the HIP SDPA path).
- **memory format:** `channels_last` applied to the SDXL UNet (non-fatal if it fails).
- **scheduler default:** `EulerAncestralDiscreteScheduler` (override per request via `scheduler=`).

## Examples

```python
# Desktop — params only, no environment.
from imference_engine import Engine, RuntimeConfig
engine = Engine(runtime=RuntimeConfig(
    device="auto",
    model_cdn="https://your-cdn.example/image",  # offline via CDN
)).load()

# Worker — env contract + hardware detection on top.
import os; from config import resolve_max_gpu_models, resolve_max_cpu_models
cfg = RuntimeConfig.from_env()                 # reads IMAGE_* env
cfg.max_gpu_models = resolve_max_gpu_models()  # auto → number (cgroup-aware)
cfg.max_cpu_models = resolve_max_cpu_models()
engine = Engine(runtime=cfg).load()
```

## Warm at deploy (optional)

`Engine.warm(specs)` pre-downloads base-components for the given
`(backend, base_model)` pairs **without loading a model** — so a worker can warm
the shared base in `setup()` and a fresh pod is "ready" with the base on disk
(the first request then only pays for the checkpoint weights, which stay lazy).
`specs` is typically the worker's catalog as distinct
`(config.engine, config.base_model)` pairs (deduped). Best-effort: a failed
prefetch logs a warning and falls back to the lazy fetch — `warm()` never raises.

```python
specs = {(c.engine, getattr(c, "base_model", None)) for c in catalog}
engine.warm(specs)   # SDXL config+VAE, each Z-Image base_model — downloaded, not loaded
```

## Per-request params (`Engine.generate`)

`model`, `prompt`, `negative_prompt`, `width=1024`, `height=1024`,
`num_steps=28`, `guidance_scale=6.0`, `clip_skip` (SDXL / SD 1.5 only),
`scheduler` (SDXL / SD 1.5 only), `batch=1`, `seed`, `source_image` +
`strength=0.75` (img2img), `backend_options` (e.g. `{"shift": 3.0}` for Z-Image).

> Full cross-engine payload + env reference: [`../../docs/reference.md`](../../docs/reference.md).
