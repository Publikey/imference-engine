# Z-Image backend configuration

Z-Image **shares `RuntimeConfig`** with SDXL — the full env↔param↔default table
(device, model cache/CDN, residency caps, CPU offload, batch sizing, offline
env) is documented once in [`../pipelines/README.md`](../pipelines/README.md).
This file only lists what is **specific to Z-Image**.

The engine is driven identically by an env var OR a constructor param. Workers
populate the env via `start.sh` and call `RuntimeConfig.from_env()`; the desktop
sidecar builds `RuntimeConfig(...)` directly.

## Z-Image specifics

| Aspect | Behaviour |
|---|---|
| Shared `RuntimeConfig` | All of `IMAGE_DEVICE`, `IMAGE_MODEL_CACHE`, `IMAGE_MODEL_CDN`, `MAX_GPU_MODELS`, `MAX_CPU_MODELS`, `IMAGE_ENABLE_CPU_OFFLOAD` apply. |
| `IMAGE_USE_TINY_VAE` | **Ignored.** Z-Image uses a different VAE architecture with no TAESD equivalent. |
| `base_model` (per-model, not env) | Z-Image finetunes need shared base-components (tokenizer + text_encoder + VAE) from a base repo (e.g. `Tongyi-MAI/Z-Image-Turbo`). Passed per model at `register_model(..., base_model=...)`, resolved offline into the flat tree (or via `IMAGE_MODEL_CDN`). |
| dtype | `bfloat16` (hard-coded). |
| scheduler | `FlowMatchEulerDiscreteScheduler`. Per-request `shift` via `backend_options={"shift": 3.0}` (3.0 ≈ 480p, 5.0 ≈ 720p; auto-detects Turbo). |
| generator | Device-aware (CUDA gen on CUDA, CPU gen on CPU/MPS). |

## Offline base-components

The base repo's text_encoder is a Qwen-family LLM (tens of GB) — prefer
**CDN-on-demand** (`IMAGE_MODEL_CDN`) over bloating any boot tarball. Under
`HF_HUB_OFFLINE=1` the base-components resolve from the flat tree / CDN with
zero HuggingFace contact. See `../pipelines/README.md` → *Offline / HuggingFace*.

## Example

```python
from imference_engine import Engine, RuntimeConfig

engine = Engine(runtime=RuntimeConfig.from_env()).load()   # reads IMAGE_* env
engine.register_model(
    "z-image-turbo", backend="zimage",
    weights_path="/cache/z-image-turbo.safetensors",
    base_model="Tongyi-MAI/Z-Image-Turbo",
)
result = engine.generate(
    model="z-image-turbo", prompt="...",
    num_steps=8, guidance_scale=1.0,
    backend_options={"shift": 3.0},
)
```
