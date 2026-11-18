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

## Z-Image base (undistilled)

`Tongyi-MAI/Z-Image` is the undistilled checkpoint (Apache-2.0): true CFG, negative
prompt, and the base for finetunes and LoRA training. It runs on this backend
unchanged — it is a catalog row, not a code path:

```yaml
- name: z-image-base
  engine: zimage
  weights: /models/z_image_bf16.safetensors
  base_model: Tongyi-MAI/Z-Image          # NOT -Turbo: its own scheduler config
  defaults:
    num_steps: 40            # upstream: 28-50
    guidance_scale: 4.0      # upstream: 3.0-5.0
```

Validated on an RTX 3090 (2026-11-13), 1024x1024, 40 steps, CFG 4.0, base components
from the CDN with HuggingFace blocked:

| mode | peak VRAM | time per image |
|---|---|---|
| resident | 23.3 GB | 81 s |
| `enable_offload=True` (model) | 13.4 GB | 92 s |

- Text-to-image, img2img (`strength`), the negative prompt and model offload all
  work; the offload render is identical to the resident one.
- **Run it with model offload on 24 GB cards.** CFG doubles the activations, and
  resident it sits at the card's limit (24.5 GB reserved); offload costs 13 % in time.
- About 2 s per step (two transformer passes): ten times slower per image than
  Turbo's 8 steps without CFG. Turbo stays the default for interactive use.
- Leave `shift` unset: the base repo's scheduler config applies. The `shift: 3.0` in
  the Turbo example above is Turbo's recipe.
