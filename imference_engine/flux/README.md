# FLUX backend configuration

FLUX **shares `RuntimeConfig`** with SDXL / Z-Image — the full env↔param↔default
table (device, model cache/CDN, residency caps, CPU offload, batch sizing,
offline env) is documented once in
[`../pipelines/README.md`](../pipelines/README.md). This file lists only what is
**specific to FLUX**.

FLUX.1 (Black Forest Labs) is a 12B rectified-flow DiT with the largest
community-finetune ecosystem after SDXL. It rides the generic
`Engine`/`ModelManager` machinery on the same diffusers 0.38 stack — no separate
engine class.

## FLUX specifics

| Aspect | Behaviour |
|---|---|
| Shared `RuntimeConfig` | `IMAGE_DEVICE`, `IMAGE_MODEL_CACHE`, `IMAGE_MODEL_CDN`, `MAX_GPU_MODELS`, `MAX_CPU_MODELS`, `IMAGE_ENABLE_CPU_OFFLOAD` all apply. |
| `IMAGE_USE_TINY_VAE` | **Ignored.** FLUX uses a 16-channel VAE with no TAESD drop-in. |
| VRAM | The 12B transformer is ~24 GB in bf16. On consumer GPUs set `IMAGE_ENABLE_CPU_OFFLOAD=1` / `RuntimeConfig(enable_cpu_offload=True)` — peak VRAM drops toward the transformer alone. Handled by the shared ModelManager offload path; no FLUX-specific code. |
| `base_model` (per-model, not env) | Community FLUX checkpoints are transformer-only and need shared components (CLIP-L + T5-XXL text encoders, their tokenizers, VAE, scheduler) from a base repo — pass per model at `register_model(..., base_model="black-forest-labs/FLUX.1-dev")`, resolved offline into the flat tree (or via `IMAGE_MODEL_CDN`). |
| dtype | `bfloat16` (hard-coded). |
| scheduler | `FlowMatchEulerDiscreteScheduler` with dynamic shifting (set at load). The per-request `scheduler` name is ignored. An explicit fixed `shift` via `backend_options={"shift": ...}` overrides dynamic shifting (advanced). |
| guidance | **Guidance-distilled.** `guidance_scale` (engine default 3.5) is an embedded conditioning signal, not classifier-free guidance. `negative_prompt` is **ignored** (true-CFG negatives are not wired in V1). |
| dev vs schnell | A per-**checkpoint** distinction, expressed as catalog **model defaults**, not the engine layer. dev ≈ 28 steps / guidance 3.5 (the engine defaults); schnell ≈ 4 steps (set `defaults: {num_steps: 4}` on its catalog row). |
| max sequence length | `512` (dev). schnell caps at 256 but tolerates the default. |
| generator | Device-aware (CUDA gen on CUDA, CPU gen on CPU/MPS), like Z-Image. |

## Offline base-components — gated repo caveat

`black-forest-labs/FLUX.1-dev` is a **GATED** HuggingFace repo (license
acceptance required) and its T5-XXL encoder is ~9.5 GB. For offline / worker use,
**mirror the base components to `IMAGE_MODEL_CDN`** (or pre-populate the flat
tree) rather than relying on a live HF pull. Under `HF_HUB_OFFLINE=1` the base
components resolve from the flat tree / CDN with zero HuggingFace contact. See
`../pipelines/README.md` → *Offline / HuggingFace*.

## Example

```python
from imference_engine import Engine, RuntimeConfig

engine = Engine(runtime=RuntimeConfig.from_env()).load()   # reads IMAGE_* env
engine.register_model(
    "flux-dev", backend="flux",
    weights_path="/cache/flux-dev.safetensors",
    base_model="black-forest-labs/FLUX.1-dev",
)
result = engine.generate(
    model="flux-dev", prompt="a photo of an astronaut riding a horse",
    num_steps=28, guidance_scale=3.5,   # or omit — these are the engine defaults
)
```

Catalog form (a schnell checkpoint expresses its 4-step recipe as model defaults):

```yaml
models:
  - name: flux-schnell
    engine: flux
    weights: /cache/flux-schnell.safetensors
    base_model: black-forest-labs/FLUX.1-schnell
    defaults:
      num_steps: 4
```
