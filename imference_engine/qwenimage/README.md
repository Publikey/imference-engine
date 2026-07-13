# Qwen-Image backend configuration

Qwen-Image **shares `RuntimeConfig`** with the other image backends — the full
env↔param↔default table lives in
[`../pipelines/README.md`](../pipelines/README.md). This file lists only what is
**specific to Qwen-Image**.

Qwen-Image (Alibaba) is a 20B MMDiT whose standout is native, reliable text
rendering (multi-line, paragraph, CJK). It rides the generic engine machinery on
the same diffusers 0.39 stack — no separate engine class.

## Qwen-Image specifics

| Aspect | Behaviour |
|---|---|
| Shared `RuntimeConfig` | `IMAGE_DEVICE`, `IMAGE_MODEL_CACHE`, `IMAGE_MODEL_CDN`, `MAX_GPU_MODELS`, `MAX_CPU_MODELS`, `IMAGE_ENABLE_CPU_OFFLOAD` apply. |
| `IMAGE_USE_TINY_VAE` | **Ignored** (custom Qwen-Image VAE, no TAESD drop-in). |
| VRAM | 20B transformer — very heavy. Set `IMAGE_ENABLE_CPU_OFFLOAD=1` and prefer a quantized build; quantization is not wired here yet. |
| `base_model` (per-model) | Community checkpoints are transformer-only; the shared **Qwen2.5-VL** text encoder (tens of GB), its tokenizer, the VAE and scheduler come from a diffusers-format base repo (`Qwen/Qwen-Image`) passed at `register_model(..., base_model=...)`. **Single** encoder — no CLIP. Prefer CDN-on-demand for the huge encoder. |
| dtype | `bfloat16` (hard-coded). |
| scheduler | `FlowMatchEulerDiscreteScheduler` (set at load). Per-request `scheduler` name ignored; explicit `shift` via `backend_options` overrides (advanced). |
| guidance | **True CFG.** The engine's `guidance_scale` (default 4.0) is mapped to the pipeline's `true_cfg_scale`, and `negative_prompt` IS honored (default " ", matching upstream). The distilled `guidance_scale` pipeline arg stays at its default. |
| num steps | Base Qwen-Image recommends ~40-50 steps; the engine default is 28 (from `GLOBAL_DEFAULTS`) — bump it per model in the catalog, or lower it for a Lightning-distilled checkpoint. |
| generator | Device-aware (CUDA gen on CUDA, CPU gen on CPU/MPS). |

## Not in scope (yet)

**Qwen-Image-Edit** (instruction-based image editing) is a different pipeline
with an image-condition call signature, not this strength-based img2img. A future
addition.

## Example

```python
engine.register_model(
    "qwen-image", backend="qwenimage",
    weights_path="/cache/qwen-image.safetensors",
    base_model="Qwen/Qwen-Image",
)
result = engine.generate(
    model="qwen-image",
    prompt='a storefront with a neon sign reading "Imference"',
    negative_prompt=" ",
    width=1328, height=1328,
    num_steps=50, guidance_scale=4.0,   # guidance_scale -> true_cfg_scale
)
```

Catalog form (bump steps for the base model):

```yaml
models:
  - name: qwen-image
    engine: qwenimage
    weights: /cache/qwen-image.safetensors
    base_model: Qwen/Qwen-Image
    defaults:
      num_steps: 50
```
