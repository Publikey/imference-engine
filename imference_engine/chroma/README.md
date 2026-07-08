# Chroma backend configuration

Chroma **shares `RuntimeConfig`** with the other image backends — the full
env↔param↔default table lives in
[`../pipelines/README.md`](../pipelines/README.md). This file lists only what is
**specific to Chroma**.

Chroma is a FLUX-derived, **de-distilled** 8.9B flow-matching DiT (Apache-2.0)
with a large community-finetune following. It rides the generic engine machinery
on the same diffusers 0.38 stack — no separate engine class.

## Chroma specifics

| Aspect | Behaviour |
|---|---|
| Shared `RuntimeConfig` | `IMAGE_DEVICE`, `IMAGE_MODEL_CACHE`, `IMAGE_MODEL_CDN`, `MAX_GPU_MODELS`, `MAX_CPU_MODELS`, `IMAGE_ENABLE_CPU_OFFLOAD` apply. |
| `IMAGE_USE_TINY_VAE` | **Ignored** (FLUX-family 16-channel VAE, no TAESD drop-in). |
| VRAM | 8.9B transformer — set `IMAGE_ENABLE_CPU_OFFLOAD=1` on consumer GPUs (shared ModelManager offload path). |
| `base_model` (per-model) | Community Chroma checkpoints are transformer-only; the shared **T5-XXL** encoder, its tokenizer, the VAE and scheduler come from a diffusers-format base repo passed at `register_model(..., base_model=...)`. **Single** text encoder — no CLIP. |
| dtype | `bfloat16` (hard-coded). |
| scheduler | `FlowMatchEulerDiscreteScheduler` (set at load). Per-request `scheduler` name ignored; explicit `shift` via `backend_options` overrides (advanced). |
| guidance | **Real CFG** (Chroma is de-distilled): engine default `guidance_scale=4.0` is true classifier-free guidance, and **`negative_prompt` IS honored** — unlike guidance-distilled FLUX.1. |
| max sequence length | `512` (T5). |
| generator | Device-aware (CUDA gen on CUDA, CPU gen on CPU/MPS). |

## The FLUX contrast (why they are separate backends)

| | FLUX.1 (`flux`) | Chroma (`chroma`) |
|---|---|---|
| Text encoders | CLIP-L **+** T5-XXL | T5-XXL only |
| Guidance | Distilled (embedded; no negative) | Real CFG (negative honored) |
| `negative_prompt` | Ignored | Used |

## Example

```python
engine.register_model(
    "chroma-hd", backend="chroma",
    weights_path="/cache/chroma-hd.safetensors",
    base_model="lodestones/Chroma1-HD",   # diffusers-format base (example)
)
result = engine.generate(
    model="chroma-hd",
    prompt="a photo of an astronaut riding a horse",
    negative_prompt="lowres, blurry",     # honored (real CFG)
    num_steps=28, guidance_scale=4.0,      # or omit — engine defaults
)
```
