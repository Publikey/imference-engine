# Qwen-Image backend configuration

Qwen-Image **shares `RuntimeConfig`** with the other image backends — the full
env↔param↔default table lives in
[`../pipelines/README.md`](../pipelines/README.md). This file lists only what is
**specific to Qwen-Image**.

Qwen-Image (Alibaba) is a 20B MMDiT whose standout is native, reliable text
rendering (multi-line, paragraph, CJK). It rides the generic engine machinery on
the same diffusers 0.40 stack — no separate engine class.

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

## Qwen-Image-2512 and the lite build

**Qwen-Image-2512** (Apache-2.0) is the December 2025 checkpoint: better human
realism and text, same architecture. It shares its encoder, VAE and configs with
Qwen-Image (identical files), so `base_model="Qwen/Qwen-Image"` serves both and
only the transformer changes.

The bf16 transformer is 41 GB. Below 40 GB of VRAM use the **lite build**, the
text-to-image twin of the Qwen-Image-Edit one:

- unsloth `qwen-image-2512-Q4_K_M.gguf` (13.3 GB);
- lightx2v's Lightning LoRA as an adapter: 4 steps (or 8), CFG 1.0;
- the NF4 Qwen2.5-VL encoder, loaded for the encode phase. It is the same encoder
  as Qwen-Image-Edit-2511, so the pre-quantized mirror copy is reused.

```python
from imference_engine import Engine, RuntimeConfig
from imference_engine.qwenimage.builds import recommend_build

build = recommend_build(vram_gb=24)                       # -> "resident"
engine = Engine(runtime=RuntimeConfig(**build.runtime_kwargs())).load()
engine.register_model("qwen-2512", backend="qwenimage", weights_path="auto")
img = engine.generate(model="qwen-2512", width=1328, height=1328,
                      prompt='a storefront with a neon sign that reads "OPEN 24 HOURS"').images[0]
```

`weights_path="auto"` fetches the GGUF; a `.gguf` path selects the same recipe. Both
carry the Lightning defaults (`num_steps=4`, `guidance_scale=1.0`) unless a catalog
row overrides them. img2img works on the lite build.

Measured on an RTX 3090, 1024x1024 (2026-11-13):

| build | VRAM floor | offload | peak VRAM | 4 steps | 8 steps |
|---|---|---|---|---|---|
| `resident` | 22 GiB | none | 21.0 GB | 9.2 s | 17.6 s |
| `offload` | 15.5 GiB | model | 14.9 GB | 17.2 s | — |
| `minimal` | below | group | ~6.5 GB (Edit build, 8 GB laptop) | host-bound | — |

- 1328x1328 (the model's native 1:1 size) takes 14.3 s resident, 21.3 GB peak.
- Text rendering is exact on the prompts tried, at 4 steps.
- Group offload is functional but its speed depends on the host's CPU and memory
  bandwidth, not on the card: on the (heavily shared) test host it took minutes per
  step, against ~52 s per image measured for the Edit build on a laptop. Treat the
  `minimal` build as "works, speed to be measured on the target machine".

| variable | values | default |
|---|---|---|
| `QWEN_IMAGE_LIGHTNING` | `4` · `8` · `none` | `4` |
| `QWEN_EDIT_ENCODER` | `auto` · `nf4` · `bf16` | `auto` (shared with the Edit backend) |
| `QWEN_EDIT_KEEP_ENCODER` | `auto` · `1` · `0` | `auto` (keep the encoder loaded when VRAM allows) |

`QWEN_IMAGE_LIGHTNING=none` runs the GGUF with true CFG (set `num_steps` ~40 and
`guidance_scale` 4.0 yourself): two passes per step, minutes per image.

CDN layout (under `IMAGE_MODEL_CDN`):

```
Qwen/Qwen-Image/…                                         # base (shared with Qwen-Image)
imference/Qwen-Image-Edit-2511-text-encoder-nf4/…         # NF4 encoder (shared with Edit)
unsloth/Qwen-Image-2512-GGUF/qwen-image-2512-Q4_K_M.gguf
lightx2v/Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-{4,8}steps-V1.0-bf16.safetensors
```

