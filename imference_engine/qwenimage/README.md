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
| `resident` | 22 GiB | none | 21.0 GB | 8.9 s | 17.6 s |
| `offload` | 17 GiB | model | 14.9 GB | 18.6 s | — |
| `minimal` | below | group | 7.4 GB (encoder kept) | 17-20 s | — |

- 1328x1328 (the model's native 1:1 size) takes 14.3 s resident, 21.3 GB peak.
- Text rendering is exact on the prompts tried, at 4 steps.
- Each build was also run under a hard VRAM cap of its card size minus 0.5 GiB
  (`validation/e2e_backend.py --cap-gb`), `offload` at 1328x1328 too (15.6 GB reserved).
- Under model offload the encoder is released before the transformer moves in unless
  both fit: the keep decision counts the transformer still on the CPU. (Before that
  fix a 16 GB card ran out of memory on the first image.)
- Group offload speed depends on the host's CPU and memory bandwidth, not on the
  card: 17-20 s per image on a host with free cores, 12 minutes on a heavily shared
  one.
- Host RAM: 27 GB peak process memory on every build (weight-file page cache included).
- **Real cards** (2026-11-13, from the CDN): an RTX 5060 Ti 16 GB reports 15.48 GiB,
  and the `offload` floor is set above it on purpose, so `auto` gives it `minimal`:
  17.0 s at 7.9 GB reserved, against 20.8 s at 15.4 GB for `offload` (15.6 GB at
  1328x1328, on a 16.6 GB card). An RTX 3070 8 GB runs `minimal` in 22.7 s (6.9 GB
  reserved, encoder released each request). `offload` is what 20 GB cards get.
- The Edit lite build on the same 16 GB card: model offload 32.7 s at 16.3 GB reserved
  (0.26 GiB of headroom), group offload 29.5 s at 8.7 GB. Prefer group offload on 16 GB.

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

## Qwen-Image-Layered (`qwenimage-layered`)

Decomposes an image into transparent RGBA layers (background first) — the only open
model that does. Apache-2.0. Stacked back in order the layers reproduce the input;
individually they can be moved, recolored or dropped.

```python
engine.register_model("layers", backend="qwenimage-layered", weights_path="auto")
r = engine.generate(model="layers", prompt="", source_image=img,
                    backend_options={"layers": 4})
r.images[0]        # the recomposed image (all layers stacked)
r.layers[0]        # [PIL RGBA, ...] background first
```

| | |
|---|---|
| `source_image` | required — the image to decompose |
| `prompt` | a caption of the whole image; empty = the pipeline captions it itself |
| `backend_options` | `layers` (default 4), `resolution` (640, the trained bucket, or 1024) |
| result | `media[i]` = recomposed image, `MediaResult.layers[i]` = its RGBA layers |
| recipe | 50 steps, true CFG 4.0 — there is no Lightning LoRA for this checkpoint |

Weights: `"auto"` (unsloth `qwen-image-layered-Q4_K_M.gguf` + the NF4 encoder), a
`.gguf`, a transformer-only `.safetensors`, or the diffusers repo. `base_model`
defaults to `Qwen/Qwen-Image-Layered`: its encoder is the shared Qwen2.5-VL file, but
its **VAE is specific** (it carries the alpha channel), so this base cannot be
replaced by `Qwen/Qwen-Image`.

Measured on an RTX 3090 (2026-11-13), GGUF Q4_K_M + NF4 encoder, 640 bucket, 4 layers:

| mode | peak VRAM | time |
|---|---|---|
| resident | 20.0 GB | 8 s per step — 6 min 40 s at 50 steps, 2 min 40 s at 20 steps |
| model offload | 19.8 GB | 7.5 s per step — no memory saving (the NF4 encoder stays on the card) |

- On two test images (a street storefront, a flat illustration) the split was clean:
  the neon sign, a lamp post and a pavement sign each came out on their own layer
  with the background repainted behind them; the recomposed image matched the source.
  A layer can come back empty when the image has fewer separable elements than
  `layers`.
- 20 steps already gave a usable split; 50 is the upstream recipe.
- It is a tool, not an interactive generator: plan for minutes per image.
- **24 GB cards only for now.** Neither offload mode was shown to bring it under
  16 GB: model offload saved nothing here, and group offload was not timed (see the
  host-bound note in the 2512 section).

GGUF note: the layered checkpoint has an extra embedding table that diffusers' GGUF
loader leaves as raw bytes (the first step then fails on a shape mismatch). The lite
loader restores it (`core/gguf.py::dequantize_non_linear`).

