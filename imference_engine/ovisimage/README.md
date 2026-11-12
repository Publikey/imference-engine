# Ovis-Image (`ovis-image`)

Alibaba AIDC's 7B text-to-image model, built for legible text: a FLUX-style
flow-matching transformer, the **Ovis2.5-2B** language tower (`Qwen3Model`) as text
encoder, and the FLUX VAE. **Apache-2.0**, ungated. `OvisImagePipeline` ships in
diffusers 0.40.

```python
from imference_engine import Engine, RuntimeConfig
from imference_engine.ovisimage import recommend_build

build = recommend_build(vram_gb=12)                # -> "offload": fp8 + fp8, model offload
engine = Engine(runtime=RuntimeConfig(**build.runtime_kwargs())).load()
engine.register_model("ovis", backend="ovis-image", weights_path="auto")
img = engine.generate(
    model="ovis",
    prompt='a storefront with a neon sign that reads "OPEN 24 HOURS", rainy night',
).images[0]
```

## What it does

| | |
|---|---|
| text-to-image | 50 steps, true CFG (`guidance_scale=5.0`), negative prompt honored |
| `source_image` | not supported — diffusers has no Ovis img2img / edit pipeline |
| `scheduler`, `strength` | ignored |
| LoRAs | not supported (no loader in the diffusers pipeline) |

Undistilled, so slow: 100 transformer passes per image, about a minute and a half
on an RTX 3090.

## Weights

`weights_path` accepts `"auto"` (`AIDC-AI/Ovis-Image-7B`), a repo id or directory in
the diffusers layout, or a transformer `.safetensors` in the diffusers key layout.

There is **no GGUF path**: diffusers 0.40 has no single-file loader for the Ovis
transformer, and the public GGUF files are stable-diffusion.cpp builds. The
low-memory builds store the repo's bf16 weights in fp8 instead.

Only the diffusers part of the repo is used and mirrored. The original 29 GB
single-file checkpoint and the full Ovis2.5-2B VLM at the repo root are not.

## Builds per machine

Measured on an RTX 3090 at 1024x1024 (2026-11-13). Times are 30-step measurements
scaled to the 50-step recipe.

| build | VRAM floor | transformer | encoder | offload | peak VRAM | s/image |
|---|---|---|---|---|---|---|
| `quality` | 22 GiB | bf16 | fp8 | none | 19.5 GB | 88 |
| `resident` | 18 GiB | fp8 | fp8 | none | 13.6 GB | 93 |
| `offload` | 11 GiB | fp8 | fp8 | model | 9.6 GB | 103 |
| `minimal` | below | fp8 | fp8 | group | not timed | — |

- **fp8 is visually near-identical to bf16**; text stayed correct (one prompt
  repeated a price tag in fp8).
- Plain bf16 with model offload peaks at 15.4 GB — the limit of a 16 GB card — so
  fp8 is used below 22 GiB.
- `minimal` is wired but not timed (group offload speed depends on the host); cards
  under 11 GiB are unverified.

## Environment

| variable | values | default |
|---|---|---|
| `OVIS_IMAGE_TRANSFORMER` | `auto` · `bf16` · `fp8` | `auto` (from the VRAM table) |
| `OVIS_IMAGE_ENCODER` | `auto` · `bf16` · `fp8` | `auto` |

## CDN layout

```
AIDC-AI/Ovis-Image-7B/.manifest.json
AIDC-AI/Ovis-Image-7B/{model_index.json, LICENSE, NOTICE, scheduler/, tokenizer/,
                       text_encoder/, vae/, transformer/}
```

Staged with `validation/stage_r2.py --engines ovis-image`.
