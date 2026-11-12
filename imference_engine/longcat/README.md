# LongCat-Image (`longcat`, `longcat-edit`)

Meituan's 6B flow-matching model, with a **Qwen2.5-VL-7B** text encoder and the FLUX
VAE. **Apache-2.0** and ungated. Two backends, one per diffusers pipeline:

| backend | pipeline | default checkpoint | recipe |
|---|---|---|---|
| `longcat` | `LongCatImagePipeline` (text-to-image) | `LongCat-Image` | 50 steps, CFG 4.0, negative prompt honored |
| `longcat-edit` | `LongCatImageEditPipeline` (instruction editing) | `LongCat-Image-Edit-Turbo` | 8 steps, guidance 1.0 |

```python
from imference_engine import Engine, RuntimeConfig
from imference_engine.longcat import recommend_build

build = recommend_build(vram_gb=12)                 # -> "offload": fp8 + fp8, model offload
engine = Engine(runtime=RuntimeConfig(**build.runtime_kwargs())).load()
engine.register_model("longcat", backend="longcat", weights_path="auto")
engine.register_model("longcat-edit", backend="longcat-edit", weights_path="auto")

img = engine.generate(model="longcat", prompt="a red fox in a snowy forest").images[0]
out = engine.generate(model="longcat-edit", source_image=img,
                      prompt="make it night with a full moon, keep the fox unchanged").images[0]
```

Both pipelines are in diffusers 0.40; the backends run on the repo-wide pin.

## What they do

| | `longcat` | `longcat-edit` |
|---|---|---|
| `source_image` | not supported (no img2img pipeline) | required — the image to edit |
| output size | `width` x `height` | follows the input image |
| `negative_prompt` | honored | honored when guidance > 1 (not on Turbo) |
| `strength`, `scheduler` | ignored | ignored |
| `backend_options` | `prompt_rewrite` (default false) | — |
| LoRAs | not supported (no loader in the diffusers pipelines) | same |

The undistilled `LongCat-Image-Edit` checkpoint also loads on `longcat-edit`
(`weights_path="meituan-longcat/LongCat-Image-Edit"`); it gets 50 steps / guidance 4.5
by default. It has not been run on GPU here — Edit-Turbo has.

**Prompt rewrite.** Upstream has the encoder rewrite the prompt before encoding it,
on by default. It is opt-in here (`backend_options={"prompt_rewrite": True}`): it
costs an LLM generation per request and renders something other than what was asked.
Unlike ERNIE's enhancer it needs no extra weights, so it is offered.

## Weights

`weights_path` accepts `"auto"` (the backend's default checkpoint), a LongCat repo id
or directory, or a transformer `.safetensors` in the diffusers key layout (finetunes).

There is **no GGUF path**: diffusers 0.40 has no single-file loader for the LongCat
transformer, so the community GGUF files cannot be loaded. The low-memory builds store
the repo's bf16 weights in fp8 instead.

**One encoder for every checkpoint.** All LongCat repos carry the same 16.6 GB encoder.
A checkpoint contributes only `model_index.json`, `scheduler/` and `transformer/`; the
encoder, VAE, tokenizer and processor come from `base_model` (default
`meituan-longcat/LongCat-Image`). Each checkpoint keeps its own scheduler config —
Edit-Turbo's differs from the others.

## Builds per machine

Measured on an RTX 3090 at 1024x1024 (2026-11-13). Text-to-image times are 30-step
measurements scaled to the 50-step recipe.

| build | VRAM floor | transformer | encoder | offload | peak VRAM | t2i (50 steps) | edit Turbo (8 steps) |
|---|---|---|---|---|---|---|---|
| `resident` | 22 GiB | fp8 | fp8 | none | 19.6 GB | 77 s | 14.8 s |
| `quality` | 15 GiB | bf16 | fp8 | model | 13.1 GB | 93 s | — |
| `offload` | 11.5 GiB | fp8 | fp8 | model | 10.3 GB (10.7 edit) | 92 s | 23.6 s |
| `minimal` | below | fp8 | fp8 | group | not timed | — | — |

- **fp8 is visually identical to bf16** on the prompts tried, legible text included.
- **Model offload costs about 10 s per image**; the encoder runs once per request.
- Plain bf16 with model offload (the upstream recipe) peaks at 17.0 GB for
  text-to-image and 18.4 GB for editing, as the model cards say (~17-18 GB).
- `minimal` is wired but not timed: group offload speed depends on the host, and the
  test host was not representative. Cards under 11.5 GiB are therefore unverified.
- Text-to-image is slow because it is undistilled (100 transformer passes): over a
  minute per image. Editing with Turbo is the interactive use.

Offload is a `RuntimeConfig` setting: the backend applies the storage choice, the
caller applies `build.runtime_kwargs()`.

## Environment

| variable | values | default |
|---|---|---|
| `LONGCAT_TRANSFORMER` | `auto` · `bf16` · `fp8` | `auto` (from the VRAM table) |
| `LONGCAT_ENCODER` | `auto` · `bf16` · `fp8` | `auto` |

## CDN layout

```
meituan-longcat/LongCat-Image/.manifest.json
meituan-longcat/LongCat-Image/{model_index.json, scheduler/, transformer/,
                               text_encoder/, tokenizer/, text_processor/, vae/}
meituan-longcat/LongCat-Image-Edit-Turbo/.manifest.json
meituan-longcat/LongCat-Image-Edit-Turbo/{model_index.json, scheduler/, transformer/}
```

Staged with `validation/stage_r2.py --engines longcat` (base + text-to-image) and
`--repo-patterns meituan-longcat/LongCat-Image-Edit-Turbo --patterns
"model_index.json,scheduler/*,transformer/*"` (checkpoint only).
