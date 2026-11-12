# Lumina-Image 2.0 (`lumina2`)

Alpha-VLLM's small text-to-image model: a 2.6B flow-matching transformer, a
**Gemma-2-2b** text encoder and the FLUX.1 VAE. Its value is the anime finetune
ecosystem built on it — **Neta Lumina** and **NetaYume Lumina** — which load as
shipped (ComfyUI single files).

`Lumina2Pipeline` ships in diffusers 0.40; this backend runs on the repo-wide pin.

```python
from imference_engine import Engine, RuntimeConfig
from imference_engine.lumina2 import recommend_build

build = recommend_build(vram_gb=8)             # -> "offload": Q8_0 + fp8 encoder, model offload
engine = Engine(runtime=RuntimeConfig(**build.runtime_kwargs())).load()
engine.register_model("lumina", backend="lumina2", weights_path="auto")
engine.register_model("netayume", backend="lumina2",
                      weights_path="/models/NetaYume_v4_all_in_one.safetensors")

img = engine.generate(model="netayume", num_steps=40, guidance_scale=5.0,
                      prompt="1girl, silver hair, school uniform, cherry blossoms",
                      negative_prompt="blurry, low quality").images[0]
```

## Licences

| part | licence |
|---|---|
| Lumina-Image 2.0 transformer | Apache-2.0 |
| Neta Lumina, NetaYume Lumina | Apache-2.0 (model cards) |
| FLUX.1 VAE (as redistributed upstream) | Apache-2.0 repo |
| **Gemma-2-2b encoder** | **Gemma Terms of Use** + Prohibited Use Policy |

The Gemma terms allow redistribution when the terms travel with the weights, which
the CDN mirror does (`GEMMA_TERMS.md`, `NOTICE`). They also bind every **use**: the
Prohibited Use Policy forbids, among other things, generating sexually explicit
content. That applies to anything rendered through this backend, whatever the
finetune — check it against the catalog before offering NSFW finetunes on it.

## What it does

| | |
|---|---|
| text-to-image | 30 steps, `guidance_scale=4.0`, true CFG (two passes per step) |
| `negative_prompt` | honored |
| `source_image` | not supported — diffusers has no Lumina img2img pipeline |
| `scheduler` | ignored (flow matching) |
| `backend_options={"shift": x}` | overrides the fixed flow shift (6.0) for that request |
| `backend_options={"cfg_trunc_ratio": r}` | guidance only on the first part of the schedule (default 1.0 = every step) |
| `backend_options={"system_prompt": "…"}` | replaces the pipeline's system prompt |
| LoRAs | not enabled yet (`supports_loras = False`; the diffusers pipeline has a loader) |

Per-model recipes belong in the catalog row: base 30-50 steps / CFG 4; Neta Lumina
30 steps / CFG 4-5.5; NetaYume 40-50 steps / CFG 4-7.

`cfg_trunc_ratio=0.25` (the upstream README example) is 1.6x faster but visibly
degrades detail and structure in our tests, so it is not the default.

## Weights

`weights_path` accepts:

- **`"auto"`** — the build for this machine (table below);
- a transformer-only **`.safetensors`** (`Unet/…`, Comfy-Org `lumina_2_model_bf16`);
- an **all-in-one `.safetensors`** (transformer + Gemma + VAE, the usual civitai
  form). diffusers cannot load these; the backend reads only the
  `model.diffusion_model.*` tensors and converts them. The encoder and VAE bundled in
  the file are the stock ones and are taken from the base instead;
- a **`.gguf`** (calcuis `lumina-gguf`, neta-art `neta-lumina-gguf`);
- a diffusers repo id or directory.

**Which weights are "Lumina-Image 2.0".** The official checkpoint is
`consolidated.00-of-01.pth`; the Comfy-Org file, the GGUFs and the finetunes all
derive from it. The `transformer/` folder of `Alpha-VLLM/Lumina-Image-2.0` holds
*different* weights (every tensor differs; the images differ too). The builds below
use the official checkpoint; the repo folder is only used if you register the repo
id itself.

## Base components

`base_model` defaults to `Alpha-VLLM/Lumina-Image-2.0`, whose Gemma encoder is stored
in fp32 (10.4 GB). With `IMAGE_MODEL_CDN` set, the backend first looks for the slim
mirror `imference/Lumina-Image-2.0-bf16`: the same components with the encoder
re-saved in bf16 (5.2 GB) plus `LICENSE`, `NOTICE` and `GEMMA_TERMS.md`. It is built
by `validation/stage_lumina_slim.py` and exists only on the mirror.

## Builds per machine

Measured on an RTX 3090 at 1024x1024, 30 steps with CFG (2026-11-12).

| build | VRAM floor | transformer | encoder | offload | peak VRAM | s/image |
|---|---|---|---|---|---|---|
| `full` | 18 GiB | bf16 (5.2 GB) | bf16 | none | 13.2 GB | 28 |
| `balanced` | 15 GiB | bf16 | fp8 | none | 11.2 GB | 28 |
| `compact` | 11.5 GiB | GGUF Q8_0 (2.8 GB) | fp8 | none | 8.8 GB | 31 |
| `offload` | 5.5 GiB | GGUF Q8_0 | fp8 | model | 3.4 GB | 33 |
| `minimal` | below 5.5 GiB | GGUF Q4_K_M (1.5 GB) | fp8 | group | 2.7 GB | 58 |

- The model is compute-bound, so quantization saves memory, not time (GGUF is a
  little slower).
- **Model offload costs about 2 s** because the encoder runs once per request: it is
  the right build for 6-10 GB cards.
- Q8_0 is indistinguishable from bf16 (LPIPS 0.01); Q4_K_M and the fp8 encoder
  change details (LPIPS 0.13-0.19) without a visible quality loss.
- The fp8 encoder keeps its token embedding in bf16 (0.6 GB) so that the pipeline,
  which reads the encoder's dtype, keeps working in bf16.

## Environment

| variable | values | default |
|---|---|---|
| `LUMINA2_ENCODER` | `auto` · `bf16` · `fp8` | `auto` (from the VRAM table) |

## CDN layout

```
imference/Lumina-Image-2.0-bf16/…                       # slim base (bf16 Gemma, licences)
Comfy-Org/Lumina_Image_2.0_Repackaged/split_files/diffusion_models/lumina_2_model_bf16.safetensors
calcuis/lumina-gguf/lumina2-q8_0.gguf
calcuis/lumina-gguf/lumina2-q4_k_m.gguf
duongve/NetaYume-Lumina-Image-2.0/Unet/v4/NetaYumev4_unet.safetensors   # finetune
```
