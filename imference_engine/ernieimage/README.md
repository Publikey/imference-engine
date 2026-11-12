# ERNIE-Image (`ernie-image`)

Baidu's text-to-image model: an 8B single-stream flow-matching transformer, a
**Ministral-3-3B** text encoder and the **FLUX.2 VAE**. **Apache-2.0** and ungated
(the Ministral 3 family is Apache-2.0 too), so it can be served from the cloud API
and mirrored on the CDN. Its strength is legible text and structured layouts
(posters, signs, multi-panel images).

`ErnieImagePipeline` ships in diffusers 0.40; this backend runs on the repo-wide pin.

```python
from imference_engine import Engine, RuntimeConfig
from imference_engine.ernieimage import recommend_build

build = recommend_build(vram_gb=16)            # -> "balanced": Q5_K_M + fp8 encoder
engine = Engine(runtime=RuntimeConfig(**build.runtime_kwargs())).load()
engine.register_model("ernie", backend="ernie-image", weights_path="auto")

img = engine.generate(
    model="ernie",
    prompt='a minimalist poster with the title "JAZZ NIGHT" and the line '
           '"Friday 9 PM - Blue Room" underneath, navy and gold',
).images[0]
```

## What it does

| | |
|---|---|
| text-to-image | 8 steps, `guidance_scale=1.0` (Turbo, DMD-distilled) |
| `source_image` | not supported — diffusers has no ERNIE img2img / edit pipeline |
| `negative_prompt` | inert on Turbo (no CFG at 1.0); honored when `guidance_scale > 1` |
| `scheduler` | ignored (flow matching) |
| `backend_options={"shift": x}` | overrides the fixed flow shift (4.0 for Turbo) for that request |
| sizes | multiples of 16; upstream recommends 1024x1024, 848x1264, 768x1376, 896x1200 and their transposes |
| LoRAs | not supported (the diffusers pipeline has no LoRA loader) |
| prompt enhancer | not loaded (see below) |

The undistilled **ERNIE-Image** (50 steps, CFG ~4.0) goes through the same backend
with `base_model="baidu/ERNIE-Image"` and `defaults: {num_steps: 50,
guidance_scale: 4.0}`. It has not been validated on GPU.

## Weights

`weights_path` accepts:

- **`"auto"`** — the build for this machine (table below), downloaded into the flat
  offline tree from `IMAGE_MODEL_CDN`, or HuggingFace when no CDN is set;
- a **`.gguf`** (unsloth `ERNIE-Image-Turbo-GGUF`);
- a transformer-only **`.safetensors`** (Comfy-Org repackage, civitai finetunes);
- a diffusers repo id or directory.

`base_model` defaults to `baidu/ERNIE-Image-Turbo` and supplies the encoder, VAE,
tokenizer, scheduler and the transformer config.

**GGUF fix.** diffusers' GGUF loader only runs `nn.Linear` layers quantized. The
unsloth files also quantize the patch-embedding `Conv2d`; diffusers leaves it as
packed bytes and fails at the first step. The backend dequantizes that one layer to
bf16 at load (`dequantize_non_linear`).

## Builds per machine

Measured on an RTX 3090 at 1024x1024, 8 steps (2026-11-12). The model is
compute-bound: about 1.5 s per step whatever the storage format.

| build | VRAM floor | transformer | encoder | offload | peak VRAM | s/image | host RAM |
|---|---|---|---|---|---|---|---|
| `quality` | 22 GiB | GGUF Q8_0 (8.7 GB) | bf16 | none | 19.4 GB | 12.5 | 18 GB |
| `balanced` | 15 GiB | GGUF Q5_K_M (5.9 GB) | fp8 | none | 12.9 GB | 14.5 | 18 GB |
| `offload` | 11.5 GiB | GGUF Q8_0 | fp8 | model | 9.8 GB | 19.7 | 21 GB |
| `compact` | 7.5 GiB | GGUF Q4_K_M (5.0 GB) | fp8 | model | 6.1 GB | 16.6 | 17 GB |
| `minimal` | below 7.5 GiB | GGUF Q4_K_M | fp8 | group | 2.7 GB | 17.7 | 19 GB |

What the measurements showed:

- **bf16 does not fit 24 GB resident** (16 GB transformer + 7.7 GB encoder). With
  model offload it peaks at 16.9 GB and took 30 s per image on the test host. There
  is no resident bf16 tier because no 32 GB card was available to measure it.
- **Q8_0 is near-lossless** against bf16 (LPIPS 0.05). Q5_K_M and Q4_K_M change
  details (LPIPS 0.12-0.15) without a visible quality loss.
- **The fp8 encoder moves the composition** more than on FLUX.2 klein (LPIPS ~0.24
  against bf16) but text stayed correct on 12 prompts, including with Q4_K_M.
- **Q8_0 under group offload is too slow** (44 s); the lowest tier streams Q4_K_M.
- Offload timings come from a datacenter host with slow RAM-to-GPU copies; expect
  better on a desktop with the card on a full PCIe link.

Offload is a `RuntimeConfig` setting: with `weights_path="auto"` the backend picks
the file and the encoder storage, and the caller applies `build.runtime_kwargs()`.

Without CUDA (CPU, Apple MPS) `auto` uses the bf16 single file: float8 storage needs
CUDA kernels and GGUF is not validated there.

## Prompt enhancer

Upstream ships a second 3B Ministral (`pe/`) that rewrites the prompt before
encoding, on by default in the diffusers pipeline. It is **not loaded** here:

- it adds 7.7 GB to download and to hold;
- the rewrite is sampled, so the result no longer follows the seed alone;
- on six test prompts it changed the composition and dropped requested details as
  often as it added some; Baidu's own GenEval numbers are lower with it.

Every call passes `use_pe=False`. Callers that want prompt rewriting should do it
before `generate()`.

## Environment

| variable | values | default |
|---|---|---|
| `ERNIE_IMAGE_ENCODER` | `auto` · `bf16` · `fp8` | `auto` (from the VRAM table) |

## CDN layout

Everything a cold load needs, under `IMAGE_MODEL_CDN`:

```
baidu/ERNIE-Image-Turbo/.manifest.json
baidu/ERNIE-Image-Turbo/{model_index.json, LICENSE, scheduler/, tokenizer/,
                         text_encoder/, vae/, transformer/config.json}
unsloth/ERNIE-Image-Turbo-GGUF/ernie-image-turbo-Q8_0.gguf      # "quality", "offload"
unsloth/ERNIE-Image-Turbo-GGUF/ernie-image-turbo-Q5_K_M.gguf    # "balanced"
unsloth/ERNIE-Image-Turbo-GGUF/ernie-image-turbo-Q4_K_M.gguf    # "compact", "minimal"
unsloth/ERNIE-Image-Turbo-GGUF/LICENSE
Comfy-Org/ERNIE-Image/diffusion_models/ernie-image-turbo.safetensors   # bf16 (no-CUDA fallback)
```

Staged with `validation/stage_r2.py --engines ernie-image` (base) and
`--repo-files <repo> --files <file,...>` (builds).
