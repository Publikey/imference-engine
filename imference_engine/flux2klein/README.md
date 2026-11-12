# FLUX.2 [klein] 4B (`flux2-klein`)

Black Forest Labs' small FLUX.2 model: a 3.9B rectified-flow transformer, a
**Qwen3-4B** text encoder and the **FLUX.2 VAE**. **Apache-2.0** (weights, VAE and
encoder) and ungated, so it can be served from the cloud API and mirrored on the
CDN. The 9B klein and FLUX.2 [dev] are non-commercial and are not this backend's
target.

`Flux2KleinPipeline` has shipped in diffusers since 0.37; this backend runs on the
repo-wide 0.40.0 pin.

```python
from imference_engine import Engine, RuntimeConfig
from imference_engine.flux2klein import recommend_build

build = recommend_build(vram_gb=12)            # -> "compact": Q4_K_M + fp8 encoder
engine = Engine(runtime=RuntimeConfig(**build.runtime_kwargs())).load()
engine.register_model("klein", backend="flux2-klein", weights_path="auto")

img = engine.generate(model="klein", prompt="a red fox in a snowy forest").images[0]
edit = engine.generate(model="klein", prompt="make it night, keep the fox",
                       source_image=img).images[0]
```

## What it does

| | |
|---|---|
| text-to-image | `generate(prompt=...)` — 4 steps, `guidance_scale=1.0` |
| reference editing | `generate(prompt=<instruction>, source_image=<image or list>)` |
| `negative_prompt` | ignored on klein 4B (step-distilled); honored on klein **Base** |
| `strength` | ignored — the image is a condition, not a noised start |
| `scheduler`, `shift` | ignored — flow matching, `mu` computed by the pipeline |
| output size (edit) | the first reference's size (capped at ~1 MP by the pipeline) |
| LoRAs | not enabled yet (`supports_loras = False`) |

Editing uses the same pipeline and the same resident weights: the reference
latents are appended to the token sequence, so an edit costs about 1.5x a
text-to-image call at the same size.

## Weights

`weights_path` accepts:

- **`"auto"`** — the build for this machine (table below), downloaded into the flat
  offline tree from `IMAGE_MODEL_CDN`, or HuggingFace when no CDN is set;
- a transformer-only **`.safetensors`** in BFL's native layout: bf16
  (`flux-2-klein-4b.safetensors`) or the official **scaled fp8**
  (`flux-2-klein-4b-fp8.safetensors`), dequantized in memory by `convert.py` and
  kept fp8-resident;
- a **`.gguf`** (unsloth `FLUX.2-klein-4B-GGUF`);
- a diffusers repo id or directory.

`base_model` defaults to `black-forest-labs/FLUX.2-klein-4B` and supplies the
encoder, VAE, tokenizer, scheduler and the transformer **config**. The config must
come from there: diffusers 0.40's single-file detection only knows FLUX.2 [dev] and
would build the 32B layout for a klein file.

NVFP4 files are refused with an explicit error.

**klein Base 4B** (undistilled, true CFG) uses the same backend: register its
single file with `base_model="black-forest-labs/FLUX.2-klein-base-4B"` and
`defaults: {num_steps: 50, guidance_scale: 4.0}`. Validated at 30 steps / CFG 4.0
(35 s per image on an RTX 3090, 18.6 GB; 21.2 GB with a negative prompt).

## Builds per machine

Measured on an RTX 3090 at 1024x1024, 4 steps (2026-11-12). `recommend_build(vram_gb)`
returns the first row whose floor fits; `build.runtime_kwargs()` gives the matching
`RuntimeConfig` fields.

| build | VRAM floor | transformer | encoder | offload | peak VRAM | s/image | host RAM |
|---|---|---|---|---|---|---|---|
| `full` | 22 GiB | bf16 (7.75 GB) | bf16 | none | 18.6 GB | 2.7 | 9 GB |
| `balanced` | 14 GiB | GGUF Q8_0 (4.3 GB) | fp8 | none | 11.1 GB | 2.9 | 17 GB |
| `compact` | 11.5 GiB | GGUF Q4_K_M (2.6 GB) | fp8 | none | 9.4 GB | 3.2 | 15 GB |
| `offload` | 7 GiB | GGUF Q8_0 | fp8 | model | 5.1 GB | 7.7 | 17 GB |
| `minimal` | below 7 GiB | GGUF Q4_K_M | fp8 | group | 2.8 GB | 6.7 | 17 GB |

What the measurements showed:

- The **encoder is as heavy as the transformer** (8 GB bf16). Quantizing the
  transformer alone never gets below ~13 GB; storing the encoder in fp8 (float8
  storage, bf16 compute) is what makes 12-16 GB cards work.
- **Q8_0 is visually identical to bf16** (LPIPS 0.02 on fixed seeds). Q4_K_M and the
  fp8 encoder change details of the composition (LPIPS 0.13-0.20) without a visible
  quality loss. On legible text, all builds including bf16 make occasional spelling
  errors at 4 steps; quantization did not make them more frequent on 12 prompts.
- **BFL's official fp8 transformer is not recommended**: same size as Q8_0, further
  from bf16 (LPIPS 0.16). It still loads when registered explicitly.
- A **4-bit NF4 encoder** was visibly worse on text and needs bitsandbytes (CUDA
  only): not wired.
- Offload timings are from a datacenter host with slow RAM-to-GPU copies; expect
  the offload builds to be faster on a desktop with the card on a full PCIe link.

Offload is a `RuntimeConfig` setting, not a backend one. With `weights_path="auto"`
the backend picks the file and the encoder storage; the caller applies
`build.runtime_kwargs()` to get the measured VRAM.

Without CUDA (CPU, Apple MPS) `auto` uses plain bf16 weights: float8 storage needs
CUDA kernels and GGUF is not validated there.

## Environment

| variable | values | default |
|---|---|---|
| `FLUX2_KLEIN_ENCODER` | `auto` · `bf16` · `fp8` | `auto` (from the VRAM table) |
| `FLUX2_KLEIN_FP8_STORAGE` | `auto` · `1` · `0` | `auto` (on for the fp8 single file) |

## CDN layout

Everything a cold load needs, under `IMAGE_MODEL_CDN`:

```
black-forest-labs/FLUX.2-klein-4B/.manifest.json
black-forest-labs/FLUX.2-klein-4B/{model_index.json, LICENSE.md, scheduler/, tokenizer/,
                                   text_encoder/, vae/, transformer/config.json}
black-forest-labs/FLUX.2-klein-4B/flux-2-klein-4b.safetensors      # "full"
unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf             # "balanced", "offload"
unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_M.gguf           # "compact", "minimal"
unsloth/FLUX.2-klein-4B-GGUF/LICENSE.md
```

Staged with `validation/stage_r2.py --engines flux2-klein` (base) and
`--repo-files <repo> --files <file,...>` (builds).
