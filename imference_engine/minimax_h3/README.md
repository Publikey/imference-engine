# MiniMax-H3 backend configuration

> ⚠️ **Built against unreleased upstream — NOT yet validated e2e.** MiniMax-H3's
> diffusers integration is [PR #14355](https://github.com/huggingface/diffusers/pull/14355)
> (branch `minimax-h3`), released in **no** diffusers version. This backend was
> written against that branch's blocks/doc and its structural tests pass, but the
> end-to-end GPU validation (`validation/validate_h3.py`) can only run once the
> PR is installable next to a supported stack. Until then the `[minimax-h3]`
> extra **cannot coexist** with the repo-wide `diffusers==0.39.0` pin — run it in
> a dedicated venv/process:
>
> ```bash
> python -m venv .venv-h3 && . .venv-h3/bin/activate
> pip install -e ".[minimax-h3,dev]"
> pip install git+https://github.com/huggingface/diffusers.git@refs/pull/14355/head
> ```
>
> The day the PR ships in a release: pin it in the extra, run
> `validation/validate_h3.py`, and fold the pin into the repo-wide one.

MiniMax-H3 (MiniMaxAI, Aug 2026) generates video **and its soundtrack jointly**
— one 33B transformer denoises a single packed sequence holding text, keyframe,
video and audio rows; there is no vocoder and no post-hoc audio pass. Components:
the `MiniMaxH3Transformer3DModel` DiT (~61.7 GB bf16), a **Qwen3-VL-32B**
conditioner (~62.1 GB bf16 — H3 reads the *unnormalized* layer-50 hidden states),
a video VAE, an audio VAE, and two `MiniMaxH3Scheduler`s (video `shift=12`,
audio `shift=3`). Integrated as **Modular Diffusers blocks only** — same load
pattern as [Anima](../anima/README.md).

## What's in scope

| Task | Status |
|---|---|
| `t2va` — text → video+audio | ✅ `generate_video(prompt=...)` |
| `fl2va` — first/last keyframe → video+audio | ✅ `image=` / `last_image=` |
| `ref2va` — omni-references (12 media, separate `transformer_ref/`) | ❌ out of scope (another ~40 GB partition) |

**One variant serves t2v AND i2v** — unlike Wan, the same checkpoint + pipeline
routes by which inputs the request carries, so there is a single builtin variant
(`minimax-h3`), not a t2v/i2v pair, and one resident pipeline covers both.

## MiniMax-H3 specifics

| Aspect | Behaviour |
|---|---|
| Load | `ModularPipeline.from_pretrained(repo)` + strict `load_components` (specs repointed at the offline tree, à la Anima). Only the FL2VA half is fetched — never `transformer_ref/`, never the original checkpoint folders. |
| Quant | `H3_PROFILE=int8` (default): torchao `Int8WeightOnlyConfig(version=2)` on the transformer + Qwen3-VL, upstream exclusion lists. Applied **at load** from a bf16 tree, or **skipped** when the tree is pre-quantized (staged with [`validation/stage_h3_int8.py`](../../validation/stage_h3_int8.py) — the intended production path: ~50 GB mirror, fast cold loads). `bf16` = multi-GPU only. |
| Offload | `H3_OFFLOAD_MODE`: `block` (24-32 GB VRAM — transformer block-streamed, Qwen3-VL leaf, VAEs resident), `leaf` (12-16 GB — video VAE offloaded too; use a small canvas), `none` (80 GB+/multi-GPU), `auto` picks by VRAM. Offloaded weights live in **host RAM: ~75 GB at int8** — the engine warns at load on smaller boxes. |
| negative_prompt / guidance | **Do not exist** — the checkpoint is guidance-distilled (one forward per step). The backend ignores + warns. |
| scheduler | Two `MiniMaxH3Scheduler`s from the repo, untouched. No `shift` knob in V1. |
| Frames / duration | Fixed **24 fps**; `num_frames` snaps up to the next `17n+5`; the **aligned** duration must stay in 5–15 s (aligned counts 124…345). Checked engine-side before any weight loads. |
| Canvas | `width`/`height` omitted → the model's native 768-short-edge canvas from the keyframe's (or 16:9) aspect. When set: multiples of 32. **960×544 is ~2.3× faster per step** than the native 1344×768. |
| steps | `num_inference_steps` counts sigma grid points *including the terminal 0* (one model eval less). Default 50 — un-validated placeholder, refine via catalog `num_steps` once measured. |
| Audio | Always generated. `MediaResult.audio` = `(2, n)` float32 numpy stereo waveform, `MediaResult.sample_rate` in Hz. Muxing is the caller's: `encode_video(res.frames, fps=24, audio=res.audio, audio_sample_rate=res.sample_rate)`. |
| Seed | One CPU generator, three draws (keyframe noise → video noise → audio noise) — same seed = same video **and** soundtrack. |
| offline / CDN | Same contract as every backend: `H3_MODEL_CDN` + flat tree (`namespace="video"`, sentinel `modular_model_index.json`). |

## Env ↔ param

| env | param | default | notes |
|---|---|---|---|
| `H3_DEVICE` | `device` | `auto` | |
| `H3_PROFILE` | `memory_profile` | `auto` (→ `int8`) | `int8` \| `bf16` |
| `H3_OFFLOAD_MODE` | `offload_mode` | `auto` | `block` \| `leaf` \| `none` |
| `H3_MAX_RESIDENT` | `max_resident_variants` | `1` | ~75 GB host RAM per resident pipeline |
| `H3_MODEL_CACHE` | `model_cache_dir` | — | flat-tree root |
| `H3_MODEL_CDN` | `model_cdn` | — | R2/S3 mirror base URL |
| `H3_VAE_TILING` | `vae_tiling` | `1` | best-effort |
| `H3_ATTENTION_BACKEND` | `attention_backend` | — | e.g. `_flash_3_hub` (Hopper) |

## Example

```python
from imference_engine.minimax_h3 import MiniMaxH3Engine, MiniMaxH3RuntimeConfig

engine = MiniMaxH3Engine(runtime=MiniMaxH3RuntimeConfig(device="auto")).load()

# t2v (+ soundtrack), fast canvas
res = engine.generate_video(
    prompt="a red fox trotting through a snowy pine forest, snow crunching underfoot",
    width=960, height=544, num_frames=124, seed=42,
)
res.frames        # list[PIL.Image], 24 fps
res.audio         # (2, n) float32 numpy stereo
res.sample_rate   # Hz

# i2v: start keyframe (canvas follows its aspect when width/height omitted)
res = engine.generate_video(prompt="the fox leaps over a fallen log", image=pil_img)

# mux (caller-side, e.g. with diffusers' util)
from diffusers.utils.export_utils import encode_video
encode_video(res.frames, fps=res.fps, output_path="fox.mp4",
             audio=res.audio, audio_sample_rate=res.sample_rate)
```

## Catalog

`kind: video` rows with `engine: minimax_h3` (shared `models.yml` with Wan rows
is fine — each engine keeps its own archs). A row exists mostly to point at a
pre-quantized mirror or pin a step recipe:

```yaml
- name: h3-int8
  kind: video
  engine: minimax_h3
  mode: t2v            # convention — the variant still serves i2v too
  repo: imference/MiniMax-H3-int8
  num_steps: 40
```

## Production path (decided design)

Target hardware **24 GB VRAM / 64 GB RAM** → `int8` + `block` offload. Weights
come from a **pre-quantized R2 mirror**: run `validation/stage_h3_int8.py` once
on a big-RAM box (quantizes sequentially on CPU, serializes, uploads, ~50 GB),
then point a catalog row at the mirror repo id with `H3_MODEL_CDN` set. The
Civitai/ComfyUI "convrot" single-file quants (int4/nvfp4) are **not** loadable
by diffusers (no `from_single_file` in the PR) — a convrot→diffusers converter
is a possible V2 if 12 GB cards become a target.
