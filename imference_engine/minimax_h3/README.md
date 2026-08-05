# MiniMax-H3 backend configuration

> ⚠️ **Built against unreleased upstream.** MiniMax-H3's diffusers integration
> is [PR #14355](https://github.com/huggingface/diffusers/pull/14355) (branch
> `minimax-h3`), released in **no** diffusers version. **Validated e2e
> 2026-08-05** on the PR head (`0.40.0.dev0`, torch 2.12/cu126, RTX 6000 Ada
> 48 GB): t2va renders + soundtrack from a ComfyUI-sourced int8 tree
> (`stage_h3_from_comfy.py`) pass `validation/validate_h3.py` — ~14.9 s/step
> at 960×544×124f under `block` offload, ~21 GB VRAM. The PR head is a moving
> target until release; the `[minimax-h3]` extra still **cannot coexist** with
> the repo-wide `diffusers==0.39.0` pin — run it in a dedicated venv/process:
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
| steps | `num_inference_steps` counts sigma grid points *including the terminal 0* (one model eval less). Default 50. **Measured** (960×544×124f, int8, RTX 6000 Ada): 30 steps is frame-quality-equivalent to 50 at ~65 % of the cost; 20 still renders well with slightly softer micro-texture (~46 %). Pin `num_steps: 30` in a catalog row for the fast recipe. |
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

# mux (caller-side, e.g. with diffusers' util — needs `pip install av`;
# encode_video wants a torch waveform, MediaResult carries portable numpy)
import torch
from diffusers.utils.export_utils import encode_video
encode_video(res.frames, fps=res.fps, output_path="fox.mp4",
             audio=torch.from_numpy(res.audio), audio_sample_rate=res.sample_rate)
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
then point a catalog row at the mirror repo id with `H3_MODEL_CDN` set.

## ComfyUI/civitai single-file checkpoints

The community stack (Comfy-Org/MiniMax-H3 on the Hub, mirrored on civitai as
"Minimax H3 INT8/INT4 ConvRot") ships H3 as four single `.safetensors` files —
**~67 GB at int8 versus ~124 GB for the official bf16 repo**, and desktop users
running ComfyUI often have them on disk already. Diffusers cannot load them
directly (no `from_single_file` in the PR, and ConvRot/NVFP4 are ComfyUI
formats), so the engine keeps consuming modular trees and
[`validation/stage_h3_from_comfy.py`](../../validation/stage_h3_from_comfy.py)
converts the stack **offline, once** (streaming, bounded RAM):

```bash
python validation/stage_h3_from_comfy.py \
    --transformer  minimax_h3_fl2va_int8_convrot.safetensors  \  # ~34 GB
    --text-encoder qwen3vl_32b_minimax_h3_int8_convrot.safetensors \  # ~27 GB
    --video-vae    minimax_h3_video_vae_fp16.safetensors \
    --audio-vae    minimax_h3_audio_vae_fp32.safetensors \
    --profile int8            # torchao-requantized tree (~35 GB); or bf16 (~125 GB)
```

ConvRot int8 dequantizes exactly (deterministic block-Hadamard, see
`minimax_h3/comfy_convert.py`); the config skeleton (tokenizer, schedulers,
latents stats) is pulled from the official repo (a few MB). Caveats the script
enforces or documents:

- **"Pruned" DiT files are refused** — they swap the timestep embedder + full
  AdaLN for a low-rank `adaln_t_table`, an architecture the diffusers model
  cannot represent. Use the non-pruned `int8_convrot` file (~34 GB).
- int4 / nvfp4 files need ComfyUI kernels — int8 (or bf16) sources only.
- The Comfy Qwen3-VL keeps only the 50 decoder layers H3 reads. A stack of
  *exactly* 50 layers cannot serve `hidden_states[50]` through transformers
  (the last entry is post-norm, and the upstream encoder rejects it), so the
  staged checkpoint carries **one dummy 51st layer** (ones-norms, ~1e-6-noise
  projections): `hidden_states[50]` is then the dummy layer's input — the raw
  layer-49 output H3 conditions on — and everything downstream of the dummy
  layer (final norm, lm_head) is never read. Config: 51 layers, tied
  embeddings.
- `--profile int8` quantizes a second time on top of ConvRot's ~0.5-2 %/layer
  error. **A/B-validated** (same seed, 50 steps, 960×544): the int8 and bf16
  trees render visually identical frames, and int8 is even marginally faster
  (14.85 vs 15.79 s/step — block-offload streaming overlaps compute either
  way) at half the host RAM (~75 vs ~103 GB) and disk (63 vs 117 GB). int8 is
  the production profile; bf16 is the fidelity/archival reference.
