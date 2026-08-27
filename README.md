# imference-engine

**One Python API for state-of-the-art diffusion. Eight image models and Wan
video behind a single `Engine`, offline-first, from a 6 GB laptop GPU to a
multi-model cloud fleet.**

`imference-engine` is a unified, Diffusers-based inference engine. Register a
checkpoint, call `generate()`, get PIL images back — the same three lines whether
you're driving SDXL, FLUX, Qwen-Image or Anima. It handles the parts that are
tedious to get right per model: transformer-only checkpoints with shared base
components, VRAM-aware batching, multi-tier model residency (GPU + CPU LRU),
offline weight resolution from a CDN mirror, weighted prompts, and img2img — all
behind one small, transport-agnostic surface.

It ships as the **sidecar for [Imference Desktop](#sidecar-or-standalone)** and
powers the Imference GPU worker fleet — but it's a plain `pip`-installable
library with **no ties to either**, so you can embed it in your own worker, batch
script, or app.

### Why it's nice to use

- **One API, many models.** SDXL · SD 1.5 · Z-Image · FLUX.1 · Chroma ·
  Qwen-Image · Anima · Krea 2 Turbo, plus **Wan 2.2** text/image-to-video and
  **MiniMax-H3** joint video+audio — each a first-class backend, none needing
  bespoke glue in your code.
- **Offline-first.** Point `*_MODEL_CDN` at an R2/S3 mirror and cold loads pull
  zero bytes from HuggingFace — immune to a repo going gated or disappearing.
  Ships with a staging tool ([`validation/stage_r2.py`](validation/stage_r2.py)).
- **Fits the hardware.** `enable_offload` runs a 12–20 B DiT on consumer VRAM;
  the multi-tier `ModelManager` turns 10–30 s model swaps into ~0.5 s promotions
  on a cloud box with room to spare.
- **Sane precedence.** Per-request → per-model → per-engine → global defaults, so
  a catalog row carries a checkpoint's recipe (e.g. a 4-step Lightning) and
  callers override only what they mean to.
- **Transport-agnostic.** `generate()` returns frames, seeds and per-image
  errors. What happens next — upload, webhook, hand to Electron — is yours.
- **Validated.** The v0.4.x stack (diffusers 0.40.0 · transformers 5.4.0 ·
  peft 0.19.1) is GPU-validated end-to-end: seven image backends + Wan 2.2
  t2v/i2v + MiniMax-H3 (2026-08-26), and Krea 2 Turbo — official scaled-fp8
  AND a civitai fp8 finetune — (2026-08-27). See [`validation/`](validation/).

---

## Install

```bash
# everything (all image backends + dev tools)
pip install -e ".[runtime,dev]"

# just the backend(s) you need — each has its own extra
pip install -e ".[sdxl,dev]"
pip install -e ".[flux,dev]"

# Wan video (adds gguf / imageio)
pip install -e ".[wan,dev]"

# pure-Python pieces only (catalog loader, batch sizer — no torch)
pip install -e ".[dev]"
```

Extras: `sdxl` · `sd15` · `zimage` · `flux` · `chroma` · `qwenimage` · `anima`
· `krea2` (and `runtime` = all eight), `wan`, `minimax-h3`, `stage` (R2
staging, boto3), `dev`. Every extra shares the one repo-wide
`diffusers==0.40.0` pin.

> **Weighted prompts** (sd-embed, `(word:1.3)` / `BREAK`) are optional and
> GitHub-only — install separately so its unconstrained torch pin can't clobber
> your CUDA build; the engine falls back to raw prompts if absent:
> ```bash
> pip install --no-deps "sd-embed @ https://github.com/xhinker/sd_embed/archive/refs/heads/main.tar.gz"
> ```

Install the GPU torch build matching your hardware **first** (the extras only
pin `torch>=2.6`, so pip would otherwise pull the CPU wheel):

- **NVIDIA (Windows/Linux):** `pip install torch --index-url https://download.pytorch.org/whl/cu124`
- **AMD ROCm (Linux):** `pip install torch --index-url https://download.pytorch.org/whl/rocm6.4` —
  the device presents as `cuda`, no engine config change needed.
- **AMD ROCm (Windows, preview):** AMD publishes Python 3.12 wheels on
  [repo.radeon.com](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)
  (Radeon RX 7000/9000 + select Ryzen AI only).
- **Apple Silicon (macOS):** plain `pip install torch` (MPS is in the default wheel).

---

## Quickstart — images

```python
from imference_engine import Engine, RuntimeConfig

engine = Engine(runtime=RuntimeConfig(device="auto")).load()
engine.register_model("sdxl", backend="sdxl", weights_path="/models/sdxl.safetensors")

result = engine.generate(
    model="sdxl",
    prompt="a red fox in a snowy forest, golden hour, 85mm",
    negative_prompt="lowres, blurry",
    width=1024, height=1024,
    num_steps=28, guidance_scale=6.5,
    scheduler="EulerAncestralDiscreteScheduler",
    batch=4, seed=42,
)

for img in result.images:      # list[PIL.Image | None]  (None where that seed errored)
    ...                        # result.seeds -> list[int];  result.errors -> list[GenerationError]
```

A **transformer-only** checkpoint (FLUX, Chroma, Z-Image, Qwen-Image) also needs
a `base_model` — the diffusers-format repo that supplies the shared text
encoder(s), VAE and scheduler the single-file omits:

```python
engine.register_model(
    "flux-dev", backend="flux",
    weights_path="/models/flux1-dev.safetensors",
    base_model="black-forest-labs/FLUX.1-dev",   # CLIP-L + T5-XXL + VAE + scheduler
)
result = engine.generate(model="flux-dev", prompt="an astronaut riding a horse")
```

**img2img** is the same call with a `source_image` + `strength`:

```python
result = engine.generate(model="sdxl", prompt="...", source_image=pil_img, strength=0.6)
```

## Quickstart — Wan video

```python
from imference_engine.wan import WanEngine, WanRuntimeConfig

engine = WanEngine(runtime=WanRuntimeConfig(device="auto", memory_profile="auto")).load()
res = engine.generate_video(variant="wan22-t2v-lightning", prompt="a red fox trotting through snow")

res.frames        # list[PIL.Image]  — caller encodes (export_to_video) + uploads
res.fps, res.num_frames, res.seeds
```

Wan is a **separate engine** (`WanEngine`) — a GGUF-MoE video stack whose
residency model differs fundamentally from images (the GPU never holds more than
the active A14B expert). Built-in variants: `wan22-t2v-lightning`,
`wan22-i2v-lightning`, `smoothmix-i2v`, `dasiwa-i2v`. New architectures
plug in as a `VideoBackend` with a new `arch` — no engine fork.

## Quickstart — MiniMax-H3 video + audio

> Requires diffusers ≥ 0.40.0 ([PR #14355](https://github.com/huggingface/diffusers/pull/14355)
> shipped in 0.40.0 — now the repo-wide pin, so H3 shares the venv with every
> other backend). Validated e2e on the PR head that became 0.40.0. See
> [`imference_engine/minimax_h3/README.md`](imference_engine/minimax_h3/README.md).

```python
from imference_engine.minimax_h3 import MiniMaxH3Engine, MiniMaxH3RuntimeConfig

engine = MiniMaxH3Engine(runtime=MiniMaxH3RuntimeConfig(device="auto")).load()
res = engine.generate_video(prompt="a red fox trotting through snow", width=960, height=544)

res.frames                    # list[PIL.Image] — fixed 24 fps
res.audio, res.sample_rate    # (2, n) float32 stereo waveform + Hz — generated JOINTLY
```

MiniMax-H3 (33B DiT + Qwen3-VL-32B conditioner, Modular Diffusers) generates the
soundtrack **with** the video in one denoising loop. One variant serves t2v and
i2v (`image=` / `last_image=` keyframes). Guidance-distilled: no negative prompt,
no guidance scale. int8 (torchao) by default — 24 GB VRAM / 64 GB RAM class with
block-streamed offload; `validation/stage_h3_int8.py` stages a pre-quantized
mirror so cold loads skip the ~90 GB bf16 download.

---

## Sidecar or standalone

`imference-engine` is the **inference core of Imference Desktop** (the desktop
app runs it as an in-process sidecar) — but it is a standalone library first. It
does **no network I/O on the result side** and knows nothing about Runqy,
FastAPI, webhooks or Electron: `generate()` returns PIL frames + metadata and the
caller decides what to do with them. That clean cut is why the *same* engine
serves three very different hosts unchanged:

| Host | How it drives the engine |
|---|---|
| **Imference Desktop** (sidecar) | Builds `RuntimeConfig(...)` from app settings, single resident model, `enable_offload` on small GPUs. |
| **GPU worker fleet** | `RuntimeConfig.from_env()` reads the `IMAGE_*` / `WAN_*` env contract, then layers cgroup-aware `max_*_models` on top. |
| **Your code** | `pip install imference-engine`, construct an `Engine`, call `generate()`. |

Every knob is settable **identically by an environment variable OR a constructor
param**, each with a default — a launcher only populates the contract, so hosts
are interchangeable and no engine logic leaks into them.

---

## The request payload

### `Engine.generate(...) -> MediaResult`

| param | type | default¹ | notes |
|---|---|---|---|
| `model` | `str` | — (required) | A registered model name. |
| `prompt` | `str` | — (required) | Positive prompt. |
| `negative_prompt` | `str?` | `None` | Honored by SDXL/SD1.5/Z-Image/Chroma/Qwen-Image/Anima; **ignored by FLUX** (guidance-distilled) and **by Krea 2 whenever `guidance_scale <= 0`** (the Turbo norm). |
| `width`, `height` | `int?` | `1024` | 512 for SD 1.5. |
| `num_steps` | `int?` | `28` | |
| `guidance_scale` | `float?` | `6.0` | Per-engine sweet spots differ (see matrix). |
| `clip_skip` | `int?` | `None` | **SDXL / SD 1.5 only**; ignored by the flow-matching DiTs. |
| `scheduler` | `str?` | `None` | Honored by SDXL / SD 1.5 only (see matrix). |
| `batch` | `int` | `1` | N independent images, one seed each. |
| `seed` | `int?` | `None` | `None` → random; batch uses `seed, seed+1, …`. |
| `source_image` | `PIL.Image?` | `None` | Present ⇒ img2img (Anima and Krea 2 excepted — t2i only). |
| `strength` | `float?` | `0.75` | img2img denoise strength. |
| `backend_options` | `dict?` | `{}` | Engine-specific, e.g. `{"shift": 3.0}` (flow-matching DiTs). |
| `loras` | `list[dict]?` | `None` | Reserved — not wired in V1 (logged + ignored). |

¹ Unset (`None`) request fields fall through the **precedence chain** —
`request > model defaults > engine defaults > GLOBAL_DEFAULTS` (finest non-None
wins). `GLOBAL_DEFAULTS = num_steps 28 · guidance 6.0 · 1024×1024 · strength
0.75`. So a per-model catalog default fills what the request omits, and a
checkpoint's recipe lives with the checkpoint.

### `WanEngine.generate_video(...) -> MediaResult`

`variant` (required) · `prompt` (required) · `image` (required for i2v) ·
`negative_prompt` · `width=832` · `height=480` · `num_frames=81` · `num_steps=4`
· `guidance_scale=1.0` · `guidance_scale_2` (low-noise expert) · `fps=16`
(metadata) · `seed`.

### `MiniMaxH3Engine.generate_video(...) -> MediaResult`

`prompt` (required) · `variant="minimax-h3"` · `image` / `last_image` (keyframes
→ i2v; both optional) · `width` / `height` (omit both for the model's native
768-short-edge canvas; multiples of 32) · `num_frames=124` (snaps to `17n+5`,
5–15 s) · `num_steps` (variant default 50) · `seed`. Fixed 24 fps; no negative
prompt / guidance (distilled).

### `MediaResult`

`.images` / `.frames` (both alias `.media`) · `.seeds` · `.errors`
(`list[GenerationError]`) · `.error` (first, or None) · `.ok` (no errors and ≥1
frame). Video calls also carry `.fps` / `.num_frames` / `.width` / `.height` /
`.variant`; MiniMax-H3 additionally fills `.audio` (`(2, n)` float32 stereo
waveform) + `.sample_rate`. `generate_video` never raises — failures come back
as `errors`.

---

## Per-engine behavior

Every backend is the single source of truth for its own recipe. Full env ↔ param
↔ default tables and payload notes are in each backend's README (linked below);
the complete cross-engine reference is in
**[`docs/reference.md`](docs/reference.md)**.

| backend | family | `negative_prompt` | guidance default | `scheduler` name | `backend_options` | img2img | `clip_skip` | dtype |
|---|---|---|---|---|---|---|---|---|
| `sdxl` | UNet (CLIP×2) | honored | 6.0 (global) | **honored** (Euler-A / DPM++) | — | ✅ | ✅ | fp16 |
| `sd15` | UNet (CLIP) | honored | 7.0 | **honored** | — | ✅ | ✅ | fp16 |
| `zimage` | flow DiT | honored | 1.0 | ignored (flow) | `shift` | ✅ | — | bf16 |
| `flux` | flow DiT (12B) | **ignored** (distilled) | 3.5 | ignored (flow) | `shift` | ✅ | — | bf16 |
| `chroma` | flow DiT (8.9B) | honored (real CFG) | 2.0 | ignored (flow) | `shift` | ✅ | — | bf16 |
| `qwenimage` | MMDiT (20B) | honored (→ `true_cfg_scale`) | 4.0 | ignored (flow) | `shift` | ✅ | — | bf16 |
| `anima` | modular DiT | honored (if set) | **ignored** (Guider block) | ignored (block) | — | ❌ (t2i) | — | bf16 |
| `krea2` | flow DiT (12.9B) | honored only if `guidance > 0` | **0.0** (Turbo, guidance off) | ignored (flow) | — | ❌ (t2i) | — | bf16 (fp8-resident for fp8 files) |

Notes worth knowing: **FLUX** ignores negatives (guidance-distilled) and defaults
`guidance 3.5`; **Chroma** is de-distilled → true CFG, `guidance 2.0` (higher
oversaturates); **Qwen-Image** maps `guidance_scale → true_cfg_scale`, negative
default is a single space `" "`, and it wants ~40–50 steps (set per-model);
**Anima** is a Modular Diffusers pipeline (repo-id `weights_path`, no img2img,
`guidance_scale` ignored — guidance is a Guider block);
**Krea 2** is Turbo-first (8 steps, `guidance 0.0` in the Krea convention —
velocity `cond + g·(cond−uncond)`, so conventional CFG ≈ 1+g; negatives only
act when g > 0), loads civitai/ComfyUI single-files **as-is** (native keys +
scaled-fp8 dequantized in memory; fp8 files stay ~13 GB fp8-resident,
`KREA2_FP8_STORAGE` overrides), requires `base_model=` (`krea/Krea-2-Turbo`,
gated), t2i only for now;
the flow-matching DiTs ignore the `scheduler` name; Z-Image/FLUX/Chroma/
Qwen-Image take an explicit `shift` via `backend_options` (Krea 2 does not —
its Turbo checkpoints pin a fixed mu internally).

Deep dives: [SDXL + SD 1.5 + shared config](imference_engine/pipelines/README.md)
· [Z-Image](imference_engine/zimage/README.md) ·
[FLUX](imference_engine/flux/README.md) ·
[Chroma](imference_engine/chroma/README.md) ·
[Qwen-Image](imference_engine/qwenimage/README.md) ·
[Anima](imference_engine/anima/README.md) ·
[Krea 2](imference_engine/krea2/README.md) ·
[Wan video](imference_engine/wan/README.md) ·
[MiniMax-H3 video+audio](imference_engine/minimax_h3/README.md).

---

## Configuration (env ↔ param)

`RuntimeConfig.from_env()` / `WanRuntimeConfig.from_env()` build a config from the
documented env contract (safe with no environment). The essentials — full tables
in [`docs/reference.md`](docs/reference.md):

**Image (`IMAGE_*`)** — `IMAGE_DEVICE` · `IMAGE_MODEL_CACHE` · `IMAGE_MODEL_CDN` ·
`MAX_GPU_MODELS` · `MAX_CPU_MODELS` · `IMAGE_USE_TINY_VAE` (SDXL/SD1.5 only) ·
`IMAGE_ENABLE_CPU_OFFLOAD` · `IMAGE_OFFLOAD_MODE` (`model` default | `group` =
block-streamed compute module, ~5-6 GB peak VRAM even for the 12-20B DiTs —
runs FLUX/Qwen-Image/Krea 2 on 8 GB cards, PCIe-bound and RAM-hungry; Krea 2
measured: 5.9 GB peak, ~30% slower than fp8-resident).

**Video (`WAN_*`)** — `WAN_DEVICE` · `WAN_PROFILE` (GGUF quant / `auto`) ·
`WAN_MAX_RESIDENT` · `WAN_MODEL_CACHE` · `WAN_MODEL_CDN` · `WAN_TEXT_ENCODER_QUANT`
· `WAN_VAE_TILING` · `WAN_ENABLE_OFFLOAD`.

**Video (`H3_*`, MiniMax-H3)** — `H3_DEVICE` · `H3_PROFILE` (`int8`/`bf16`/`auto`)
· `H3_OFFLOAD_MODE` (`block`/`leaf`/`none`/`auto`) · `H3_MAX_RESIDENT` ·
`H3_MODEL_CACHE` · `H3_MODEL_CDN` · `H3_VAE_TILING` · `H3_ATTENTION_BACKEND`.

**Global** — `HF_HUB_OFFLINE=1` (guardrail: any stray HF call fails loudly) ·
`HF_HOME` (flat-tree fallback root) · `IMAGE_CDN_THREADS` / `WAN_CDN_THREADS` ·
`BATCH_VRAM_RESERVE_MB` · `MAX_BATCH_SIZE`.

> `auto` resolution is **host-side**: `from_env()` leaves `MAX_*_MODELS=auto` /
> `WAN_PROFILE=auto` at engine defaults, and the worker overrides them with
> cgroup-aware hardware detection. The engine never probes hardware itself.

### Offline & the CDN mirror

Set `IMAGE_MODEL_CDN` / `WAN_MODEL_CDN` to an R2/S3 mirror of the `<repo>/<file>`
layout and shared base components resolve from the CDN instead of HuggingFace —
with `HF_HUB_OFFLINE=1`, zero HF contact. Stage the bases onto R2 with
[`validation/stage_r2.py`](validation/stage_r2.py) (pulls the exact base
patterns, writes the manifest the reader expects, uploads idempotently).

### Multi-model residency (worker path)

```python
engine = Engine(runtime=RuntimeConfig(
    device="auto", max_gpu_models=2, max_cpu_models=8,
)).load()
engine.set_lifecycle_hooks(on_model_loaded=disk_cache.protect,
                           on_model_evicted=disk_cache.unprotect)
```

`max_gpu_models` pipes stay resident in VRAM; `max_cpu_models` demoted-but-warm
pipes stay in CPU RAM for ~0.5 s re-promotion. A **catalog** (`models.yml`) can
register every model with its per-model defaults —
`Engine(catalog_path="models.yml")` or `engine.load_catalog(...)`; see
[`docs/catalog-design.md`](docs/catalog-design.md).

---

## Architecture

```
imference_engine/
  engine.py            # Engine — public image entry point (register_model, generate)
  __init__.py          # Engine, RuntimeConfig, MediaResult, GenerationError
  core/                # shared trunk (unified image+video core)
    result.py          #   MediaResult / GenerationError
    config.py          #   BaseRuntimeConfig (device, cache, cdn, enable_offload)
    engine_base.py     #   BaseEngine (device resolve, cuda tune, seed)
    backend.py         #   Backend / PipelineBackend ABCs
  pipelines/           # base.py (PipelineBackend) + sdxl.py + sd15.py
  zimage/ flux/ chroma/ qwenimage/ anima/ krea2/   # one package per image backend
  video/               # video architecture layer
    backend.py         #   VideoBackend ABC + VideoBuildContext
    residency.py       #   generic ResidencyManager
    backends/wan.py    #   WanBackend (arch="wan")
    backends/minimax_h3.py  # MiniMaxH3Backend (arch="minimax_h3")
  wan/                 # WanEngine, WanRuntimeConfig, presets (variants), loader
  minimax_h3/          # MiniMaxH3Engine, config, presets, loader (video+audio)
  catalog/             # models.yml loader + GenerationDefaults precedence
  managers/            # batch.py (BatchSizer) + model.py (ModelManager LRU)
  prompting/           # weighted.py (sd-embed + BREAK)
  runtime/             # device.py, offline.py (flat tree / CDN), download.py
```

Design docs: [unified engine core](docs/unified-engine-core.md) ·
[catalog loader](docs/catalog-design.md).

## Validation

[`validation/`](validation/) holds a GPU harness that loads each engine's base
model, renders, and reports pass/fail — `python validation/validate.py`
(`validate_wan.py` for video). The full v0.4.0 stack — seven image backends,
Wan t2v/i2v, MiniMax-H3 — passed end-to-end on 2026-08-26 (diffusers 0.40.0 ·
transformers 5.4.0); the Krea 2 backend passed on 2026-08-27 (official
scaled-fp8 + a civitai plain-fp8 finetune, RTX PRO 4000 24 GB). See
[`validation/README.md`](validation/README.md).

## Status & scope

Wired and validated (v0.4.0 stack, 2026-08-26): seven image backends, Wan 2.2
video, MiniMax-H3 video+audio (released diffusers 0.40.0, int8 R2 mirror),
img2img, multi-tier residency, weighted prompts, the catalog loader +
precedence chain, and offline/CDN resolution — including the offline converter
for ComfyUI/civitai H3 int8-ConvRot single-files
(`validation/stage_h3_from_comfy.py` — ~67 GB downloaded instead of ~124 GB).
The **Krea 2 Turbo backend** (civitai/ComfyUI single-file + scaled-fp8 load
path) is validated as of 2026-08-27 — official scaled-fp8 and a civitai
plain-fp8 finetune both render clean on a 24 GB card.
**Not yet wired:** `LoRAManager` (image LoRA stacking — `loras=` is accepted
but ignored), Qwen-Image-Edit, quantized image builds, and MiniMax-H3 `ref2va`
/ int4-nvfp4 ConvRot loading (needs ComfyUI kernels). MPS (Apple Silicon) is
untested.
