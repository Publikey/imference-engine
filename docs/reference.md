# Reference — variables & payload

The complete, cross-engine contract: every environment variable, the full
request payload, the defaults precedence chain, and each backend's exact
inference behavior. This is the authoritative companion to the per-backend
READMEs (which add prose + examples). Values here are read straight from the
backend source.

- [Public API surface](#public-api-surface)
- [Environment variables](#environment-variables) — [image](#image-runtimeconfig--imagefrom_env) · [video](#video-wanruntimeconfig--wanfrom_env) · [global](#global--offline)
- [Request payload](#request-payload) — [`generate`](#enginegenerate) · [`generate_video`](#wanenginegenerate_video) · [`MediaResult`](#mediaresult)
- [Defaults precedence chain](#defaults-precedence-chain)
- [Per-engine inference contract](#per-engine-inference-contract)
- [Wan variants](#wan-variants--quant-profiles)

---

## Public API surface

```python
from imference_engine import Engine, RuntimeConfig, MediaResult, GenerationError
from imference_engine.wan import WanEngine, WanRuntimeConfig, WanVariant, WanLora, MemoryProfile
```

`Engine` — `.load()` · `.register_model(name, *, backend, weights_path, base_model=None, defaults=None)`
· `.load_catalog(path=None)` · `.warm(specs)` · `.set_lifecycle_hooks(on_model_loaded=, on_model_evicted=)`
· `.generate(**payload) -> MediaResult`.

`WanEngine` — `.load()` · `.register_variant(v)` · `.list_variants()` ·
`.load_catalog(path)` · `.warm()` · `.resident` · `.generate_video(**payload) -> MediaResult`.

---

## Environment variables

Every knob is settable **identically by an env var OR a constructor param**.
`from_env()` is safe to call with no environment set.

### Image (`RuntimeConfig` / `.from_env()`)

| Env var | param | type | default | meaning |
|---|---|---|---|---|
| `IMAGE_DEVICE` | `device` | str | `auto` | `auto` \| `cuda` \| `cuda:N` \| `mps` \| `cpu`. |
| `IMAGE_MODEL_CACHE` | `model_cache_dir` | str/None | `None` → `$HF_HOME/image` or `~/.cache/image` | Root of the flat, symlink-free offline model tree (`<root>/<repo>/<file>`). |
| `IMAGE_MODEL_CDN` | `model_cdn` | str/None | `None` | Base URL of an R2/CDN mirror of the same `<repo>/<file>` layout. Set ⇒ base components fetch from the CDN, not HuggingFace. |
| `MAX_GPU_MODELS` | `max_gpu_models` | int/None (`auto`→None) | `None` (=1) | Pipes concurrently resident in VRAM. |
| `MAX_CPU_MODELS` | `max_cpu_models` | int/None (`auto`→None) | `None` (=0) | Demoted-but-warm pipes kept in CPU RAM for fast GPU re-promotion. Forced to 0 when `enable_offload`. |
| `IMAGE_USE_TINY_VAE` | `use_tiny_vae` | bool | `false` | SDXL → TAESDxl, SD 1.5 → TAESD (~5 MB, fast decode, slight quality loss). **Ignored by all other backends.** |
| `IMAGE_ENABLE_CPU_OFFLOAD` | `enable_offload` | bool | `false` | `enable_model_cpu_offload()` — peak VRAM ≈ largest submodel, ~10–30 % slower. |

`lora_cache_dir` is a constructor field only (no env var); LoRA is not wired in V1.

### Video (`WanRuntimeConfig` / `.from_env()`)

| Env var | param | type | default | meaning |
|---|---|---|---|---|
| `WAN_DEVICE` | `device` | str | `auto` | `auto` \| `cuda` \| `cuda:N` \| `cpu` (mps untested). |
| `WAN_PROFILE` | `memory_profile` | str/enum | `auto` (from_env) / `gguf_q8` (dataclass) | GGUF quant: `auto` \| `gguf_q8` \| `gguf_q6` \| `gguf_q5` \| `gguf_q4` \| `bf16`. `auto` resolved at `load()` from **VRAM and RAM** (picks the lighter). Only GGUF profiles are wired — `bf16` raises `NotImplementedError`. |
| `WAN_MAX_RESIDENT` | `max_resident_variants` | int/None (`auto`→None, treated 1) | `1` | Built pipelines kept warm in CPU RAM (LRU). ~31 GB RAM per Q8 variant + ~30 GB working set. |
| `WAN_MODEL_CACHE` | `model_cache_dir` | str/None | `None` → `$HF_HOME/wan` | Flat model-tree root; GGUF experts are ~15 GB each — use a big volume. |
| `WAN_MODEL_CDN` | `model_cdn` | str/None | `None` | R2/CDN mirror of `<repo>/<filename>`; GGUF experts + LoRAs fetch on demand. |
| `WAN_TEXT_ENCODER_QUANT` | `text_encoder_quant` | str | `int8` | Shared UMT5 weight-only quant: `int8` (torchao, ~4–6 GB saved, graceful bf16 fallback) \| `none`. |
| `WAN_VAE_TILING` | `vae_tiling` | bool | `true` | VAE tiling + slicing — cuts decode VRAM for long/large videos. |
| `WAN_ENABLE_OFFLOAD` | `enable_offload` | bool | `true` | `enable_model_cpu_offload()` per pipe (VRAM ≈ one expert, ~17 GB). |

Quant mapping: `gguf_q8→Q8_0`, `gguf_q6→Q6_K`, `gguf_q5→Q5_K_M`, `gguf_q4→Q4_K_M`,
`bf16→None`. Auto thresholds — VRAM: ≥20→Q8, ≥14→Q6, else Q4; RAM: ≥64→Q8,
≥56→Q6, else Q4 (RAM sizing is cgroup-aware); the engine picks the lighter.

### Global / offline

| Env var | type | default | meaning |
|---|---|---|---|
| `HF_HUB_OFFLINE` | `1`/`true` | unset | Strict offline: trust the sentinel-bearing local tree, no HF download; a stray HF call fails loudly. |
| `HF_HOME` | path | unset → `~/.cache/<ns>` | Fallback flat-tree root when `*_MODEL_CACHE` is unset (`$HF_HOME/image`, `$HF_HOME/wan`). |
| `IMAGE_CDN_THREADS` | int | falls to `WAN_CDN_THREADS` then `8` | Parallel HTTP streams for image-side CDN pulls. |
| `WAN_CDN_THREADS` | int | `8` | Parallel HTTP streams for CDN pulls. |
| `BATCH_VRAM_RESERVE_MB` | int | `512` | VRAM headroom the batch sizer keeps free. |
| `MAX_BATCH_SIZE` | int | `8` | Hard ceiling on images per forward. |

`HF_TOKEN` is not read by the engine — HF auth is left to `huggingface_hub`'s own
defaults (gated repos like `FLUX.1-dev` need `hf auth login`, or mirror the base
to your CDN). `env_bool` truthy set: `{1, true, yes, on}`; `env_int_or_none` maps
unset/empty/`auto`/non-integer → `None`.

---

## Request payload

### `Engine.generate`

Keyword-only. Returns `MediaResult(kind="image")`.

| param | type | default | notes |
|---|---|---|---|
| `model` | `str` | required | Registered model name (`register_model` / catalog). |
| `prompt` | `str` | required | Positive prompt. Weighted syntax + `BREAK` when sd-embed is installed. |
| `negative_prompt` | `str \| None` | `None` | See per-engine table — honored by all except FLUX. |
| `width` | `int \| None` | `1024` | 512 native for SD 1.5. |
| `height` | `int \| None` | `1024` | |
| `num_steps` | `int \| None` | `28` | |
| `guidance_scale` | `float \| None` | `6.0` | Per-engine defaults differ (table below). |
| `clip_skip` | `int \| None` | `None` | Applied only by SDXL / SD 1.5 (and only when `> 0`). |
| `scheduler` | `str \| None` | `None` | Applied only by SDXL / SD 1.5. |
| `batch` | `int` | `1` | N images; seeds `seed, seed+1, …`. |
| `seed` | `int \| None` | `None` | `None` → random per image. |
| `source_image` | `PIL.Image \| None` | `None` | Present ⇒ img2img (all except Anima). |
| `strength` | `float \| None` | `0.75` | img2img denoise strength. |
| `backend_options` | `dict \| None` | `{}` | Engine-specific; merged key-wise through the precedence chain. |
| `loras` | `list[dict] \| None` | `None` | **Not wired in V1** — logged and ignored. |

### `WanEngine.generate_video`

Keyword-only. Returns `MediaResult(kind="video")`. Never raises — a generation
failure comes back as `errors=[GenerationError(...)]` with empty `media`.

| param | type | default | notes |
|---|---|---|---|
| `variant` | `str` | required | Registered variant; `KeyError` if unknown. |
| `prompt` | `str` | required | |
| `image` | `PIL.Image \| None` | `None` | **Required for i2v** variants (`ValueError` otherwise); ignored for t2v. |
| `negative_prompt` | `str \| None` | `None` | |
| `width` | `int` | `832` | |
| `height` | `int` | `480` | 832×480 ≈ 480p. |
| `num_frames` | `int` | `81` | |
| `num_steps` | `int` | `4` | 4 = Lightning 4-step default. |
| `guidance_scale` | `float` | `1.0` | CFG for the high-noise expert. |
| `guidance_scale_2` | `float \| None` | `None` | CFG for the low-noise expert (`transformer_2`). |
| `fps` | `int` | `16` | **Metadata only** — not passed to the pipe. |
| `seed` | `int \| None` | `None` | `None` → engine picks; seed generator is always CPU. |

### `MediaResult`

| member | type | meaning |
|---|---|---|
| `kind` | `str` | `"image"` \| `"video"`. |
| `media` | `list[PIL.Image \| None]` | Batch of images, or the clip's frames. `None` at a batch index that errored. |
| `images` / `frames` | property | Both alias `media` (read naturally per modality). |
| `seeds` | `list[int]` | N seeds (image) or `[seed]` (video). |
| `errors` | `list[GenerationError]` | `error: str`, `seed?`, `batch_index?`. |
| `error` | property | First error or `None` (handy for single-clip video). |
| `ok` | property | No errors **and** ≥1 non-None frame. |
| `fps` / `num_frames` / `width` / `height` / `variant` | video-only | `None` for images. |

---

## Defaults precedence chain

Each unset (`None`) sampling field is resolved finest-wins:

```
request  >  model defaults  >  engine defaults  >  GLOBAL_DEFAULTS
```

- **request** — what you pass to `generate()`. A `None` field does *not* shadow a
  lower layer (that's why the signature defaults are `None`, not concrete).
- **model defaults** — `register_model(..., defaults=GenerationDefaults(...))` or
  a catalog row's `defaults:` (a checkpoint's recipe, e.g. `{num_steps: 4}` for a
  Lightning FLUX-schnell).
- **engine defaults** — `backend.engine_defaults()` (the per-backend column below).
- **GLOBAL_DEFAULTS** — `num_steps 28 · guidance_scale 6.0 · width 1024 ·
  height 1024 · strength 0.75`. `scheduler` / `clip_skip` / `negative_prompt`
  stay `None` here.

`backend_options` merges **key-wise** across layers (a request `shift` overrides a
model `shift` without dropping other keys).

---

## Per-engine inference contract

Extracted from each backend. "engine default" = what `engine_defaults()` sets;
blank fields fall through to `GLOBAL_DEFAULTS`.

### Image backends

| backend | engine defaults | negative_prompt | scheduler | `backend_options` | img2img | clip_skip | max_seq_len | dtype |
|---|---|---|---|---|---|---|---|---|
| **sdxl** | `scheduler=EulerAncestralDiscreteScheduler` | honored (weighted / BREAK) | **name honored**: `DPMSolverMultistepScheduler` (Karras) / `EulerAncestralDiscreteScheduler`; unknown → Euler-A + Karras. Cached per config. | — | ✅ `StableDiffusionXLImg2ImgPipeline` | ✅ (`>0`) | — (77-tok CLIP, weighted-chunked) | fp16 |
| **sd15** | `width=512, height=512, guidance_scale=7.0, scheduler=EulerAncestralDiscreteScheduler` | honored (weighted, single CLIP-L) | **name honored** (same logic as SDXL) | — | ✅ `StableDiffusionImg2ImgPipeline` | ✅ (`>0`) | — | fp16 |
| **zimage** | `guidance_scale=1.0` | honored (raw; default `""`) | ignored; fixed `FlowMatchEulerDiscreteScheduler`. `shift` rebuilds it. | `shift` (≈3.0 480p / 5.0 720p) | ✅ `ZImageImg2ImgPipeline` | ignored | — (no 77-tok limit) | bf16 |
| **flux** | `guidance_scale=3.5, num_steps=28` | **ignored** (guidance-distilled; logs if supplied) | ignored; flow-match w/ dynamic shifting. `shift` → fixed shift. | `shift` | ✅ `FluxImg2ImgPipeline` | ignored | **512** | bf16 |
| **chroma** | `guidance_scale=2.0, num_steps=28` | **honored** (de-distilled, real CFG; default `""`) | ignored; flow-match. `shift` → fixed. | `shift` | ✅ `ChromaImg2ImgPipeline` (single T5, no CLIP) | ignored | **512** | bf16 |
| **qwenimage** | `guidance_scale=4.0` | **honored**, mapped to `true_cfg_scale`; default `" "` (single space) | ignored; flow-match. `shift` → fixed. | `shift` | ✅ `QwenImageImg2ImgPipeline` | ignored | — | bf16 |
| **anima** | *(none — all global)* | honored **only if truthy** (no default injected) | ignored; block-defined in the modular pipeline (no-op) | — | ❌ raises `NotImplementedError` (t2i only) | ignored | — | bf16 |

> **Anima ignores `guidance_scale`.** Its modular pipeline configures guidance via
> a separate Guider block, not a `guidance_scale` __call__ kwarg — the backend
> does not forward it, so a request/engine `guidance_scale` has **no effect** on
> Anima (unlike every other backend).

Behavioral highlights:

- **Scheduler name** is honored **only** by `sdxl` / `sd15`. The four
  flow-matching DiTs (`zimage`, `flux`, `chroma`, `qwenimage`) ignore the name and
  react only to `backend_options["shift"]`; `anima`'s scheduler is block-defined
  and ignored entirely.
- **`negative_prompt`** — ignored by FLUX (guidance-distilled); honored elsewhere.
  Empty/None normalizes to `""` for zimage/chroma, but to `" "` (a single space)
  for qwenimage, matching upstream; anima injects no default and only sends a
  negative when one is truthy.
- **`clip_skip`** — SDXL / SD 1.5 only (and only when `> 0`).
- **`guidance_scale` semantics** — SDXL/SD1.5 classic CFG; FLUX a distilled
  embedded signal (not CFG); Chroma/Z-Image true CFG; Qwen-Image mapped to
  `true_cfg_scale` (the distilled `guidance_scale` pipe arg stays at its default).
- **`base_model`** — required by the transformer-only backends (`zimage`, `flux`,
  `chroma`, `qwenimage`) to supply shared text encoder(s) + VAE + scheduler;
  unused by `sdxl` / `sd15` (full checkpoints) and `anima` (self-contained modular
  repo — its `weights_path` **is** the repo id/dir).
- **Offline / CDN** — all image backends (including `anima`, whose repo-id
  `weights_path` is resolved through `local_repo_dir`) load shared/base components
  from the flat tree or `IMAGE_MODEL_CDN`.

---

## Wan variants & quant profiles

### Built-in variants

| name | mode | Lightning | gguf_repo |
|---|---|---|---|
| `wan22-t2v-lightning` | t2v | LoRA (`lightx2v/Wan2.2-Lightning`, 4-step) | `QuantStack/Wan2.2-T2V-A14B-GGUF` |
| `wan22-i2v-lightning` | i2v | LoRA, 4-step | `bullerwins/Wan2.2-I2V-A14B-GGUF` † |
| `smoothmix-i2v` | i2v | baked (no LoRA) | `Bedovyy/smoothMixWan22-I2V-GGUF` |
| `dasiwa-i2v` | i2v | baked (no LoRA) | `Bedovyy/dasiwaWAN22I2V14B-GGUF` |

All are `arch="wan"`, `flow_shift=3.0`. The two A14B experts split high-noise →
`transformer`, low-noise → `transformer_2`. LoRAs are applied via `set_adapters`
(never fused). "Steps" is per-request (`num_steps`, default 4), not a variant field.

> **† Why i2v uses bullerwins, not QuantStack.** `QuantStack/Wan2.2-I2V-A14B-GGUF`
> renders **moving-pixel mush** on the diffusers 0.39 / torch 2.12 stack — its
> `patch_embedding` (the i2v image-conditioning input, 36 channels) dequantizes
> wrong. Ruled out by elimination on GPU: not the engine refactor (i2v call is
> byte-identical pre/post), not diffusers 0.39 (`transformer_wan.py`,
> `pipeline_wan_i2v.py`, `gguf/utils.py` are byte-identical to the 0.38 that
> worked), not the Lightning LoRA (`--no-lora` still mush), not settings. The
> **same official Wan2.2-I2V-A14B model from bullerwins** (different quantizer,
> flat lowercase filenames) renders clean on the identical stack. The t2v
> QuantStack GGUF is fine (16-channel patch_embedding). If you serve i2v from a
> CDN mirror, stage the bullerwins repo (the QuantStack i2v mirror is unusable).

### `WanVariant` fields

`name` · `mode` (`t2v`/`i2v`) · `base_repo` (config + shared UMT5/VAE) ·
`gguf_repo` · `arch="wan"` · `lightning_baked=False` · `loras=[]` ·
`flow_shift=3.0` · `gguf_high_name` / `gguf_low_name` (exact filename, priority 1)
· `gguf_high_template` / `gguf_low_template` (`{quant}` template, priority 2,
offline-safe). Filename resolution: exact → template → online auto-discovery.

`WanLora`: `repo` · `high_weight_name` · `low_weight_name` · `high_weight=1.0` ·
`low_weight=1.0`.

### Arch routing

`generate_video` routes on `variant.arch`. The engine keeps a `{arch:
VideoBackend}` registry (only `wan` registered today) and lazily builds one
`ResidencyManager` per arch. A new architecture (e.g. LTX) is a `VideoBackend`
subclass with `engine="ltx"` registered into `_backends`; variants with
`arch="ltx"` route to it automatically — no engine fork. Catalog rows are
validated against `known_archs`, so an unknown-arch row is rejected.
