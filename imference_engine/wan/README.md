# Wan video engine configuration

The Wan video sub-package uses its own `WanRuntimeConfig` (video residency works
differently from image — the GPU never holds more than the active MoE expert, so
there is no GPU-LRU tier). Like the image side, it is driven **identically by an
env var OR a constructor param**, each with a default.

- **Worker path:** `wan-video-im-engine/start.sh` exports the env below →
  `main.py` calls `WanRuntimeConfig.from_env()` → overrides
  `max_resident_variants` with a cgroup-aware value.
- **Desktop path:** build `WanRuntimeConfig(...)` directly with params, no env.

## `WanRuntimeConfig` — env ↔ param ↔ default

| Env var | `WanRuntimeConfig` param | Default | Effect |
|---|---|---|---|
| `WAN_DEVICE` | `device` | `auto` | `auto` \| `cuda` \| `cuda:N` \| `cpu` (mps untested). |
| `WAN_PROFILE` | `memory_profile` | `auto` | GGUF quant: `auto` (pick from VRAM/RAM at load) \| `gguf_q8` \| `gguf_q6` \| `gguf_q5` \| `gguf_q4` \| `bf16`. |
| `WAN_MAX_RESIDENT` | `max_resident_variants` | `1` | Built pipelines kept warm in CPU RAM (LRU). ~31 GB RAM per Q8 variant + ~30 GB working set. `auto`/unset → `1` (worker overrides with a cgroup-aware value). |
| `WAN_MODEL_CACHE` | `model_cache_dir` | `None` (HF default / `$HF_HOME`) | Flat model tree root; GGUF experts are ~15 GB each — put it on a big volume. |
| `WAN_MODEL_CDN` | `model_cdn` | `None` | R2/CDN mirror of `<repo>/<filename>`; GGUF experts + LoRAs fetch on demand from the CDN instead of HuggingFace. |
| `WAN_TEXT_ENCODER_QUANT` | `text_encoder_quant` | `int8` | UMT5 weight-only quant: `int8` (torchao, ~4–6 GB RAM saved, graceful bf16 fallback) \| `none`. |
| `WAN_VAE_TILING` | `vae_tiling` | `true` | VAE tiling + slicing — cuts decode VRAM for long/large videos. |
| `WAN_ENABLE_OFFLOAD` | `enable_offload` | `true` | `enable_model_cpu_offload()` on each pipe (VRAM ≈ one expert, ~17 GB). Disable only on a GPU big enough to hold a whole variant. |

> **`auto` resolution is worker-side.** `from_env()` leaves `WAN_PROFILE=auto`
> as the string `"auto"` (the engine resolves the GGUF quant from VRAM/RAM at
> `load()`) and maps `WAN_MAX_RESIDENT=auto`/unset to `1` (the worker's
> cgroup-aware `_max_resident()` overrides it).

## Offline / HuggingFace env (honoured implicitly)

| Env var | Effect |
|---|---|
| `HF_HUB_OFFLINE=1` | Guardrail: stray HuggingFace calls fail loudly. Set it when relying on the flat tree / CDN. |
| `HF_HOME` | Fallback flat-tree root when `WAN_MODEL_CACHE` is unset (engine uses `$HF_HOME/wan`). |
| `WAN_CDN_THREADS` | Parallel HTTP streams for CDN downloads. Default `8`. |

## Warm at deploy (optional)

`WanEngine.warm()` pre-downloads the shared base-components (UMT5 / VAE, ~11.5 GB)
+ the registered variants' base configs **without loading anything** — so a fresh
pod is "ready" with the base on disk and the first request only builds the variant
(GGUF experts + LoRAs stay lazy). Best-effort: a failure logs a warning and falls
back to lazy fetch — `warm()` never raises, so it's safe in a worker's `setup()`.

## Per-request params (`WanEngine.generate_video`)

`variant`, `prompt`, `image` (i2v), `negative_prompt`, `width=832`,
`height=480`, `num_frames=81`, `num_steps=4`, `guidance_scale=1.0`,
`guidance_scale_2`, `fps=16`, `seed`. The i2v input image is resized preserving
its aspect ratio within the `width*height` area budget (aligned to the Wan latent
grid), not squashed to `width×height`.

## Built-in variants & the i2v GGUF caveat

| variant | mode | GGUF repo |
|---|---|---|
| `wan22-t2v-lightning` | t2v | `QuantStack/Wan2.2-T2V-A14B-GGUF` + Seko Lightning LoRA |
| `wan22-i2v-lightning` | i2v | **`bullerwins/Wan2.2-I2V-A14B-GGUF`** + Seko Lightning LoRA |
| `smoothmix-i2v`, `dasiwa-i2v` | i2v | Bedovyy merges, Lightning **baked** (no LoRA) |

> ⚠️ **The i2v default is bullerwins, not QuantStack — on purpose.**
> `QuantStack/Wan2.2-I2V-A14B-GGUF` renders **mush** on the diffusers 0.39 /
> torch 2.12 stack (its `patch_embedding` conditioning tensor dequantizes wrong);
> the **same official model from bullerwins** renders clean. Proven by
> elimination on GPU — not the engine, not diffusers 0.39 (Wan code byte-identical
> to 0.38), not the LoRA. QuantStack **t2v** is unaffected. Full write-up in
> [`../../docs/reference.md`](../../docs/reference.md) → *Wan variants*.
> **CDN note:** serving i2v offline requires mirroring the *bullerwins* repo to
> `WAN_MODEL_CDN` (the QuantStack i2v mirror renders mush).

Override the GGUF provider/templates per catalog row (`gguf_repo`,
`gguf_high_template`, `gguf_low_template`) or, for a quick A/B on the harness,
`validation/validate_wan.py --gguf-repo … --gguf-high-template … --gguf-low-template …`.

## Examples

```python
from imference_engine.wan import WanEngine, WanRuntimeConfig

# Desktop — params only.
engine = WanEngine(runtime=WanRuntimeConfig(device="auto", memory_profile="auto")).load()

# Worker — env contract + cgroup-aware residency override.
cfg = WanRuntimeConfig.from_env()           # reads WAN_* env
cfg.max_resident_variants = _max_resident()  # cgroup-aware
engine = WanEngine(runtime=cfg).load()
```
