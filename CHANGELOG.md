# Changelog

All notable changes to imference-engine. Workers pin a **tagged** version (see
[`RELEASING.md`](RELEASING.md)); this file is the migration guide between tags.
Format loosely follows [Keep a Changelog](https://keepachangelog.com); versioning
is semver (pre-1.0: breaking changes may ride a minor bump — read **Breaking**).

## [0.3.0] — 2026-07-13

Big release: five new image backends, the unified image+video core, the catalog
loader, offline/CDN staging, and a full GPU re-validation on diffusers 0.39
(all seven image backends + Wan t2v/i2v render end-to-end; the R2 mirror is proven
to serve offline). **It contains breaking API changes — update consumers in
lockstep with the tag.**

### ⚠️ Breaking

- **Result types unified → `MediaResult`.** The image-side `GenerationResult` and
  the video-side `WanVideoResult` are **gone** (no compat shim for
  `WanVideoResult`). Both `Engine.generate()` and `WanEngine.generate_video()`
  now return `MediaResult`. Migrate by reading `.images` / `.frames` (both alias
  `.media`), `.seeds`, `.errors` (`list[GenerationError]`), `.ok`, `.error`. A
  worker importing `WanVideoResult` will fail to import.
- **Public core relocated.** `types.py` is gone; import from the package root
  (`from imference_engine import Engine, RuntimeConfig, MediaResult,
  GenerationError`) or `imference_engine.core.*`. Internals moved to `core/`
  (`result`, `config`, `engine_base`, `backend`) and a new `video/` package.
- **Unified offload knob.** `RuntimeConfig` / `WanRuntimeConfig` now subclass a
  shared `BaseRuntimeConfig` with one `enable_offload` field. The **env contract
  is unchanged** (`IMAGE_ENABLE_CPU_OFFLOAD`, `WAN_ENABLE_OFFLOAD`), but code that
  set the old per-modality Python field must use `enable_offload`.
- **diffusers pinned 0.38.0 → 0.39.0** across `[runtime]` and `[wan]` (kept on one
  shared version). Re-validate any custom `from_single_file` / LoRA code.

### Added

- **Image backends: FLUX.1, Chroma, SD 1.5, Qwen-Image, Anima** — seven total
  (with SDXL, Z-Image), each its own sub-package on the shared `PipelineBackend`
  ABC. Anima is a Modular Diffusers pipeline (repo-id `weights_path`, t2i only).
- **Catalog loader** (`models.yml`) with a 3-layer defaults precedence chain
  (`request > model > engine > GLOBAL_DEFAULTS`); `Engine(catalog_path=...)` /
  `load_catalog()`, image + `kind: video` rows.
- **Unified image+video core** — `MediaResult`, `BaseRuntimeConfig`, `BaseEngine`,
  `Backend` / `VideoBackend` ABCs, a generic `ResidencyManager`, and an `arch`
  registry so a new video architecture (e.g. LTX) plugs in as a `VideoBackend`
  with no engine fork.
- **Offline / CDN staging** — `validation/stage_r2.py` mirrors image base
  components (and, via `--wan-gguf`, Wan GGUF experts) onto an R2/S3 bucket; the
  engine loads bases from `IMAGE_MODEL_CDN` / `WAN_MODEL_CDN` with zero HuggingFace
  contact. `Engine.warm()` / `WanEngine.warm()` deploy-time prefetch.
- **GPU validation harness** — `validation/validate.py` (per-image-engine, base
  model → render → report) and `validation/validate_wan.py` (video), plus
  `--rm-weights`, `--flow-shift` / `--guidance` / `--no-lora` / `--gguf-repo`
  sweep flags and build DIAG logging.
- **Docs** — a vendeur + sidecar/standalone README, the complete cross-engine
  [`docs/reference.md`](docs/reference.md) (env vars, payload, per-engine matrix),
  and the [unified-core](docs/unified-engine-core.md) / [catalog](docs/catalog-design.md)
  design docs.

### Changed

- **Wan i2v default GGUF: QuantStack → `bullerwins/Wan2.2-I2V-A14B-GGUF`.** The
  QuantStack i2v GGUF renders moving-pixel mush on the diffusers 0.39 / torch 2.12
  stack (its `patch_embedding` conditioning tensor dequantizes wrong); the same
  official model from bullerwins is clean. Ruled out by elimination on GPU — not
  the engine, not diffusers 0.39 (Wan code byte-identical to 0.38), not the LoRA.
  T2V is unaffected. **Update prod `models.yml` i2v rows to bullerwins and mirror
  that repo to `WAN_MODEL_CDN`.**
- **Chroma** default `guidance_scale` 4.0 → 2.0 (validated: higher oversaturates).
- **Wan i2v** input image is resized preserving aspect ratio within the
  `width*height` area budget (was hard-resized/squashed).

### Fixed

- **Wan prompt ignored** — the UMT5 input embedding is re-tied to `shared` on load
  (transformers 5.1 left `encoder.embed_tokens.weight` zero-init, so the encoder
  emitted garbage and generation ignored the prompt).
- **Offline cache** — a truncated HF snapshot is now healed instead of poisoning
  the cache (`.hf_complete` marker; the exact bug that ate two FLUX runs).
- **SD 1.5** cold load resolves the `feature_extractor` (safety-checker config).
- **Anima** stops forwarding `guidance_scale` (its modular pipeline ignores it —
  silences a per-render warning; output unchanged).
- Validation harness now honors `IMAGE_MODEL_CDN` / `WAN_MODEL_CDN` (was dropped).

### Validated

All seven image backends and Wan 2.2 t2v + i2v render end-to-end on diffusers
0.39 (RTX PRO 5000 Blackwell, torch 2.12). The R2 base mirror is proven to serve
offline (Anima fully from R2; FLUX/Chroma/Qwen-Image bases from R2).

### Known gaps

- Image LoRA stacking (`LoRAManager`) is not wired — `loras=` is accepted and
  ignored.
- Qwen-Image-Edit and quantized image builds are out of scope for now.
- MPS (Apple Silicon) is untested.
