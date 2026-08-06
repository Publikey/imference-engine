# Changelog

All notable changes to imference-engine. Workers pin a **tagged** version (see
[`RELEASING.md`](RELEASING.md)); this file is the migration guide between tags.
Format loosely follows [Keep a Changelog](https://keepachangelog.com); versioning
is semver (pre-1.0: breaking changes may ride a minor bump — read **Breaking**).

## [Unreleased]

### Added

- **MiniMax-H3 video backend** (`imference_engine.minimax_h3`) — joint
  **video + audio** generation (33B DiT + Qwen3-VL-32B conditioner, Modular
  Diffusers). New `MiniMaxH3Engine` / `MiniMaxH3RuntimeConfig` (`H3_*` env
  contract), `MiniMaxH3Backend` as the second `VideoBackend` arch
  (`minimax_h3`), one builtin variant `minimax-h3` serving **both** t2v and
  i2v (`image`/`last_image` keyframes route the task — one resident pipeline
  covers both). Quant profiles `int8` (torchao weight-only v2, on-the-fly or
  from a pre-quantized mirror staged with the new
  `validation/stage_h3_int8.py`) and `bf16`; offload modes `block` (24–32 GB
  VRAM) / `leaf` (12–16 GB) / `none`. Engine-side mirrors of the model's
  constraints (`17n+5` frames, 5–15 s @ 24 fps, mod-32 canvas) fail fast
  before any weights load. New `[minimax-h3]` extra and
  `validation/validate_h3.py` GPU harness.
  ⚠️ **Requires unreleased diffusers** (PR #14355) — cannot coexist with the
  repo-wide `diffusers==0.39.0` pin, so it runs in a dedicated venv until the
  PR ships in a release. **Validated e2e 2026-08-05** on the PR head
  (t2va render + soundtrack from a ComfyUI-sourced int8 tree; ~14.9 s/step at
  960×544×124f, ~21 GB VRAM under `block` offload, RTX 6000 Ada).
  See `imference_engine/minimax_h3/README.md`.
- **ComfyUI/civitai MiniMax-H3 checkpoint support** (offline conversion) — new
  `imference_engine/minimax_h3/comfy_convert.py` (pure-torch: exact ConvRot
  int8 dequantization via the deterministic block-Hadamard, original-layout →
  diffusers key mapping vendored from the PR's convert script, audio-VAE
  weight-norm resynthesis) and `validation/stage_h3_from_comfy.py`, which
  builds a loader-ready modular tree from the four Comfy-Org/civitai
  single-files (**~67 GB downloaded instead of ~124 GB**; `--profile int8`
  emits a ~35 GB torchao-requantized tree, `bf16` a full-fidelity one). The
  engine itself is unchanged — it keeps consuming modular trees. "Pruned" DiT
  repackages (low-rank `adaln_t_table` architecture) and int4/nvfp4 files are
  detected and refused with an explanation. Covered by GPU-free unit tests
  (`tests/test_h3_comfy_convert.py`).
- **`MediaResult.audio` / `MediaResult.sample_rate`** — new optional fields
  carrying a generated soundtrack (`(channels, n)` float32 numpy waveform +
  Hz). `None` for images and video-only backends (Wan); non-breaking.
- **`VideoBuildContext.offload_mode` / `.attention_backend`** — optional
  arch-specific knobs (defaulted; existing backends unaffected).

### Changed

- **Shared video catalogs across engines.** Video rows are now validated
  against every *known* video arch (`imference_engine.video.KNOWN_VIDEO_ARCHS`)
  and each engine registers only its own — one `models.yml` can carry `wan`
  and `minimax_h3` rows without either engine rejecting the other's (a typo'd
  arch still fails loudly). Previously `WanEngine` raised `CatalogError` on any
  non-wan video row.

### Fixed

- **MiniMax-H3 loader: `transformer_ref` no longer streamed from the Hub.**
  `_load_components` passed every null component to `load_components`,
  including the out-of-scope Ref2VA partition whose spec points at the hub
  repo — a cold load would silently start downloading its ~66 GB. Now
  explicitly excluded.
- **MiniMax-H3 int8 load path follows current PR #14355 semantics.** The PR
  head *rejects* `low_cpu_mem_usage=False` with a quantization config (earlier
  drafts required it); the loader and `stage_h3_int8.py` now leave it at its
  default.
- **Test-suite hygiene on torch-installed boxes.** Two `tests/test_device.py`
  cases popped `torch` out of `sys.modules` without restoring it; torch cannot
  be re-imported in the same process, so on machines where it *is* installed
  every later torch-using test failed (40+ failures). Now uses
  `monkeypatch.delitem`, which restores. Also normalized a Windows
  path-separator assertion in `tests/test_offline_snapshot.py`.

## [0.3.2] — 2026-07-17

Bugfix + hardware release: Anima loads on strict-offline workers again, and the
engine runs on AMD GPUs. No API changes — a drop-in upgrade from 0.3.1.

### Added

- **AMD GPU (ROCm) support.** PyTorch's ROCm build masquerades as CUDA
  (`torch.cuda.*` works, device strings stay `cuda:N`), so the engine now runs
  on AMD GPUs with **zero config change** — install the ROCm torch wheel
  (`--index-url https://download.pytorch.org/whl/rocm6.4` on Linux; Python 3.12
  preview wheels from repo.radeon.com on Windows) and `device="auto"` resolves
  to the GPU. New `runtime.device.is_rocm()` helper and `Device.backend`
  property (`"rocm"` on a HIP build, else same as `kind`) for vendor-aware
  logs/UI — `Device.kind` and `torch_str` are unchanged, so all existing
  `kind == "cuda"` branches keep working. The CPU-fallback warning and install
  docs now give the per-vendor torch index instead of assuming NVIDIA.

### Fixed

- **Anima components failed to load offline.** Loading any Anima model on a
  worker with `HF_HUB_OFFLINE=1` left the scheduler, text conditioner, Qwen3 text
  encoder and VAE unset; generation then failed with `'NoneType' object has no
  attribute 'dtype'` from inside the text-encoder block. `modular_model_index.json`
  records each component's source as the **hub repo id**, not a path, so resolving
  the base repo into the offline tree and calling `ModularPipeline.from_pretrained`
  on the local dir was not enough — the component specs still pointed at
  huggingface.co. They are now repointed at the mirrored tree before loading (only
  where the component's subfolder actually exists there, so a component sourced
  from a different repo is left alone and warned about instead of mis-pointed).
  Affects both load paths: single-file DiT + base repo, and the whole modular repo.
  The DiT itself always loaded — the single-file path was never at fault.
- **Anima load failures are no longer silent.** `load_components()` catches
  per-component failures and only *warns*, so a failed load left the attribute
  `None` and surfaced much later as an `AttributeError` in an unrelated denoise
  block. A component that ends up `None` now raises a `RuntimeError` naming it, at
  load time. Note that the offline fast path in `local_repo_dir` trusts any tree
  carrying `modular_model_index.json` without verifying it, so a partially staged
  mirror surfaces here: complete the mirror rather than disabling offline mode.

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
