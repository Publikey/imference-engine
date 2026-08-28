# Changelog

All notable changes to imference-engine. Workers pin a **tagged** version (see
[`RELEASING.md`](RELEASING.md)); this file is the migration guide between tags.
Format loosely follows [Keep a Changelog](https://keepachangelog.com); versioning
is semver (pre-1.0: breaking changes may ride a minor bump — read **Breaking**).

## [0.4.3] — 2026-08-28

### Fixed

- **Krea 2: single-file checkpoints with mixed float dtypes no longer crash
  mid-inference.** Some finetune tooling saves fp32 biases (or norm scales)
  next to bf16/fp8 weights; `load_state_dict(assign=True)` kept them fp32 and
  the first denoising step died with `RuntimeError: self and mat2 must have
  the same dtype, but got Float and BFloat16` (hit in production with a
  civitai fp8_scaled finetune carrying 7 fp32 biases — same failure family as
  the Illustrious SDXL fp32 biases). `prepare_krea2_state_dict` now unifies
  EVERY floating tensor to the compute dtype after dequantization, matching
  `from_pretrained(dtype=...)`. Side effect: the official file's fp32 norm
  scales are now bf16 too (upstream-canonical; renders shift imperceptibly
  vs. v0.4.2 at the same seed).

## [0.4.2] — 2026-08-28

### Added

- **Krea 2: quantized-checkpoint dequant streams through the GPU.** The
  fp8/int8 conversion multiplies in fp32 on CPU (~25 s of the cold load on a
  datacenter pod); with a CUDA device present, the per-tensor math now runs on
  GPU — ONE tensor at a time (<1 GB peak VRAM, largest layer ≈ 100 MB
  quantized), result immediately back to CPU in bf16, so it cannot OOM even on
  8 GB cards. Any device error falls back to the CPU path for the remaining
  tensors. ``prepare_krea2_state_dict`` gains an optional ``device=``;
  ``comfy_convert.rotate_weight`` now builds its Hadamard on the weight's
  device (no-op for the H3 offline converter, which stays CPU).

- **Krea 2: ComfyUI int8 "ConvRot" checkpoints load as-is.** After the fp8
  wave, the other dominant civitai Krea 2 format (aimed at pre-fp8 cards —
  RTX 3000 and older; several popular finetunes ship *only* int8):
  ``<base>.weight`` int8 block-Hadamard-rotated + per-output-channel
  ``.weight_scale`` + per-layer ``.comfy_quant`` JSON config.
  ``krea2/convert.py`` now detects the triplet (gated on the WEIGHT dtype —
  scaled-fp8 files can carry ``.comfy_quant`` tags too) and dequantizes
  exactly via the vendored comfy-kitchen ConvRot dequant already shared with
  the MiniMax-H3 converter (verified byte-level against Comfy-Org/Krea-2
  ``int8_convrot``: ``{"format": "int8_tensorwise", "convrot": true,
  "convrot_groupsize": 256}``). Previously these files were **silently
  corrupted** — the ``.weight_scale`` keys matched the scaled-fp8 path, which
  multiplied without un-rotating. The fp8-resident auto now covers all
  quantized sources (fp8 or int8 — ~13 GB resident; ``KREA2_FP8_STORAGE``
  still overrides), int8 tensors without a recognized config are refused
  loudly (int4/nvfp4/mxfp8 too), and ``prepare_krea2_state_dict``'s flag is
  now ``source_was_quantized``.

## [0.4.1] — 2026-08-28

### Added

- **User LoRAs on SDXL** (`generate(loras=[{source, weight, adapter_name}])`
  — `path`/`url` accepted as aliases; the pre-LoRA `loras=` warn-and-ignore
  survives on every other backend via the new
  `PipelineBackend.supports_loras` gate). New `managers/lora.py` LoRAManager,
  ported from the legacy sdxl-multimodel worker with the Wan loader's
  lessons: adapters applied with `set_adapters` and **never fused**,
  deactivated in a `finally` after every request (resident ModelManager pipes
  stay clean), loaded in the offline-safe `(dir, weight_name)` form,
  per-pipe adapter cache (LRU-evicted beyond `MAX_CACHED_LORAS`, default 5)
  that dies with the pipe on eviction. URL sources download through the
  engine's parallel downloader into an LRU-pruned cache dir
  (`IMAGE_LORA_CACHE` / `RuntimeConfig.lora_cache_dir`). A failed LoRA load
  fails the request as an error result — it never silently renders without
  the LoRA. `validation/validate.py` passes an optional per-entry `loras:`
  stack through. 13 GPU-free tests.

### Fixed

- **Prompts with lone UTF-16 surrogates no longer crash the tokenizer.**
  Broken copy-pastes (an emoji/astral codepoint split in half) survive
  JS strings → JSON → Python and then kill the Rust fast tokenizer at the
  PyO3 boundary with the opaque `TypeError: TextEncodeInput must be
  Union[...]` (hit in production via the desktop sidecar with a pasted
  Chinese prompt). All three engines now strip lone surrogates at their
  `generate()` boundary (`core/text.py`, warning logged) — valid text is
  passed through untouched, real emoji included.

### Added

- **Group offloading for the image engine** (`offload_mode="group"`, env
  `IMAGE_OFFLOAD_MODE`; default `"model"` = unchanged behavior). With
  `enable_offload=True` and mode `"group"`, the ModelManager wires diffusers
  group offloading instead of `enable_model_cpu_offload`: the backend's
  compute module (unet/transformer) streams **block-by-block** with a
  CUDA-stream prefetch, text encoders go leaf-level, the VAE stays resident.
  Peak VRAM drops to ~5-6 GB even for the 12-20B DiTs, so FLUX / Qwen-Image /
  Krea 2 become runnable on 8 GB cards — at the cost of host RAM holding the
  full pipe and PCIe-bound step times. **GPU-validated 2026-08-27** (RTX PRO
  4000 Blackwell): Krea 2 12.9B renders in 108.8 s at a **5.9 GB peak** (vs
  82.9 s fp8-resident), pixel-identical output. CUDA-only; falls back to model
  offload elsewhere or on any wiring failure. Host buffers are **UNPINNED by
  default** (`low_cpu_mem_usage=True`): pinning the multi-GB pipe kills the
  process under a container RLIMIT_MEMLOCK cap (8 MB on vast.ai/Docker —
  SIGKILL, no traceback) and, on a low-RAM Windows host, fails as a
  "CUDA error: out of memory" whose async report poisons the CUDA context
  (observed on the desktop sidecar). Unpinned measured ~free on a datacenter
  pod; `IMAGE_GROUP_PINNED=1` opts back in (refused under a small memlock).
  **Krea 2's fp8-resident composes with group offloading** (GPU-validated:
  83.6 s / 4.7 GB peak VRAM for the 12.9B — faster than the bf16 group run,
  half the bytes per streamed block; render identical) and halves the HOST RAM
  footprint (~13 GB transformer instead of ~26) — the difference between
  fitting and swapping on a 32 GB machine. Works with every image backend via
  `get_compute_module` (Anima's modular pipe falls back to model offload when
  it exposes no compute module).

- **Krea 2 (Turbo) image backend** (`imference_engine.krea2`, engine id
  `krea2`) — Krea AI's 12.9B single-stream flow-matching DiT (Qwen3-VL-4B
  text encoder tapped at 12 layers, Qwen-Image VAE), riding the existing
  `Krea2Pipeline` from the pinned diffusers 0.40.0. Built for the
  **civitai/ComfyUI Turbo finetune ecosystem**: transformer-only single-files
  in the NATIVE key layout, predominantly ComfyUI "scaled fp8". diffusers has
  no `from_single_file` for Krea 2 (issue #14122, PRs #14126/#14264 unmerged)
  and no scaled-fp8 path, so `krea2/convert.py` normalizes the file IN MEMORY
  at load — prefix strip → exact per-tensor fp8 dequant (`w = fp8 ×
  weight_scale`) → native→diffusers key remap (vendored from InvokeAI,
  Apache-2.0; matches the unmerged upstream PR) — then composes with the base
  repo's components (`base_model` REQUIRED, e.g. `krea/Krea-2-Turbo`; gated,
  Krea 2 Community License). fp8 checkpoints stay **fp8-resident** (~13 GB,
  layerwise float8 storage + bf16 compute; `KREA2_FP8_STORAGE=1|0` overrides
  the auto). Turbo defaults: `num_steps=8`, `guidance_scale=0.0` (Krea CFG
  convention: 0 = off, velocity `cond + g·(cond−uncond)`); `negative_prompt`
  only acts when g > 0; scheduler name ignored; **t2i only** (no diffusers
  img2img yet — upstream PR #14290). New `[krea2]` extra (byte-identical to
  the other image extras, folded into `[runtime]`), GPU-free unit tests
  (`test_krea2_convert`, `test_krea2_backend_flags`), a gated e2e smoke
  (`IMFERENCE_TEST_KREA2_PATH`/`_BASE`), and a `base_models.yaml` validation
  row (Comfy-Org/Krea-2 `krea2_turbo_fp8_scaled`). **GPU-validated e2e
  2026-08-27** (RTX PRO 4000 Blackwell 24 GB): the official Turbo scaled-fp8
  AND a civitai plain-fp8 finetune (GonzaLomo Krea 2 v4.0 — 82.9 s total with
  cached base components) both render clean at seed 42.

## [0.4.0] — 2026-08-26

### ⚠️ Breaking

- **diffusers pinned 0.39.0 → 0.40.0 across EVERY extra** (the 7 image
  backends, `[wan]`, and `[minimax-h3]` — one diffusers repo-wide again).
  0.40.0 ships H3's PR #14355, which is what unblocks the fold. Consumer
  impact: mixed-rank LoRAs without alpha keys now load at their intended
  scale (upstream fix — previously arbitrary and key-order-dependent), and
  Flax/`Flax*` classes are gone from diffusers (unused here). GPU-validated
  in full on 2026-08-26 (RTX PRO 4500 Blackwell, torch 2.11+cu128): 7 image
  backends + Wan t2v/i2v + H3 — see `validation/README.md` for the caveats
  (qwenimage engine-residency path needs a ≥48 GB card).
- **Version 0.4.0** (dependency-combo change per RELEASING.md). Also re-syncs
  `imference_engine.__version__`, which had drifted to "0.3.1" while
  pyproject said "0.3.4".

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

- **MiniMax-H3's dedicated-venv era is over.** PR #14355 shipped in diffusers
  0.40.0 (2026-08), so `[minimax-h3]` pins `diffusers==0.40.0` — the same
  repo-wide pin as every other extra — instead of documenting a
  `pip install git+...@refs/pull/14355/head` side-install; the loader's guard
  message now points at a stale-venv reinstall. Supersedes the "requires
  unreleased diffusers" warning on the H3 entry below. Prod stays on the
  validated torch 2.11+/cu128 combo even though released 0.40.0 dropped the
  PR head's `torch.nn.functional.ScalingType` import (the source of the old
  torch>=2.10 floor).
- **`torch_dtype=` → `dtype=` across all backends** (sdxl, sd15, zimage,
  flux, chroma, qwenimage, anima, wan; H3 already used `dtype`). diffusers
  0.40 deprecates `torch_dtype` (removal slated for v1.0) and transformers 5
  prefers `dtype` — both accept it today, so this is warning-hygiene, not a
  behavior change.
- **Shared video catalogs across engines.** Video rows are now validated
  against every *known* video arch (`imference_engine.video.KNOWN_VIDEO_ARCHS`)
  and each engine registers only its own — one `models.yml` can carry `wan`
  and `minimax_h3` rows without either engine rejecting the other's (a typo'd
  arch still fails loudly). Previously `WanEngine` raised `CatalogError` on any
  non-wan video row.

### Fixed

- **transformers pinned 5.1.0 → 5.4.0 (image extras; `[wan]` floor ≥5.4,
  `[minimax-h3]` floor ≥5.3).** The RELEASED diffusers 0.40.0 H3 text-encoder
  step diverged from the validated PR #14355 head: it now calls
  `Qwen3VLProcessor.create_mm_token_type_ids` (transformers ≥5.2) and passes
  `mm_token_type_ids` into the Qwen3-VL forward, which the model only accepts
  from transformers **5.3.0** — on 5.1.0 the pipeline fails with
  `AttributeError: 'Qwen3VLProcessor' object has no attribute
  'create_mm_token_type_ids'`. Caught by `validate_h3.py` on the 0.40
  validation pod (2026-08-26); H3 renders green on 5.4.0 (124f + soundtrack,
  int8 R2 mirror, block offload). The whole stack (7 image backends, Wan
  t2v/i2v, H3) was re-validated on the final combo: diffusers 0.40.0 +
  transformers 5.4.0 + peft 0.19.1 + torchao 0.18 + accelerate 1.12.0.
- **peft pinned 0.18.1 → 0.19.1 (all extras); torchao floor raised to 0.18 in
  `[minimax-h3]`.** peft 0.18.1's LoRA torchao dispatcher imports
  `LinearActivationQuantizedTensor`, removed in torchao 0.18 — with torchao
  importable in the venv, **any** `load_lora_weights` call crashes
  (`ImportError` at `peft/tuners/lora/torchao.py`). Harmless while H3's
  torchao lived in its own venv; fatal in the unified 0.40 venv where
  `[wan]` (Lightning LoRA) and `[minimax-h3]` (torchao) coexist — caught by
  `validate_wan.py` i2v on the 0.40 validation pod (2026-08-26). peft 0.19.1
  dispatches via `torchao.utils.TorchAOBaseTensor` (v2 API) and coexists with
  torchao 0.18. The torchao floor moves to 0.18 because the R2 int8 mirror
  tree is serialized with it.
- **`protobuf` added to every image extra and `[wan]`.** transformers 5.1 needs
  it to convert a slow sentencepiece tokenizer (a base repo shipping only
  `spiece.model`, no `tokenizer.json` — Chroma1-HD does) and otherwise falls
  back to a tiktoken extractor that cannot parse the file
  (`ValueError: tiktoken is required…`). A clean venv built from the extras
  alone never worked for Chroma — earlier validation boxes had protobuf
  transitively. Caught on a clean pod during the 0.40.0 GPU validation
  (2026-08-26).
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
