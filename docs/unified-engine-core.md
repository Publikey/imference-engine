# Unified engine core — design

Status: **design / approved to proceed (pre-prod, breaking changes allowed)**.

## 0. Decision & context

Today there are two parallel worlds: the image `Engine` (+ `PipelineBackend`,
`ModelManager`, `RuntimeConfig`, `GenerationDefaults`/catalog, `GenerationResult`)
and the video `WanEngine` (+ `WanVariant` presets, `ResidencyManager`,
`WanRuntimeConfig`, `WanVideoResult`). ~60% of the machinery
(device/offline/CDN/warm/lifecycle/config-philosophy/offload-residency) is
duplicated in spirit; the genuinely distinct part is GGUF-MoE weight loading and
the video output modality.

**Decision: unify now.** Rationale: not yet in production (breaking changes are
free), more video architectures are coming (HunyuanVideo / LTX-Video / Mochi …),
and every added video model makes unification more expensive. All 8 stacks (7
image + Wan) are now validated e2e on diffusers 0.39 — that is our regression
baseline: re-run `validation/validate.py` + `validation/validate_wan.py` after
each phase.

**The real axis** is not image-vs-video: it is *many-architectures* (image needs
polymorphism → `PipelineBackend`) vs *one-architecture* (Wan is a monolith with
data-driven variants). The moment a 2nd video architecture lands, Wan re-derives
the same backend ABC. So the keystone of this refactor is a **`VideoBackend` ABC
+ Wan becomes a backend**, sitting on a **shared engine core**.

## 1. Target architecture

```
runtime/                     # already shared: device, offline (flat tree+CDN), download, env
core/
  engine_base.py   BaseEngine        # device resolve, cuda tune, warm, lifecycle hooks,
                                      #   catalog load, residency wiring, defaults precedence
  backend.py       Backend (proto)   # load_pipeline, prefetch_base, engine_defaults,
                                      #   get_compute_module, make_generator   (modality-agnostic)
  residency.py     Residency (proto) # get_or_load(spec)->pipe, resident(), lifecycle hooks
  config.py        RuntimeConfig      # shared base: device, model_cache_dir, model_cdn, offload
  catalog/         one models.yml     # image models AND video variants, `kind:` discriminator
  result.py        MediaResult        # media (images|frames) + seeds + errors + optional fps/num_frames

image/
  engine.py        Engine(BaseEngine)         # generate(...) -> MediaResult(kind=image)
  backend.py       ImageBackend(Backend)      # + encode_prompts, apply_scheduler,
                                              #   build_inference_kwargs, make_img2img
  backends/        sdxl, sd15, zimage, flux, chroma, qwenimage, anima
  residency.py     ImageResidency(Residency)  # two-tier GPU-LRU + CPU-LRU (+ offload mode)
  config.py        ImageRuntimeConfig(RuntimeConfig)  # + max_gpu/cpu_models, use_tiny_vae

video/
  engine.py        VideoEngine(BaseEngine)    # generate_video(...) -> MediaResult(kind=video)
  backend.py       VideoBackend(Backend)      # + build_video_kwargs (num_frames/fps/i2v), moe/lora hooks
  backends/        wan  (+ future: hunyuan, ltx, mochi)
  residency.py     VideoResidency(Residency)  # CPU-LRU + offload (Wan today)
  config.py        VideoRuntimeConfig(RuntimeConfig)  # + memory_profile(GGUF), text_encoder_quant, ...
```

Partition line:
- **Shared (`core/` + `runtime/`):** device, offline tree + CDN, `warm()`,
  lifecycle hooks, residency *interface*, config *base*, catalog, `MediaResult`,
  OOM/seed helpers, the "config = single source of truth, env-or-param" contract.
- **Modality-specific:** image = prompt-encoding / scheduler / img2img / batch
  chunking; video = frames/fps/i2v / GGUF-MoE / runtime LoRA / `MemoryProfile`.

## 2. The load-bearing decision — the frontend

**Recommended: one `Engine`-per-modality on a shared `BaseEngine`, keeping the
two natural call surfaces** (`generate()` for images, `generate_video()` for
video). NOT a single polymorphic `generate()` with a params superset (that would
leak `num_frames`/`fps` onto image callers and `clip_skip`/`batch` onto video).

Three options considered:

| Option | Surface | Verdict |
|---|---|---|
| **A. `ImageEngine` / `VideoEngine` : BaseEngine** (rec.) | two engines, shared core, each keeps its call surface | cleanest; no leaky superset; caller picks by modality (as today) but 60% code shared |
| B. One `Engine`, two methods `generate` + `generate_video` | single entry class | feels unified but one class with two disjoint surfaces is awkward; routing by backend modality |
| C. One `Engine.generate()` polymorphic → `MediaResult` | one method | most "unified" surface, leakiest params; rejected |

→ **Option A.** This is the one decision to confirm before Phase 3 (it sets the
public API). Everything else (config/catalog/result/residency) is compatible with
any of them.

## 3. Component targets

- **`BaseEngine`** — owns: `load()` (device resolve + cuda tune), `warm(specs)`,
  `set_lifecycle_hooks`, catalog loading, the `engine < model < request` defaults
  precedence, and holds a `Residency`. Subclasses add the modality `generate*`.
- **`Backend` protocol** — the modality-agnostic slice of today's
  `PipelineBackend`: `load_pipeline`, `prefetch_base`, `engine_defaults`,
  `get_compute_module`, `make_generator`. `ImageBackend` adds prompt/scheduler/
  inference-kwargs/img2img; `VideoBackend` adds video-kwargs + MoE/LoRA hooks.
  Wan becomes `WanBackend(VideoBackend)`.
- **`RuntimeConfig` base** — `device`, `model_cache_dir`, `model_cdn`,
  `enable_offload` (the offload mode already exists on BOTH sides). Image/Video
  configs extend with their specific knobs. `from_env()` stays per-modality
  (different env prefixes: `IMAGE_*` / `WAN_*` → keep, or unify to `IM_*`? —
  minor, decide in Phase 2).
- **Unified catalog** — one `models.yml`; each row has `kind: image|video` and
  `engine:` (backend key). Image rows carry `GenerationDefaults`; video rows carry
  the `WanVariant` fields (mode t2v/i2v, gguf_repo, loras, flow_shift, …) as a
  `video:`/`variant:` sub-block. `WanVariant` becomes the parsed form of a video
  catalog row. **Single source of truth for every model.**
- **`MediaResult`** — `media: list[Image|None]`, `seeds`, `errors`, `kind`,
  and video-only `fps`/`num_frames` (None for images). Replaces
  `GenerationResult` + `WanVideoResult`. One `GenerationError`.
- **`Residency` protocol** — `get_or_load(spec) -> (pipe, backend)`, `resident()`,
  hooks. `ImageResidency` = today's two-tier LRU (note: its `enable_cpu_offload`
  path already mirrors Wan's model). `VideoResidency` = Wan's CPU-LRU + offload.
  Converge the offload code where it's literally the same call.

## 4. Phased migration (each phase = a green checkpoint)

Run both validation harnesses after every phase; all 8 must still render.

- **Phase 1 — leaf types.** Introduce `core/result.py` `MediaResult` +
  `GenerationError`; make `Engine`/`WanEngine` return it. *Breaking:* callers read
  `.media`/`.frames`. Small, mechanical.
- **Phase 2 — config base.** Extract `core/config.py` `RuntimeConfig`; image/video
  configs extend it. Unify shared fields; keep modality knobs in subclasses.
- **Phase 3 — `BaseEngine` + frontend split (Option A).** Extract the shared
  engine core; `ImageEngine`/`VideoEngine` inherit. *Breaking:* import paths /
  class names (`Engine` → `ImageEngine`; `WanEngine` → `VideoEngine`). This is the
  decision to confirm first.
- **Phase 4 — `Backend`/`VideoBackend` ABCs.** Split `PipelineBackend` into the
  shared `Backend` + `ImageBackend`; introduce `VideoBackend`; refactor Wan into
  `WanBackend`. *Enables the 2nd video architecture* — the whole point.
- **Phase 5 — unified catalog.** One `models.yml` schema (`kind:` discriminator)
  covering image defaults AND video variants; `WanVariant` ← catalog. Retire the
  hard-coded `BUILTIN_VARIANTS` into shipped catalog rows.
- **Phase 6 — residency convergence.** One `Residency` protocol, two impls;
  dedupe the offload path.

Phases 1-2 are low-risk plumbing. Phase 3 sets the API (needs sign-off). Phase 4
is the highest-value (video extensibility). Phases 5-6 are consolidation.

## 5. Breaking-change inventory (deliberate, pre-prod)

- `Engine` → `ImageEngine`; `WanEngine` → `VideoEngine` (import paths).
- `RuntimeConfig` → `ImageRuntimeConfig`; `WanRuntimeConfig` → `VideoRuntimeConfig`
  (shared base `RuntimeConfig`).
- `GenerationResult` / `WanVideoResult` → `MediaResult` (`.images`/`.frames` →
  `.media`).
- `register_variant()` → unified `register_model()` / catalog rows.
- Env vars: possibly `IMAGE_*` + `WAN_*` kept, or unified — TBD Phase 2.

Callers to update: the worker fleet + the desktop sidecar (both out-of-repo). No
in-repo callers besides tests + the validation harnesses (updated per phase).

## 6. Risks & mitigations

- **Wan is the only thing that ran in anger** (now on 0.39 too). Mitigation: it's
  validated green *before* we start; re-validate after each phase; refactor in
  small steps (precedent: `runtime/offline.py` was already promoted out of
  `wan/loader.py` without breakage).
- **Over-abstraction.** Keep the `Backend` protocol to what is *actually* shared
  today (5 methods); don't invent hooks for hypothetical modalities.
- **Scope creep.** Each phase lands independently and leaves the tree green; we can
  stop after Phase 4 (the extensibility win) if the rest isn't worth it.

## 7. Progress

- **Phase 1 ✅** `core/result.py` `MediaResult` (+ `GenerationError`). `.images`/
  `.frames` aliases kept callers unchanged.
- **Phase 2 ✅** `core/config.py` `BaseRuntimeConfig`; image/video configs subclass
  it. `enable_offload` unified (env `IMAGE_ENABLE_CPU_OFFLOAD` kept). Decisions:
  1 = unify offload, 2 = keep `IMAGE_*`/`WAN_*` env prefixes, 3 = (b) keep public
  `Engine`/`RuntimeConfig`/`WanEngine` names; base classes internal to `core/`.
- **Phase 3 ✅** `core/engine_base.py` `BaseEngine` (device/load/seed/cache). Both
  engines subclass it; `_setup()` hook holds modality construction.
- **Phase 4 ⏭️ next** — detailed below.

## 8. Phase 4 addendum — VideoBackend / WanBackend / generic residency

**Reality check (good news):** the video builder seam already exists. The Wan
pipeline is NOT built inside the engine — `wan/loader.py` already exposes clean
functions and `ResidencyManager` delegates to them:

- `load_shared_components(base_repo, cache_dir, text_encoder_quant, cdn_base) ->
  SharedComponents` (the shared UMT5 text_encoder + Wan VAE, loaded once).
- `build_pipeline(variant, quant, shared, device, enable_offload, vae_tiling,
  cache_dir, cdn_base) -> pipe`.
- `ResidencyManager` holds the CPU-LRU, calls those two, and has a Wan-specific
  `_teardown(pipe)` (removes accelerate hooks + drops the MoE `transformer` /
  `transformer_2`).

So Phase 4 is mostly **re-homing** that Wan-specific logic behind an ABC, not a
rewrite.

**Honest scope of the shared `Backend` trunk.** Image `PipelineBackend` (load a
single-file diffusion pipe, encode prompts, scheduler, img2img) and a video
backend (assemble GGUF-MoE experts + LoRA + shared components, produce frames)
share almost NO method signatures. Forcing them into one rich interface would be
fake unification. So the shared `core/backend.py` `Backend` trunk stays minimal:

```python
class Backend(Protocol):
    engine: ClassVar[str]                 # id ("sdxl" | "wan" | "hunyuan" | ...)
    def engine_defaults(self) -> GenerationDefaults: ...   # layer-1 defaults
```

`PipelineBackend` (image) and `VideoBackend` (video) each extend it with their
modality methods. The value of Phase 4 is the **`VideoBackend` ABC + generic
residency**, so a 2nd video arch plugs in — NOT a common image/video method set.

**`VideoBackend` ABC** (`core` or `video/backend.py`) — maps 1:1 onto today's
loader functions:

```python
class VideoBackend(Backend):
    def load_shared(self, cfg) -> Any: ...             # <- load_shared_components
    def build(self, spec, shared, cfg) -> pipe: ...    # <- build_pipeline
    def teardown(self, pipe) -> None: ...              # <- ResidencyManager._teardown
    def build_call(self, *, prompt, negative_prompt, width, height,
                   num_frames, num_steps, guidance_scale, guidance_scale_2,
                   image, generator) -> dict: ...      # the pipe(**call) kwargs
    def warm(self, spec, cfg) -> None: ...             # prefetch base + variant cfgs
```

(`cfg` carries quant/device/offload/vae_tiling/cache/cdn — a small builder-config,
derived from `VideoRuntimeConfig`.)

**`WanBackend(VideoBackend)`** — wraps `wan/loader.py`: `load_shared` →
`load_shared_components`, `build` → `build_pipeline`, `teardown` → the current
`_teardown`, `build_call` → the dict `WanEngine.generate_video` builds today,
`warm` → the shared+variant prefetch in `WanEngine.warm`. `engine = "wan"`.

**Generic `ResidencyManager`** — becomes backend-agnostic: holds a
`backend: VideoBackend`, keeps the CPU-LRU, and calls `backend.load_shared()` /
`backend.build(spec, shared)` / `backend.teardown(pipe)`. No Wan specifics remain
in the manager. (Same shape can later host `ImageResidency` in Phase 6.)

**`VideoEngine` (WanEngine) routing.** Today one backend (Wan). To prepare the 2nd
arch, add an `arch` discriminator to the variant (`WanVariant.arch = "wan"`
default) and a `{arch: VideoBackend}` registry on the engine; `generate_video`
selects `backends[variant.arch]`. Until a 2nd arch lands this is a 1-entry map —
but the seam is in place, which is the point.

**Sub-steps (each green + re-validate Wan on GPU):**
- **4a** — add `core/backend.py` `Backend` trunk; make `PipelineBackend` extend it
  (no behaviour change).
- **4b** — add `VideoBackend` ABC + `WanBackend` wrapping `wan/loader.py`; leave
  `ResidencyManager` calling `WanBackend` internally (behaviour identical).
- **4c** — generify `ResidencyManager` to take a `VideoBackend`; add the `arch`
  registry on the engine. Now HunyuanVideo/LTX = a new `VideoBackend` subclass.

**Open question to confirm:** where does `VideoBackend` live — `core/backend.py`
(alongside the trunk) or `video/backend.py` (a new `video/` package, mirroring the
`image/` split the doc sketches)? Recommendation: **`video/backend.py`** in a new
`video/` sub-package that re-exports Wan, so the image/video symmetry is real and
the 2nd arch has an obvious home. That is a larger move (new package) — the
alternative is to keep everything under `wan/` for 4a-4b and only introduce
`video/` when the 2nd arch actually arrives (YAGNI). Lean YAGNI unless a 2nd video
model is imminent.
