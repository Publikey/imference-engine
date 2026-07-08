# Catalog YAML loader — design

Status: **Phases A + B implemented; disk_cache (C) + remote_sync (D) pending**.
This document is the plan for the `imference_engine/catalog/` package. Phase A
shipped the precedence chain (`catalog/defaults.py`, `engine_defaults()` on
backends, the `Optional`/merge refactor of `Engine.generate()`,
`RegisteredModel.defaults`). Phase B shipped `catalog/loader.py` (`models.yml` ->
`list[ModelConfig]` with strict validation) and the `Engine(catalog_path=...)` /
`Engine.load_catalog()` wiring. `disk_cache.py` / `remote_sync.py` remain stubbed
in `catalog/__init__.py`.

It replaces one-by-one `Engine.register_model(...)` calls with a declarative
`models.yml`, and — critically — introduces a clean **defaults precedence
chain** so per-model sampler settings have somewhere to live. It is the
image-side equivalent of the Wan sub-package's `presets.py` (`WanVariant` /
`BUILTIN_VARIANTS`), YAML-driven and hot-reloadable instead of a hard-coded
dict.

## 0. Guiding principle — the 3-layer defaults precedence chain

Engine defaults and model defaults are **different layers** and must not be
conflated. Three layers, resolved general → fine:

| Layer | Lives in | Granularity | Example |
|---|---|---|---|
| **1. Engine defaults** | code, on the `PipelineBackend` | per **engine family** | all SDXL → `EulerAncestral`; all Z-Image → flow matching, `guidance≈1` |
| **2. Model defaults** | `models.yml`, `defaults:` block | per **checkpoint** | *this* Z-Image-Turbo → `shift=3.0, steps=8`; *this* SDXL anime finetune → `clip_skip=2, DPM++Karras` |
| **3. Request params** | `Engine.generate()` args | per **call** | caller passes `num_steps=50` |

**Resolution: `request > model > engine`** (finest wins). The effective value
is the first non-`None` walking down the stack.

Above the chain, **not overridable**: engine invariants (dtype, attention,
memory_format) stay hard-coded in the backend — they are NOT part of this
chain.

### The technical trap that breaks the chain

`generate()` today has concrete defaults in its signature
(`width: int = 1024`, `num_steps: int = 28`, `guidance_scale: float = 6.0`).
Once `1024` is substituted by the signature, you can no longer tell "caller
asked for 1024" from "caller said nothing" — so a model default `width: 768`
can never win, because `1024` already looks explicit.

➡️ **Prerequisite**: the overridable params of `generate()` become
`Optional[...] = None`. `None` = "not set at this layer", so a lower layer can
fill it. This is the pivot refactor (Phase A below).

## 1. Data structures

### `GenerationDefaults` — `catalog/defaults.py` (new)

Common carrier across all three layers. Every field `Optional`; `None` = unset.

```python
@dataclass
class GenerationDefaults:
    num_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    scheduler: Optional[str] = None
    clip_skip: Optional[int] = None
    negative_prompt: Optional[str] = None
    strength: Optional[float] = None
    # engine-specific knobs (Z-Image shift, ...) — free dict, merged key-wise
    backend_options: dict = field(default_factory=dict)

    def merged_over(self, base: "GenerationDefaults") -> "GenerationDefaults":
        """self wins where set; base fills the rest. backend_options merges
        key-wise (self overrides base per key)."""
        out = {}
        for f in fields(self):
            if f.name == "backend_options":
                out[f.name] = {**base.backend_options, **self.backend_options}
            else:
                sv = getattr(self, f.name)
                out[f.name] = sv if sv is not None else getattr(base, f.name)
        return GenerationDefaults(**out)
```

**Why `shift` stays in `backend_options`, not a first-class field:** keep the
core engine-agnostic. `shift` only exists for Z-Image; promoting it pollutes the
struct for SDXL/FLUX. The free dict + key-wise merge handles it and matches
today's `backend_options={"shift": 3.0}` API.

### `ModelConfig` — catalog view (new)

```python
@dataclass
class ModelConfig:
    name: str
    engine: str                       # = backend key ("sdxl" | "zimage" | ...)
    weights: str                      # local path OR repo/URL (disk_cache resolves)
    base_model: Optional[str] = None
    defaults: GenerationDefaults = field(default_factory=GenerationDefaults)  # layer 2
```

`RegisteredModel` (in `managers/model.py`) stays the ModelManager's internal
residency/LRU struct — we just **add `defaults: GenerationDefaults`** to it so
`get_or_load` can hand it back to `generate()`. `ModelConfig` is the catalog
view; `register_model` bridges the two.

## 2. Layer 1 — engine defaults on the backend

Add to the ABC (`pipelines/base.py`). The base returns an **empty**
`GenerationDefaults()` (no opinion); the concrete global fallbacks live once in
`GLOBAL_DEFAULTS` (see §0/§3) so the magic numbers are not duplicated across
backends:

```python
def engine_defaults(self) -> GenerationDefaults:
    """Family-wide defaults for this engine. Return only what is opinionated
    for the family; anything left None falls through to GLOBAL_DEFAULTS.
    (dtype/attention invariants do NOT go through here.)"""
    return GenerationDefaults()
```

- **SDXL** override → `GenerationDefaults(scheduler="EulerAncestralDiscreteScheduler")`.
- **Z-Image** override → `GenerationDefaults(guidance_scale=1.0)`; no `scheduler`
  (ignored by the backend), no `shift` (that is a *model* default — a non-turbo
  Z-Image wants a different shift).

The engine layer thus carries only family opinions; the concrete
`1024/28/6.0/0.75` fallbacks sit below it in `GLOBAL_DEFAULTS`, in one place, off
the `generate()` signature.

## 3. The merge in `Engine.generate()`

Overridable params become `Optional = None`. Body:

```python
req = GenerationDefaults(num_steps=num_steps, guidance_scale=guidance_scale,
                         width=width, height=height, scheduler=scheduler,
                         clip_skip=clip_skip, negative_prompt=negative_prompt,
                         strength=strength, backend_options=backend_options or {})

meta = self._models.config_for(model)          # ModelConfig (layer 2)
eff = backend.engine_defaults()                # layer 1
eff = meta.defaults.merged_over(eff)           # layer 2 beats 1
eff = req.merged_over(eff)                      # layer 3 beats 2
eff = eff.merged_over(GLOBAL_HARD_FALLBACK)     # safety net for anything still None
```

Then `apply_scheduler(pipe, eff.scheduler, **eff.backend_options)`,
`build_inference_kwargs(... eff.num_steps ...)`, etc. **No sampling logic in the
caller** — the engine config philosophy is preserved.

**Backward compatibility:** a caller doing `generate(num_steps=30)` with no
catalog must behave exactly as today. Because `engine_defaults()` carries the
same values as the old signature, and a model registered via `register_model`
(no YAML) has empty `defaults`, the chain degenerates to "request > engine
defaults" = current behavior. Covered by `test_generate_backward_compat.py`.

## 4. `catalog/loader.py`

### YAML schema

```yaml
version: 1
models:
  - name: z-image-turbo
    engine: zimage
    weights: /cache/z-image-turbo.safetensors   # or repo id, or https URL
    base_model: Tongyi-MAI/Z-Image-Turbo
    defaults:                                     # layer 2, all optional
      num_steps: 8
      guidance_scale: 1.0
      backend_options:
        shift: 3.0
  - name: dreamshaper-xl
    engine: sdxl
    weights: /cache/dreamshaper.safetensors
    defaults:
      scheduler: DPMSolverMultistepScheduler
      clip_skip: 2
```

### Responsibilities

1. Parse (`yaml.safe_load`) + **strict validation**: `engine` in known
   backends; `name` unique; `weights` present; keys under `defaults:` are valid
   `GenerationDefaults` fields (an unknown key is an explicit error, not a silent
   drop — otherwise `stpes: 8` passes unnoticed).
2. `backend_options:` is an **explicit nested block** (decision: explicit over
   "unknown keys → backend_options" — validatable, no typo trap).
3. Returns `list[ModelConfig]`.
4. Opens **no weight files** — pure metadata read (like `presets.py` for Wan).

### Decision: explicit `backend_options:` sub-block

Engine-specific knobs (e.g. Z-Image `shift`) go under an explicit
`backend_options:` sub-block rather than flat keys auto-routed to the free dict.
Rationale: validatable, clear cross-engine vs engine-specific boundary, no
silent-typo trap (a flat `shft: 3.0` would land unnoticed in `backend_options`).

## 5. `catalog/disk_cache.py`

On-**disk** LRU for remote weights. Decoupled from `ModelManager` (which owns
RAM/VRAM residency).

- **Role:** when `weights` is a repo/URL, resolve → local `.safetensors` under a
  size budget, evict the disk-LRU.
- **Interface:** `resolve(weights_ref) -> local_path`, `protect(name)`,
  `unprotect(name)`.
- **Coordination:** exactly what the existing lifecycle hooks plug into —
  ```python
  engine.set_lifecycle_hooks(on_model_loaded=disk_cache.protect,
                             on_model_evicted=disk_cache.unprotect)
  ```
  Two coordinated LRUs: memory (`ModelManager`) protects disk (`disk_cache`) via
  the hooks. The mechanism already exists; disk_cache is the missing consumer.
  Lifted near-verbatim from the worker.
- **Optional at first cut:** with a local `weights` path (desktop / worker with
  a pre-populated cache) disk_cache is bypassed. Phase C, not blocking for the
  loader.

## 6. `catalog/remote_sync.py`

Hot-reload `models.yml` from an HTTP URL: poll → diff → `register` new / drop
removed. The URL becomes a constructor arg (was hardcoded in the worker).
Phase D — when prod needs it. Not blocking.

## 7. Wiring in `Engine`

`Engine.__init__(catalog_path=...)` already exists as a stub. At `load()` (or an
explicit `load_catalog()`):

```python
if self._catalog_path:
    for mc in loader.load(self._catalog_path):
        self._models.register(RegisteredModel(
            name=mc.name, backend=mc.engine, weights_path=mc.weights,
            base_model=mc.base_model, defaults=mc.defaults))
```

Manual `register_model()` stays — it builds a `RegisteredModel` with an empty
`GenerationDefaults()`. Both paths coexist.

## 8. Phasing

| Phase | Content | Blocks |
|---|---|---|
| **A** | `GenerationDefaults` + `engine_defaults()` on backends + `generate()` `Optional`/merge refactor + regression tests | everything else |
| **B** | `loader.py` + `ModelConfig` + `catalog_path` wiring + YAML validation | 3rd backend (FLUX/Qwen) |
| **C** | `disk_cache.py` + hook coordination | remote weights (Civitai/CDN on-demand) |
| **D** | `remote_sync.py` | prod hot-reload |

**A + B** is the real "catalog loader" deliverable. C and D follow demand.

### Decision: `generate()` signature refactor

Overridable params move to `Optional = None` directly (not a `_UNSET`
sentinel behind an unchanged-looking signature). Rationale: it is the clean
prerequisite for the precedence chain; behavior with no catalog stays identical,
guaranteed by `test_generate_backward_compat.py`. The sentinel alternative keeps
`1024/28/6.0` visible in the signature but muddies the code for a cosmetic gain.

## 9. Tests

- `test_defaults_precedence.py`: engine < model < request, including key-wise
  `backend_options` merge, and "None does not shadow a lower layer".
- `test_catalog_loader.py`: parse OK; unknown `engine` → error; unknown
  `defaults` key → error; duplicate `name` → error.
- `test_generate_backward_compat.py`: with no catalog, behavior identical to
  today.
- YAML fixtures under `tests/`.

## Why this before the 3rd backend

With 2 backends and hand-registration, `register_model`'s 4 fields suffice. With
FLUX (distilled guidance), Qwen (shift-like), SDXL (clip_skip/scheduler),
Z-Image (shift), per-model defaults multiply and `register_model` cannot carry
them — you would end up hard-coding sampling in callers, exactly what the engine
config philosophy forbids. Land the catalog loader (Phase A + B) before the
third backend.
