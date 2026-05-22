# imference-engine

Unified Python inference engine for Diffusers-based image generation. Single
codebase serves both the gen-image / Imference worker fleet (Runqy + GPU
workers) and the upcoming Imference Desktop sidecar.

## Status

Early extraction from `gen-image-worker/workers/sdxl-multimodel` and
`zimage-multimodel`. The two workers shared ~80% of the code; this package
unifies them behind a single `Engine` API and adds the abstraction needed for
future desktop / MPS / quantization support.

Inference + backends are wired (SDXL + Z-Image). The multi-tier `ModelManager`
(GPU LRU + optional CPU LRU) is in. Remaining gaps: `LoRAManager`, img2img
(`Engine.generate(source_image=...)` still raises `NotImplementedError`), and
the catalog YAML loader (callers register models one by one for now).

## Scope

- **In:** SDXL, Z-Image pipelines, LoRA stacking, dynamic GPU batch sizing,
  CPU/GPU/disk LRU model management, weighted prompt embeddings with BREAK
  keyword support.
- **Out (intentionally):** Wan video, ComfyUI workflows, cloud-API wrappers
  (Azure, Vertex, OpenAI). Those stay as their own one-shot workers — they
  don't share an inference loop with diffusion models.
- **Future:** MPS (Apple Silicon), CPU fallback, quantization
  (bitsandbytes / quanto / gguf), Civitai download, user-data-dir conventions
  for desktop sidecar.

## Boundaries

The engine performs **no network I/O on the result side**. `generate()` returns
PIL images, seeds, and per-image errors. The caller decides what to do with
them — upload to Azure, return inline, POST a webhook, hand to Electron, etc.
This is the clean cut between *engine* (pure inference) and *transport*
(workers, sidecar, etc.).

## Install

```bash
pip install -e ".[runtime,dev]"
# minimal (catalog / batch-sizing logic only, no torch):
pip install -e ".[dev]"
```

## Usage

### Desktop / single-user (defaults — single resident model)

```python
from imference_engine import Engine, RuntimeConfig

engine = Engine(runtime=RuntimeConfig(device="auto")).load()
engine.register_model("sdxl", backend="sdxl", weights_path="/path/to/sdxl.safetensors")

result = engine.generate(
    model="sdxl",
    prompt="masterpiece, best quality, ...",
    negative_prompt="lowres, ...",
    width=1024, height=1024,
    num_steps=30, guidance_scale=7,
    scheduler="EulerAncestralDiscreteScheduler",
    batch=4,
    seed=42,
)
# result.images: list[PIL.Image | None]
# result.seeds:  list[int]
# result.errors: list[GenerationError]
```

### Cloud worker / multi-model (GPU LRU + CPU warm cache)

```python
engine = Engine(runtime=RuntimeConfig(
    device="auto",
    max_gpu_models=2,    # up to 2 pipes concurrently in VRAM
    max_cpu_models=8,    # up to 8 demoted-but-warm pipes in CPU RAM
)).load()

# Plug disk-cache lifecycle so .safetensors actively in use can't be
# garbage-collected by a separate disk-pressure monitor.
engine.set_lifecycle_hooks(
    on_model_loaded=disk_cache.protect,
    on_model_evicted=disk_cache.unprotect,
)

for model_meta in catalog.entries():
    engine.register_model(model_meta.name, backend=model_meta.engine,
                          weights_path=model_meta.path, base_model=model_meta.base)

# Repeated switches between A/B/C are now ~0.5s swaps instead of 10-30s disk reads,
# as long as they fit in (max_gpu + max_cpu) total residency.
result = engine.generate(model="some-model", prompt="...", ...)
```

## Layout

```
imference_engine/
  engine.py            # public Engine class
  types.py             # GenerationResult, GenerationError, RuntimeConfig
  pipelines/
    base.py            # PipelineBackend ABC
    sdxl.py            # (PR2) SDXL backend
    zimage.py          # (PR2) Z-Image backend
  managers/
    batch.py           # BatchSizer (lifted, generalized)
    model.py           # ModelManager — GPU + optional CPU LRU
    lora.py            # (TBD) LoRAManager — dynamic stacking
  catalog/
    loader.py          # (PR2) YAML model registry
    disk_cache.py      # (PR2) on-disk LRU for downloaded weights
    remote_sync.py     # (PR2) hot-reload models.yml from HTTP
  prompting/
    weighted.py        # (PR2) sd_embed wrapper + BREAK keyword
  runtime/
    device.py          # cuda | mps | cpu detection
    resources.py       # (PR2) RAM/disk/VRAM detection lifted from worker
```
