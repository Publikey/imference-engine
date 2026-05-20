# imference-engine

Unified Python inference engine for Diffusers-based image generation. Single
codebase serves both the gen-image / Imference worker fleet (Runqy + GPU
workers) and the upcoming Imference Desktop sidecar.

## Status

Early extraction from `gen-image-worker/workers/sdxl-multimodel` and
`zimage-multimodel`. The two workers shared ~80% of the code; this package
unifies them behind a single `Engine` API and adds the abstraction needed for
future desktop / MPS / quantization support.

The package shape is stabilizing; **inference is not wired up yet** — the
`Engine` class currently raises `NotImplementedError`. Lifting `ModelManager`,
`LoRAManager`, and the two concrete backends comes in subsequent commits.

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

## Usage (target API)

```python
from imference_engine import Engine, RuntimeConfig

engine = Engine(
    catalog_path="models.yml",
    runtime=RuntimeConfig(device="auto", max_gpu_models=1),
)
engine.load()

result = engine.generate(
    model="illustrijevo",
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
    model.py           # (PR2) ModelManager — CPU/GPU LRU
    lora.py            # (PR2) LoRAManager — dynamic stacking
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
