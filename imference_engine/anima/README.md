# Anima backend configuration

> ℹ️ **Modular Diffusers backend — validated e2e (text-to-image) on diffusers
> 0.39** (RTX PRO 5000 Blackwell, torch 2.12). Anima is the only backend not built
> on the standard `DiffusionPipeline` API; it adapts the modular API onto
> `PipelineBackend`, and the modular `__call__` accepts the standard kwarg set
> this backend passes. Isolated in its own sub-package. img2img is unsupported
> (no documented modular variant).

Anima (CircleStone Labs + Comfy Org) is a text-to-image model shipped in
diffusers as a **Modular Diffusers pipeline** — there is no standard
`AnimaPipeline`. Architecture: a `CosmosTransformer3DModel` DiT + a Qwen3 text
encoder + an `AnimaTextConditioner` (learned T5 tokens cross-attending Qwen3
hidden states) + the `AutoencoderKLQwenImage` VAE.

## Anima specifics

| Aspect | Behaviour |
|---|---|
| Load | `ModularPipeline.from_pretrained(weights)` + `pipe.load_components(torch_dtype=bfloat16)`. **`weights` is a diffusers-format repo id or local dir** (e.g. `circlestone-labs/Anima-Base-v1.0-Diffusers`), NOT a single .safetensors. |
| `base_model` | Unused (no transformer/base split). |
| `IMAGE_USE_TINY_VAE` | Ignored. |
| img2img | **Not supported** (no documented modular img2img) — `make_img2img` raises; call `generate()` without `source_image`. |
| dtype | `bfloat16`. |
| scheduler | Flow-matching (`FlowMatchEulerDiscreteScheduler`, block-defined). The `scheduler` *name* arg is ignored (DPM/Euler/Karras don't map onto flow matching — same as FLUX/Z-Image); pass `shift` via `backend_options` (e.g. `{"shift": 3.0}`) to rebuild it with a fixed shift. |
| guidance | **`guidance_scale` IS honored** — applied by `apply_guidance` to the `ClassifierFreeGuidance` *guider* component (`pipe.guider.guidance_scale = …`), NOT as a `__call__` kwarg (which the modular pipeline warns on and ignores). Code default is 4.0; `guidance_scale ≈ 1.0` disables CFG and skips the negative-prompt encode. |
| clip_skip | **No-op.** Anima's text stack is Qwen3 + T5 — there is no CLIP layer to skip. |
| device / residency | `pipe.to(...)` is supported → the ModelManager's GPU/CPU moves work. |
| offline flat-tree / CDN | **Supported.** A repo-id `weights` is resolved through `local_repo_dir` (whole modular repo) before `ModularPipeline.from_pretrained`, so with `IMAGE_MODEL_CDN` set it loads from the R2 mirror — same contract as the other backends. A `weights` that is already a local dir is used verbatim. |

## Validation

**Validated e2e (text-to-image) on diffusers 0.39** (RTX PRO 5000 Blackwell, torch
2.12) against `circlestone-labs/Anima-Base-v1.0-Diffusers`: loads via
`ModularPipeline`; `pipe.to(device)` residency moves work; `pipe(...).images`
returns images; bf16. The modular `__call__` **accepts** `num_inference_steps`,
`width`, `height`, `generator`, `num_images_per_prompt`, `negative_prompt` (when
set) and `sigmas` — but NOT `guidance_scale` (set on the guider by
`apply_guidance`) and NOT `clip_skip` (no CLIP). If a future diffusers changes
the modular signature, `apply_guidance` / `apply_scheduler` /
`build_inference_kwargs` / `encode_prompts` are the single spot to adjust.

## Example

```python
from imference_engine import Engine, RuntimeConfig

engine = Engine(runtime=RuntimeConfig(device="auto")).load()
engine.register_model(
    "anima", backend="anima",
    weights_path="circlestone-labs/Anima-Base-v1.0-Diffusers",  # repo id / diffusers dir
)
result = engine.generate(
    model="anima",
    prompt="masterpiece, best quality, 1girl, solo, city lights",
)
```
