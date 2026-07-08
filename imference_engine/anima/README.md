# Anima backend configuration

> ⚠️ **Modular Diffusers backend — partially unverified.** Anima is the only
> backend not built on the standard `DiffusionPipeline` API. It is wired from the
> diffusers docs/source but **not yet run e2e**; the GPU validation gate is the
> modular `__call__` kwargs (see below). Isolated in its own sub-package so it
> can't destabilize the standard-pipeline backends.

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
| scheduler | Block-defined in the modular pipeline; the `scheduler` name arg is ignored. |
| device / residency | `pipe.to(...)` is supported → the ModelManager's GPU/CPU moves work. |
| offline flat-tree / CDN | NOT plumbed through the modular loader here (unlike the other backends). |

## Verified vs unverified

**Verified** (diffusers docs/source, main + v0.39.0): loads via `ModularPipeline`;
`pipe.to("cuda")` works; `pipe(prompt=...).images[0]` returns images; bf16;
`AnimaModularPipeline` / `AnimaAutoBlocks` / `AnimaTextConditioner` exist.

**Unverified — the GPU validation gate:** the documented example passes only
`prompt`. This backend also passes `num_inference_steps`, `guidance_scale`,
`width`, `height`, `generator`, `num_images_per_prompt`, and `negative_prompt`
(when set). If the modular `__call__` rejects any of these on the first run,
trim it in `build_inference_kwargs` / `encode_prompts` (the module docstring
points to the exact spot). `num_images_per_prompt` (batching) is the most likely
to need removal.

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
