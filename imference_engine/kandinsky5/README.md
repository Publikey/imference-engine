# Kandinsky 5.0 Image Lite (`kandinsky5`, `kandinsky5-edit`)

> **Status: written and unit-tested, NOT validated on a GPU.** The validation pod
> was lost during this backend's first runs. Everything below marked "expected" is
> inferred from the code and from LongCat-Image (same transformer size, same
> encoder), not measured. Do not offer it to users before running the commands in
> [Validation still to do](#validation-still-to-do).

A 6B flow-matching model with **Qwen2.5-VL-7B** and **CLIP ViT-L** text encoders and
the FLUX VAE. **MIT**, ungated. Two backends, one per diffusers pipeline:

| backend | pipeline | default checkpoint | recipe |
|---|---|---|---|
| `kandinsky5` | `Kandinsky5T2IPipeline` | `Kandinsky-5.0-T2I-Lite-sft-Diffusers` | 50 steps, CFG 3.5 |
| `kandinsky5-edit` | `Kandinsky5I2IPipeline` (instruction editing) | `Kandinsky-5.0-I2I-Lite-sft-Diffusers` | 50 steps, CFG 3.5 |

```python
engine.register_model("k5", backend="kandinsky5", weights_path="auto")
engine.register_model("k5-edit", backend="kandinsky5-edit", weights_path="auto")
```

Both are undistilled (true CFG, negative prompt honored): expect about a minute and a
half per image, like LongCat-Image text-to-image.

## What is known

- Both pipelines exist in diffusers 0.40.0; the backends load and register
  (`tests/test_kandinsky5_backend_flags.py`).
- The first GPU runs loaded the pipeline and denoised, then failed on the result:
  the Kandinsky pipelines return `.image`, not `.images`. The backend now wraps the
  pipeline to expose `.images` (unit-tested). **That fix has not been run on GPU.**
- The Qwen2.5-VL encoder is byte-identical to LongCat-Image's (same sha256 on
  HuggingFace).
- The repos also carry CLIP as flax / tf / pickle (5 GB of copies): only
  `text_encoder_2/config.json` and `model.safetensors` are pulled.
- No single-file / GGUF loader for this transformer in diffusers 0.40: builds are
  fp8 or bf16 storage, as for LongCat.

## Weights and base

`weights_path`: `"auto"`, a repo id / directory in the diffusers layout, or a
transformer `.safetensors` in the diffusers key layout. A checkpoint contributes only
`model_index.json`, `scheduler/` and `transformer/`; encoders, tokenizers and VAE come
from `base_model` (default: the T2I repo), so the I2I checkpoint does not mirror the
16.6 GB encoder again.

LoRAs: the pipelines carry a loader; `supports_loras` stays False until validated.

## Builds per machine (not measured)

`builds.py` is conservative on purpose: fp8 storage with model offload on every CUDA
card of 12 GiB or more (expected ~10 GB peak, from LongCat), group offload below, and
plain bf16 with model offload without CUDA. Once measured, add resident tiers for
24 GB cards as LongCat has.

| variable | values | default |
|---|---|---|
| `KANDINSKY5_TRANSFORMER` | `auto` · `bf16` · `fp8` | `auto` |
| `KANDINSKY5_ENCODER` | `auto` · `bf16` · `fp8` | `auto` |

## Validation still to do

On a 24 GB card, from the repo root (weights come from `IMAGE_MODEL_CDN` once staged,
else HuggingFace):

```bash
P="a photo of a red fox in a snowy forest, golden hour, sharp focus, 85mm"
# reference (bf16, upstream recipe) and the default build
KANDINSKY5_TRANSFORMER=bf16 KANDINSKY5_ENCODER=bf16 python validation/e2e_backend.py \
    --backend kandinsky5 --offload model --steps 30 --prompt "$P" --tag K5_ref
python validation/e2e_backend.py --backend kandinsky5 --steps 30 --prompt "$P" --tag K5_auto
# resident candidates for a 24 GB tier
KANDINSKY5_TRANSFORMER=fp8 KANDINSKY5_ENCODER=fp8 python validation/e2e_backend.py \
    --backend kandinsky5 --offload none --steps 30 --prompt "$P" --tag K5_fp8_resident
# editing
python validation/e2e_backend.py --backend kandinsky5-edit --steps 30 \
    --source validation/renders/K5_ref/0.png \
    --prompt "make it night with a full moon, keep the fox unchanged" --tag K5E_auto
```

Compare the renders against `K5_ref`, read the `E2E` lines (peak / reserved VRAM,
seconds per call), then rewrite `builds.py` and this README with the measured table.

## CDN layout

```
kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers/{model_index.json, scheduler/, transformer/,
        text_encoder/, tokenizer/, text_encoder_2/{config.json,model.safetensors}, tokenizer_2/, vae/}
kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers/{model_index.json, scheduler/, transformer/}
```
