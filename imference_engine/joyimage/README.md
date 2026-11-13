# JoyAI-Image-Edit-Plus (`joyimage-edit`)

> **Status: written and unit-tested, NOT validated on a GPU.** The validation pod
> was lost before this backend could be run at all. Nothing below has been
> measured — including whether it fits a 24 GB card. Do not offer it to users before
> running the commands in [Validation still to do](#validation-still-to-do).

JD's multi-image instruction-editing model: a 16B MMDiT conditioned on
**Qwen3-VL-8B** (which also looks at the reference images) with the Wan VAE.
**Apache-2.0**, ungated. `JoyImageEditPlusPipeline` ships in diffusers 0.40.

```python
engine.register_model("joy", backend="joyimage-edit", weights_path="auto")
out = engine.generate(
    model="joy",
    prompt="The woman is lovingly holding the cute puppy in her arms",
    source_image=[woman, puppy],          # 1 to 5 references
).images[0]
```

## What it does

| | |
|---|---|
| `source_image` | required — one image or a list of up to five references |
| output size | follows the LAST reference (the pipeline's 1024-base buckets) |
| recipe | 30 steps, true CFG 4.0, negative prompt honored |
| `strength`, `scheduler`, `width`, `height` | ignored |
| batch | one image per pipeline call: a batch is rendered image by image |
| LoRAs | not supported (no loader in the diffusers pipeline) |

There is no text-to-image mode. The single-image `JoyAI-Image-Edit` checkpoint uses a
different pipeline class and is not wired.

## Weights

`weights_path`: `"auto"` (`jdopensource/JoyAI-Image-Edit-Plus-Diffusers`) or a repo id /
directory in the diffusers layout. No single-file / GGUF loader exists for this
transformer in diffusers 0.40 (the public GGUF files are ComfyUI builds), so the
builds store the bf16 weights in fp8.

## Builds per machine (not measured)

| | bf16 | fp8 (expected) |
|---|---|---|
| transformer | 32.5 GB | ~16 GB |
| Qwen3-VL-8B encoder | 17.5 GB | ~9 GB |

Neither pair fits 24 GB resident. `builds.py` therefore offers a single build for
24 GB cards — fp8 storage with model offload — and group offload below. Its peak
should be the fp8 transformer plus activations; with true CFG and several reference
images that may exceed 24 GB. This is the first thing to verify. The 2026-11 audit
rated this model "cloud first, 24 GB minimum".

It will be slow in any case: 60 passes of a 16B transformer per image.

| variable | values | default |
|---|---|---|
| `JOYIMAGE_TRANSFORMER` | `auto` · `bf16` · `fp8` | `auto` |
| `JOYIMAGE_ENCODER` | `auto` · `bf16` · `fp8` | `auto` |

## Validation still to do

On a 24 GB card with at least 64 GB of RAM (the bf16 weights are loaded before the
fp8 cast):

```bash
python validation/e2e_backend.py --backend joyimage-edit --source ref.png --runs 1 \
    --prompt "make it night with a full moon, keep the subject unchanged" --tag JOY_auto
python validation/e2e_backend.py --backend joyimage-edit --offload group --source ref.png \
    --runs 1 --prompt "make it night with a full moon, keep the subject unchanged" --tag JOY_group
```

Check in this order: that the fp8 encoder does not break the pipeline (it hooks the
last language-model layer to read hidden states); that model offload fits 24 GB with
one reference, then with two (`--source` takes one image: test a list from Python);
then compare an fp8 render with a bf16 one on a larger card if available. Rewrite
`builds.py` and this README with the measured numbers.

## CDN layout

```
jdopensource/JoyAI-Image-Edit-Plus-Diffusers/{model_index.json, scheduler/, tokenizer/,
                                              processor/, text_encoder/, vae/, transformer/}
```
