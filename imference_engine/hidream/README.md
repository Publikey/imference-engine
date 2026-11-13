# HiDream-I1 (`hidream`)

> **Status: written and unit-tested, NOT validated on a GPU.** The validation pod
> was lost before HiDream could be run at all. Nothing below has been measured. Do
> not offer it to users before running the commands in
> [Validation still to do](#validation-still-to-do).

A 17B sparse flow-matching DiT conditioned on four text encoders at once — CLIP-L,
CLIP-G, T5-XXL and **Llama-3.1-8B-Instruct** — with the FLUX VAE.
`HiDreamImagePipeline` ships in diffusers 0.40.

## Licences

| part | licence |
|---|---|
| HiDream-I1 transformer | MIT |
| T5-XXL encoder | Apache-2.0 |
| CLIP-L, CLIP-G encoders | MIT |
| FLUX.1 [schnell] VAE | Apache-2.0 |
| **Llama-3.1-8B-Instruct encoder** | **Llama 3.1 Community License** + Acceptable Use Policy |

The Llama licence allows redistribution when the agreement travels with the weights
and products built on it show "Built with Llama". The mirror copy of the encoder
therefore carries `LICENSE`, `USE_POLICY.md` and a `NOTICE`
(`validation/stage_llama_encoder.py`). The attribution in the product ("Built with
Llama") and the acceptable-use terms are obligations on whoever ships HiDream to
users — Desktop and cloud alike. Not legal advice.

## Checkpoints

| checkpoint | base repo (`base_model`) | recipe |
|---|---|---|
| **Fast** (default) | `HiDream-ai/HiDream-I1-Fast` | 16 steps, guidance 0.0 |
| Dev | `HiDream-ai/HiDream-I1-Dev` | 28 steps, guidance 0.0 |
| Full | `HiDream-ai/HiDream-I1-Full` | 50 steps, CFG 5.0, negative prompt honored |

Each base repo carries its own scheduler, so a Dev or Full transformer must be
registered with its own `base_model`. Steps and guidance follow the file name
(`weights_defaults`) unless a catalog row sets them.

Text-to-image only; the HiDream-E1 editing model is a different pipeline and is not
wired. LoRAs: the pipeline has a loader, not enabled until validated.

## Weights

- `"auto"` — the GGUF picked by `builds.recommend_build`, fetched from
  `IMAGE_MODEL_CDN` or HuggingFace;
- a `.gguf` (city96 `HiDream-I1-{Fast,Dev,Full}-gguf`);
- a transformer-only `.safetensors` (Comfy-Org repackage);
- a diffusers repo id or directory.

The bf16 transformer is 34 GB, so GGUF is the practical form. The loader applies
`core/gguf.dequantize_non_linear` in case the file quantizes a non-linear layer.

**The Llama encoder is not in the HiDream repos.** It is loaded from
`HIDREAM_LLAMA_REPO` (default `unsloth/Meta-Llama-3.1-8B-Instruct`, an ungated copy
of Meta's gated repo — same files, same sha256).

## Builds per machine (not measured)

Nothing here fits a 24 GB card resident: the encoders alone are 29 GB in bf16. Every
build therefore runs with offload, and `HIDREAM_ENCODER=fp8` (the default with CUDA)
stores Llama and T5 in fp8 (about 8.5 GB and 4.8 GB).

| build | VRAM floor | transformer | offload | basis |
|---|---|---|---|---|
| `quality` | 23 GiB | GGUF Q8_0 (18.7 GB) | model | file size + activations: may not fit 24 GB — to verify |
| `balanced` | 15 GiB | GGUF Q4_K_M (11.5 GB) | model | file size + activations |
| `minimal` | below | GGUF Q4_K_M | group | — |

Host RAM will be the other constraint: the four encoders and the transformer are all
held in RAM between requests under model offload (40 GB or more).

| variable | values | default |
|---|---|---|
| `HIDREAM_ENCODER` | `auto` · `bf16` · `fp8` | `auto` |
| `HIDREAM_LLAMA_REPO` | repo id or directory | `unsloth/Meta-Llama-3.1-8B-Instruct` |

## Validation still to do

On a 24 GB card with at least 64 GB of RAM:

```bash
P="a photo of a red fox in a snowy forest, golden hour, sharp focus, 85mm"
python validation/e2e_backend.py --backend hidream --vram 16 --prompt "$P" --tag HD_q4      # Q4_K_M
python validation/e2e_backend.py --backend hidream --prompt "$P" --tag HD_q8                # Q8_0 on 24 GB
HIDREAM_ENCODER=bf16 python validation/e2e_backend.py --backend hidream --vram 16 \
    --prompt "$P" --tag HD_q4_bf16enc                                                         # encoder fp8 vs bf16
```

Things to check first, in this order: that the GGUF loads (HiDream is in diffusers'
single-file list, untested here); that the fp8 Llama and T5 encoders do not break the
pipeline's dtype handling; that Q8_0 actually fits 24 GB with model offload. Then
compare renders, read the `E2E` lines and rewrite `builds.py` and this README.

## CDN layout

```
HiDream-ai/HiDream-I1-Fast/{model_index.json, scheduler/, tokenizer{,_2,_3}/,
                            text_encoder{,_2,_3}/, vae/, transformer/config.json}
city96/HiDream-I1-Fast-gguf/hidream-i1-fast-{Q8_0,Q4_K_M}.gguf
unsloth/Meta-Llama-3.1-8B-Instruct/…  + LICENSE, USE_POLICY.md, NOTICE
```
