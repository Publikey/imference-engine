# Krea 2 (Turbo) backend

Krea AI's **Krea 2** — a 12.9B single-stream flow-matching DiT (GQA, gated
sigmoid attention, 3-axis RoPE) conditioned by a **Qwen3-VL-4B** text encoder
(hidden states tapped at 12 layers, fused inside the transformer) and decoded
by the **Qwen-Image VAE**. Open-weights June 2026 under the Krea 2 Community
License; diffusers `Krea2Pipeline` since 0.39 (modular in 0.40).

This backend targets **Krea 2 Turbo** — the TDM few-step distillate — and the
civitai/ComfyUI finetune ecosystem around it (CyberRealistic Krea 2, Moody
Krea 2 Mix, …). Output is native 1K–2K.

## The load path (why this backend has a `convert.py`)

Civitai/ComfyUI Krea 2 checkpoints are **transformer-only single-files in the
NATIVE key layout** (`blocks.N.attn.wq`, `txtfusion.*`), predominantly ComfyUI
**"scaled fp8"** (float8_e4m3fn weights + per-tensor float32 `weight_scale` +
`_quantization_metadata` header). diffusers (0.40, and `main` as of 2026-08-27)
has **no `from_single_file` for Krea 2** (issue #14122; PRs #14126/#14264
unmerged) and **no scaled-fp8 handling for any model** — so nothing upstream
loads these files.

`convert.py` bridges the gap **in memory at load time** (no on-disk conversion,
no offline step; register the civitai file as-is):

1. strip an optional `model.diffusion_model.` prefix (all-in-one checkpoints);
2. dequantize scaled fp8 exactly — `w = w_fp8.float() × weight_scale`, computed
   in fp32, stored per-tensor in bf16 (lossless w.r.t. the distributed file;
   raw-fp8 files without scales are upcast too);
3. remap native → diffusers `Krea2Transformer2DModel` keys (vendored from
   InvokeAI, Apache-2.0 — the same mapping as the unmerged upstream PR #14126;
   mixed-layout files are rejected, incomplete loads fail loudly at load time).

When upstream ships single-file + scaled-fp8 support, `convert.py` can be
dropped for `Krea2Transformer2DModel.from_single_file`.

## fp8-resident storage

An fp8-on-disk checkpoint is kept **fp8-resident**: after the exact dequant,
the transformer re-casts to float8_e4m3fn storage with bf16 compute (diffusers
layerwise casting — the InvokeAI approach). ~13 GB resident instead of ~26 GB,
which is the whole point of the fp8 ecosystem on 16–24 GB cards. Auto when the
source was fp8 and CUDA is available; `KREA2_FP8_STORAGE=1|0` forces it.
bf16 checkpoints load as plain bf16 (~26 GB) — pair with
`RuntimeConfig(enable_offload=True)` on consumer VRAM.

## Behavior

| knob | behavior |
|---|---|
| `base_model` | **REQUIRED** — diffusers base repo for the Qwen3-VL encoder + VAE + tokenizer + scheduler. `krea/Krea-2-Turbo` (gated: accept the license, `hf auth login`; mirror on the CDN for offline workers). |
| `guidance_scale` | **Krea convention**: velocity `cond + g·(cond−uncond)`; `0.0` disables guidance (the Turbo norm and the engine default). Conventional CFG scale ≈ `1 + g`. Raw/undistilled rows want ~4.5. |
| `negative_prompt` | Passed through; the pipeline **ignores it whenever `guidance_scale <= 0`**. |
| `num_steps` | Engine default **8** (Turbo/TDM). Raw checkpoints: ~28–52 via catalog row. |
| `scheduler` / `shift` | Ignored — FlowMatchEulerDiscreteScheduler with dynamic shifting; Turbo checkpoints carry `is_distilled=true` → fixed `mu=1.15` inside the pipeline. |
| `clip_skip` | Ignored (no CLIP anywhere). |
| img2img | **Unsupported** (`make_img2img` raises) — no diffusers Krea 2 img2img yet (upstream PR #14290). |
| LoRA | Not wired (engine-wide V1 limitation); diffusers has `Krea2LoraLoaderMixin` for the future image-LoRA work. |

Catalog row example:

```yaml
models:
  - name: cyberrealistic-krea2
    engine: krea2
    weights: /cache/CyberRealistic_Krea2_FP8.safetensors   # civitai file, as-is
    base_model: krea/Krea-2-Turbo
    # engine defaults already carry the Turbo recipe (8 steps, guidance 0.0)
```

## Known upstream quirks (handled in `backend.py`)

- The base repo's `model_index.json` declares the slow `Qwen2Tokenizer` but
  ships only `tokenizer.json` → the tokenizer is loaded via `AutoTokenizer`
  (fast) with `extra_special_tokens={}` and injected explicitly.
- Older transformers read Qwen3-VL rope settings from `rope_scaling` while the
  repo stores `rope_parameters`. Not patched — the repo floor
  (transformers ≥ 5.3) reads `rope_parameters` natively. Revisit if a
  validation run crashes in the rotary embedding.
- The 0.40.0 transformer already contains the GQA `repeat_interleave` fix
  (upstream #14523) — no VRAM blow-up with attention masks.

## Status

Wired; unit-tested GPU-free (`tests/test_krea2_convert.py`,
`tests/test_krea2_backend_flags.py`); e2e smoke gated by
`IMFERENCE_TEST_KREA2_PATH` / `IMFERENCE_TEST_KREA2_BASE`. **First GPU
validation run pending** — `python validation/validate.py --engines krea2`
(the `base_models.yaml` row loads Comfy-Org/Krea-2's
`krea2_turbo_fp8_scaled.safetensors`, exercising the exact civitai load path).
