"""Spike P2a — quantize the shared UMT5 text encoder to cut CPU RAM.

The UMT5-XXL text encoder is ~11 GB in bf16 and sits resident in RAM the whole
time under ``enable_model_cpu_offload`` — a big fixed cost that, on small-RAM
hosts, is what forces Q4 (or OOMs). It runs ONCE per generation (encode the
prompt), so it tolerates weight quantization far better than the diffusion
transformer. This spike answers, with numbers, "does it actually save RAM and
does quality hold?".

It loads the text encoder in bf16 / int8 / fp8 (torchao weight-only), prints the
RAM each costs, builds the real Wan pipeline (engine loader) with that encoder,
and renders the SAME prompt/seed so you can eyeball quality side by side.

    pip install torchao
    export WAN_MODEL_CACHE=/workspace/wan-tree WAN_MODEL_CDN=https://…/wan22 HF_HUB_OFFLINE=1
    python p2a_text_encoder_quant.py --quant none  --gguf-quant Q4_K_M
    python p2a_text_encoder_quant.py --quant int8  --gguf-quant Q4_K_M
    python p2a_text_encoder_quant.py --quant fp8   --gguf-quant Q4_K_M   # needs Ada/Hopper

Compare out/te_none.mp4 vs out/te_int8.mp4 (same seed) + the printed RAM lines.
Notes:
- int8 weight-only works on any CUDA GPU (Ampere incl. 3090 Ti).
- fp8 weight-only wants Ada/Hopper (sm89+); on Ampere it may error or be slow.
- Keep --gguf-quant FIXED across runs so only the text encoder changes.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p2a")


def _rss_gb() -> float:
    """Resident set size (process RAM) in GiB, from /proc — no psutil dep."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    except OSError:
        return float("nan")


def _apply_quant(model, quant: str) -> None:
    from torchao.quantization import (float8_weight_only, int8_weight_only,
                                      quantize_)
    if quant == "int8":
        quantize_(model, int8_weight_only())
    elif quant == "fp8":
        quantize_(model, float8_weight_only())
    else:
        raise ValueError(quant)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=["none", "int8", "fp8"], default="none")
    ap.add_argument("--mode", choices=["t2v", "i2v"], default="i2v")
    ap.add_argument("--variant", default=None, help="default: wan22-{mode}-lightning")
    ap.add_argument("--gguf-quant", default="Q4_K_M", help="FIXED across runs")
    ap.add_argument("--image-url",
                    default="https://blob.gen-image.com/small/3e7045c2-0b5b-4c01-bd64-7d16bdf35d90.webp")
    ap.add_argument("--prompt", default="the subject turns and smiles, gentle camera push-in")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from diffusers.utils import export_to_video
    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers import AutoencoderKLWan

    from imference_engine.wan.loader import (SharedComponents, _SHARED_PATTERNS,
                                             _local_repo_dir, build_pipeline)
    from imference_engine.wan.presets import BUILTIN_VARIANTS

    cache_dir = os.getenv("WAN_MODEL_CACHE") or None
    cdn = os.getenv("WAN_MODEL_CDN") or None
    name = args.variant or f"wan22-{args.mode}-lightning"
    variant = BUILTIN_VARIANTS[name]
    # The text_encoder/VAE/tokenizer ship ONLY under the shared T2V base in the
    # flat tree (P1g: they're interchangeable T2V<->I2V). The I2V dir has just
    # configs. So load shared components from here, like the engine's manager.
    shared_base = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    base = _rss_gb()
    log.info("RSS baseline: %.1f GiB", base)

    # --- load the text encoder, measure bf16 then (optionally) quantized ---
    d = _local_repo_dir(shared_base, _SHARED_PATTERNS, cache_dir)
    te = UMT5EncoderModel.from_pretrained(d, subfolder="text_encoder",
                                          torch_dtype=torch.bfloat16)
    gc.collect()
    rss_bf16 = _rss_gb()
    log.info("text encoder bf16: +%.1f GiB (RSS %.1f)", rss_bf16 - base, rss_bf16)
    if args.quant != "none":
        _apply_quant(te, args.quant)
        gc.collect()
        rss_q = _rss_gb()
        log.info("text encoder %s : +%.1f GiB (RSS %.1f)  --> SAVED %.1f GiB",
                 args.quant, rss_q - base, rss_q, rss_bf16 - rss_q)

    tok = AutoTokenizer.from_pretrained(d, subfolder="tokenizer")
    vae = AutoencoderKLWan.from_pretrained(d, subfolder="vae", torch_dtype=torch.float32)
    shared = SharedComponents(variant.base_repo, te, tok, vae)

    # --- build the real pipeline with this encoder + run it ---
    pipe = build_pipeline(variant, quant=args.gguf_quant, shared=shared,
                          device="cuda", enable_offload=True, vae_tiling=True,
                          cache_dir=cache_dir, cdn_base=cdn)
    log.info("pipeline built (RSS %.1f GiB)", _rss_gb())

    call = dict(prompt=args.prompt, negative_prompt="",
                height=args.height, width=args.width, num_frames=args.frames,
                num_inference_steps=args.steps, guidance_scale=1.0,
                guidance_scale_2=1.0, output_type="pil",
                generator=torch.Generator("cpu").manual_seed(args.seed))
    if args.mode == "i2v":
        from diffusers.utils import load_image
        call["image"] = load_image(args.image_url).convert("RGB").resize(
            (args.width, args.height))

    log.info("generating %s %dx%d frames=%d steps=%d seed=%d (te=%s)",
             args.mode, args.width, args.height, args.frames, args.steps,
             args.seed, args.quant)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    frames = pipe(**call).frames[0]
    if torch.cuda.is_available():
        log.info("peak VRAM: %.1f GiB", torch.cuda.max_memory_allocated() / 1024 ** 3)
    log.info("peak RSS during run: %.1f GiB", _rss_gb())

    os.makedirs("out", exist_ok=True)
    out = args.out or f"out/te_{args.quant}.mp4"
    export_to_video(frames, out, fps=args.fps)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
