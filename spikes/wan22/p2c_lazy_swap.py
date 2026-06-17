"""Spike P2c — REAL lazy-expert swap (the production mechanism, runnable first).

This is the actual swap that will go into the engine behind WAN_LAZY_EXPERTS=1.
It builds the pipeline with ONLY the high-noise expert resident (transformer_2
aliases transformer), then at the boundary crossing a callback_on_step_end frees
the high expert and loads the low one from the GGUF on disk, re-attaching the
cpu-offload hook to the freshly loaded module. Goal: prove (1) the swap produces
a CORRECT video and (2) the real peak RSS drops to ~one expert (~20 GiB vs ~29).

Uses a *baked* variant (dasiwa-i2v, no LoRA) to isolate the expert-swap from the
LoRA layer. If this runs green, the engine integration is a transcription of
`_swap_to_low` + the lazy build, plus re-applying the per-expert LoRA for the
official (non-baked) variants.

    export WAN_MODEL_CACHE=/workspace/wan-tree WAN_MODEL_CDN=https://…/wan22 HF_HUB_OFFLINE=1
    python p2c_lazy_swap.py --variant dasiwa-i2v --gguf-quant Q4_K_M
    # compare out/lazy_swap.mp4 to a normal render; check the peak RSS line
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p2c")


def _rss_gb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    except OSError:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dasiwa-i2v", help="use a BAKED variant (no LoRA)")
    ap.add_argument("--gguf-quant", default="Q4_K_M")
    ap.add_argument("--image-url",
                    default="https://blob.gen-image.com/small/3e7045c2-0b5b-4c01-bd64-7d16bdf35d90.webp")
    ap.add_argument("--prompt", default="the subject turns and smiles, gentle camera push-in")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-lazy", action="store_true", help="baseline: both experts, no swap")
    args = ap.parse_args()

    import torch
    from diffusers import UniPCMultistepScheduler, WanImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    from imference_engine.wan.loader import (
        _BASE_CFG_PATTERNS, _load_gguf_transformer, _local_repo_dir,
        _resolve_gguf, load_shared_components)
    from imference_engine.wan.presets import BUILTIN_VARIANTS

    cache_dir = os.getenv("WAN_MODEL_CACHE") or None
    cdn = os.getenv("WAN_MODEL_CDN") or None
    variant = BUILTIN_VARIANTS[args.variant]
    if variant.loras:
        log.warning("variant %s has LoRA — this spike isolates the swap on a baked "
                    "variant; LoRA re-application is the next layer", args.variant)
    q = args.gguf_quant
    device = "cuda"

    base_dir = _local_repo_dir(variant.base_repo, _BASE_CFG_PATTERNS, cache_dir)
    high_path = _resolve_gguf(variant.gguf_repo, q, "high", variant.gguf_high_name,
                              variant.gguf_high_template, cdn_base=cdn, cache_dir=cache_dir)
    low_path = _resolve_gguf(variant.gguf_repo, q, "low", variant.gguf_low_name,
                             variant.gguf_low_template, cdn_base=cdn, cache_dir=cache_dir)
    shared = load_shared_components("Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                                    cache_dir=cache_dir, text_encoder_quant="int8")

    log.info("loading HIGH expert only (lazy build)…")
    high = _load_gguf_transformer(high_path, base_dir, "transformer")
    low_for_baseline = None
    if args.no_lazy:
        low_for_baseline = _load_gguf_transformer(low_path, base_dir, "transformer_2")

    pipe = WanImageToVideoPipeline.from_pretrained(
        base_dir, transformer=high,
        transformer_2=(low_for_baseline if args.no_lazy else high),  # alias high if lazy
        vae=shared.vae, text_encoder=shared.text_encoder,
        tokenizer=shared.tokenizer, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=variant.flow_shift)
    pipe.enable_model_cpu_offload(device=device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    log.info("pipeline built (RSS %.1f GiB) lazy=%s", _rss_gb(), not args.no_lazy)

    # ---- the swap: free high, load low, re-hook offload (the production core) ----
    swap = {"done": False, "secs": 0.0}

    def _swap_to_low(p) -> None:
        t0 = time.time()
        p.remove_all_hooks()                 # drop offload hooks (incl. high)
        p.transformer = None
        p.transformer_2 = None
        gc.collect()
        torch.cuda.empty_cache()             # free the high expert (~9 GiB)
        low = _load_gguf_transformer(low_path, base_dir, "transformer_2")
        p.transformer = low                  # alias; not used in the low phase
        p.transformer_2 = low
        p.enable_model_cpu_offload(device=device)   # re-attach hooks to `low`
        swap["secs"] = time.time() - t0
        swap["done"] = True
        log.info("  [swap] high -> low done in %.1fs (RSS %.1f GiB)", swap["secs"], _rss_gb())

    def _cb(p, i, t, kw):
        if args.no_lazy or swap["done"]:
            return kw
        br = p.config.get("boundary_ratio", None)
        if br is None:
            return kw
        boundary = br * p.scheduler.config.num_train_timesteps
        ts = p.scheduler.timesteps
        nxt = float(ts[i + 1]) if i + 1 < len(ts) else -1.0
        if 0 <= nxt < boundary:              # next step is the low-noise stage
            _swap_to_low(p)
        return kw

    img = load_image(args.image_url).convert("RGB").resize((args.width, args.height))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    frames = pipe(prompt=args.prompt, negative_prompt="", image=img,
                  height=args.height, width=args.width, num_frames=args.frames,
                  num_inference_steps=args.steps, guidance_scale=1.0,
                  guidance_scale_2=1.0, output_type="pil",
                  generator=torch.Generator("cpu").manual_seed(args.seed),
                  callback_on_step_end=_cb).frames[0]
    gen_s = time.time() - t0

    out = "out/lazy_baseline.mp4" if args.no_lazy else "out/lazy_swap.mp4"
    os.makedirs("out", exist_ok=True)
    export_to_video(frames, out, fps=16)
    log.info("")
    log.info("==== P2c RESULT (lazy=%s) ====", not args.no_lazy)
    log.info("peak RSS  : %.1f GiB", _rss_gb())
    log.info("peak VRAM : %.1f GiB", torch.cuda.max_memory_allocated() / 1024 ** 3)
    log.info("gen time  : %.0fs  (swap %.1fs)", gen_s, swap["secs"])
    log.info("wrote %s", out)
    log.info("Run once with --no-lazy for the both-experts baseline and compare the")
    log.info("two mp4s (must be identical) + peak RSS (~20 vs ~29 GiB).")


if __name__ == "__main__":
    main()
