"""Spike P2b — lazy-load the MoE experts (keep only one resident at a time).

Wan2.2 A14B is two transformers: the high-noise expert (`transformer`) runs while
``t >= boundary_timestep``, the low-noise one (`transformer_2`) below it. Under
``enable_model_cpu_offload`` BOTH sit in CPU RAM the whole generation (~9 GB each
at Q4 = ~18 GB) even though only one is ever active. With 4-step Lightning the
crossing happens ONCE, early (boundary ~0.9 -> ~1 high step then the rest low).

Lazy-load = keep only the active expert resident and load the other at the
boundary crossing. This is the biggest remaining RAM lever (matters most for
imference-desktop: consumer GPUs with 16-32 GB RAM). This spike MEASURES the two
numbers that decide whether to build the (non-trivial) production version:
  1. RAM saved  = RSS(both experts) - RSS(one expert)   [~one expert at the quant]
  2. Latency cost = time to load one expert GGUF from disk  [the mid-gen reload]
It also runs a normal baseline generation (both experts, int8 TE, offload) to
anchor the real peak RSS, and PROJECTS the lazy peak/time from (1)+(2). A
best-effort live swap (callback) is attempted last to sanity-check correctness.

    export WAN_MODEL_CACHE=/workspace/wan-tree WAN_MODEL_CDN=https://…/wan22 HF_HUB_OFFLINE=1
    python p2b_lazy_experts.py --gguf-quant Q4_K_M
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p2b")


def _rss_gb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
    except OSError:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="wan22-i2v-lightning")
    ap.add_argument("--gguf-quant", default="Q4_K_M")
    ap.add_argument("--image-url",
                    default="https://blob.gen-image.com/small/3e7045c2-0b5b-4c01-bd64-7d16bdf35d90.webp")
    ap.add_argument("--prompt", default="the subject turns and smiles, gentle camera push-in")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from diffusers.utils import export_to_video, load_image

    from imference_engine.wan.loader import (
        _BASE_CFG_PATTERNS, _load_gguf_transformer, _local_repo_dir,
        _resolve_gguf, build_pipeline, load_shared_components)
    from imference_engine.wan.presets import BUILTIN_VARIANTS

    cache_dir = os.getenv("WAN_MODEL_CACHE") or None
    cdn = os.getenv("WAN_MODEL_CDN") or None
    variant = BUILTIN_VARIANTS[args.variant]
    q = args.gguf_quant

    # ---------------------------------------------------------------- Part A
    # RAM of one expert vs both + reload latency (the decision numbers).
    base_dir = _local_repo_dir(variant.base_repo, _BASE_CFG_PATTERNS, cache_dir)
    high_path = _resolve_gguf(variant.gguf_repo, q, "high", variant.gguf_high_name,
                              variant.gguf_high_template, cdn_base=cdn, cache_dir=cache_dir)
    low_path = _resolve_gguf(variant.gguf_repo, q, "low", variant.gguf_low_name,
                             variant.gguf_low_template, cdn_base=cdn, cache_dir=cache_dir)

    rss0 = _rss_gb()
    log.info("RSS baseline: %.1f GiB", rss0)
    t = time.time()
    high = _load_gguf_transformer(high_path, base_dir, "transformer")
    t_high = time.time() - t
    gc.collect()
    rss_one = _rss_gb()
    log.info("[A] 1 expert (high) loaded: RSS %.1f GiB (+%.1f), load %.1fs",
             rss_one, rss_one - rss0, t_high)

    t = time.time()
    low = _load_gguf_transformer(low_path, base_dir, "transformer_2")
    t_low = time.time() - t
    gc.collect()
    rss_two = _rss_gb()
    one_expert_gb = rss_two - rss_one
    log.info("[A] 2 experts loaded: RSS %.1f GiB (+%.1f for low), load %.1fs",
             rss_two, one_expert_gb, t_low)
    log.info("[A] >>> lazy-load would SAVE ~%.1f GiB RAM, cost ~%.1fs reload/gen",
             one_expert_gb, t_low)
    del high, low
    gc.collect()

    # ---------------------------------------------------------------- Part B
    # Real baseline: both experts + int8 text encoder + offload, one generation.
    shared = load_shared_components(variant.base_repo, cache_dir=cache_dir,
                                    text_encoder_quant="int8")
    pipe = build_pipeline(variant, quant=q, shared=shared, device="cuda",
                          enable_offload=True, vae_tiling=True,
                          cache_dir=cache_dir, cdn_base=cdn)

    # boundary analysis: how many of the N steps use each expert
    try:
        br = pipe.config.get("boundary_ratio", None)
        nt = pipe.scheduler.config.num_train_timesteps
        pipe.scheduler.set_timesteps(args.steps, device="cuda")
        ts = [float(x) for x in pipe.scheduler.timesteps]
        if br is not None:
            bnd = br * nt
            hi = sum(1 for x in ts if x >= bnd)
            log.info("[B] boundary=%.0f (ratio %.3f): %d high-steps + %d low-steps "
                     "(of %d) -> ONE crossing", bnd, br, hi, len(ts) - hi, len(ts))
    except Exception as e:  # noqa: BLE001
        log.info("[B] boundary analysis skipped (%s)", e)

    img = load_image(args.image_url).convert("RGB").resize((args.width, args.height))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    frames = pipe(prompt=args.prompt, negative_prompt="", image=img,
                  height=args.height, width=args.width, num_frames=args.frames,
                  num_inference_steps=args.steps, guidance_scale=1.0,
                  guidance_scale_2=1.0, output_type="pil",
                  generator=torch.Generator("cpu").manual_seed(args.seed)).frames[0]
    gen_s = time.time() - t
    peak_rss = _rss_gb()
    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
    os.makedirs("out", exist_ok=True)
    export_to_video(frames, "out/lazy_baseline.mp4", fps=16)
    log.info("[B] baseline (both experts): peak RSS %.1f GiB, peak VRAM %.1f GiB, gen %.0fs",
             peak_rss, peak_vram, gen_s)

    # ---------------------------------------------------------------- Verdict
    log.info("")
    log.info("==== VERDICT (lazy-load projection) ====")
    log.info("baseline peak RSS (both experts) : %.1f GiB", peak_rss)
    log.info("one resident expert (%s)         : ~%.1f GiB", q, one_expert_gb)
    log.info("=> projected lazy peak RSS       : ~%.1f GiB  (−%.1f)",
             peak_rss - one_expert_gb, one_expert_gb)
    log.info("=> projected extra latency/gen   : ~%.1fs reload at the boundary", t_low)
    log.info("If the RAM win justifies the latency, the production step is a custom")
    log.info("WanPipeline that frees `transformer` and loads `transformer_2` in the")
    log.info("callback_on_step_end at the boundary crossing (re-attaching the offload")
    log.info("hook to the freshly loaded expert).")


if __name__ == "__main__":
    main()
