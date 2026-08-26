#!/usr/bin/env python
"""Qwen-Image validation for 24-32 GB cards — engine load path + group offload.

The image Engine's only offload mode is ``enable_model_cpu_offload``
(whole-module), which needs the full ~41 GB bf16 Qwen-Image transformer
on-GPU during the denoise — OOM on anything under ~48 GB. This harness reuses
the ENGINE'S OWN load path (``QwenImageBackend``: from_single_file transformer
+ base components, exactly what validate.py exercises) and only swaps the
memory placement for diffusers group offloading (block_level + stream, the
same recipe as the MiniMax-H3 loader). It therefore validates the qwenimage
load + render path on the pinned stack, NOT the ModelManager residency
(hardware-gated to ≥48 GB cards — validate.py covers it there).

Measured 2026-08-26 (RTX PRO 4500 Blackwell 32 GB, diffusers 0.40.0): 40 steps
true-CFG 1024x1024 in ~250 s, peak 4.7 GB VRAM.

    python validation/validate_qwen_offload.py \
        --weights validation/weights/Comfy-Org__Qwen-Image_ComfyUI/split_files/diffusion_models/qwen_image_bf16.safetensors
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen-Image group-offload validation (24-32 GB cards)")
    ap.add_argument("--weights", required=True, help="qwen_image bf16 single-file .safetensors")
    ap.add_argument("--base-model", default="Qwen/Qwen-Image")
    ap.add_argument("--out", default=str(HERE / "renders_qwen_offload" / "qwenimage_seed42.png"))
    ap.add_argument("--prompt", default='a bookstore window with a sign that reads "Imference Engine"')
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.0, help="mapped to true_cfg_scale")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from diffusers.hooks import apply_group_offloading

    from imference_engine.qwenimage.backend import QwenImageBackend

    backend = QwenImageBackend()
    t0 = time.time()
    pipe = backend.load_pipeline(local_path=args.weights, base_model=args.base_model)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    offload = dict(onload_device=torch.device("cuda"),
                   offload_device=torch.device("cpu"), use_stream=True)
    pipe.transformer.enable_group_offload(
        offload_type="block_level", num_blocks_per_group=1, **offload)
    apply_group_offloading(pipe.text_encoder, offload_type="leaf_level", **offload)
    pipe.vae.to("cuda")
    print("group offload wired (transformer block_level, text_encoder leaf_level)", flush=True)

    gen = backend.make_generator(args.seed, "cuda")
    t0 = time.time()
    result = pipe(
        prompt=args.prompt,
        negative_prompt=" ",
        true_cfg_scale=args.guidance,
        num_inference_steps=args.steps,
        width=1024, height=1024,
        generator=gen,
    )
    dt = time.time() - t0
    img = result.images[0]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img.save(args.out)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[OK ] {dt:.1f}s  {args.out}  peak VRAM {peak:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
