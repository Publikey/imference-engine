#!/usr/bin/env python
"""GPU validation harness for the MiniMax-H3 VIDEO engine (separate from
validate.py / validate_wan.py).

Loads the builtin variant, renders a short clip + its soundtrack, muxes an mp4,
exports a sample frame, and reports pass/fail. Passed e2e 2026-08-05 on the
PR #14355 head that became diffusers 0.40.0 (see
imference_engine/minimax_h3/README.md) — re-run once on the released 0.40.0
pin to confirm the backend on it.

Usage (GPU instance, dedicated venv: `pip install -e ".[minimax-h3,dev]"` —
pins diffusers==0.40.0):

    # text-to-video+audio (small fast canvas; native canvas ~2.3x slower/step)
    python validation/validate_h3.py

    # image-to-video (start keyframe; canvas follows the image aspect)
    python validation/validate_h3.py --image /path/in.png

    # native-quality canvas, full duration
    python validation/validate_h3.py --width 1344 --height 768 --frames 345

    # from a staged int8 mirror instead of HF (see stage_h3_int8.py)
    H3_MODEL_CDN=https://.../video python validation/validate_h3.py --repo imference/MiniMax-H3-int8

Heavy: ~50 GB (int8 mirror) to ~90 GB (bf16 FL2VA half) of weights, ~75 GB host
RAM under offload. Target hardware: 24 GB VRAM / 64+ GB RAM ("block" offload).
Exit code is non-zero on failure.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
DEFAULT_OUTDIR = HERE / "renders"
DEFAULT_PROMPT = (
    "a red fox trotting through a snowy pine forest at golden hour, "
    "snow crunching underfoot, distant birdsong, cinematic"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU validation harness for the MiniMax-H3 engine")
    ap.add_argument("--variant", default="minimax-h3")
    ap.add_argument("--repo", default=None,
                    help="override the variant's repo (e.g. a staged int8 mirror)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--image", default=None, help="start keyframe -> i2v")
    ap.add_argument("--last-image", default=None, help="end keyframe")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--profile", default="auto", help="auto|int8|bf16")
    ap.add_argument("--offload", default="auto", help="auto|block|leaf|none")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--frames", type=int, default=124, help="snapped to 17n+5; 5-15 s at 24 fps")
    ap.add_argument("--steps", type=int, default=None, help="default: the variant's recipe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--list", action="store_true", help="list builtin variants (no torch)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from imference_engine.minimax_h3 import (BUILTIN_VARIANTS, H3Variant,
                                             MiniMaxH3Engine,
                                             MiniMaxH3RuntimeConfig)
    if args.list:
        for name, v in BUILTIN_VARIANTS.items():
            print(f"{name}: repo={v.repo} num_steps={v.num_steps}")
        return 0

    runtime = MiniMaxH3RuntimeConfig.from_env()
    runtime.device = args.device
    runtime.memory_profile = args.profile
    runtime.offload_mode = args.offload

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.time()
        engine = MiniMaxH3Engine(runtime=runtime).load()
        if args.repo:
            engine.register_variant(H3Variant(name=args.variant, repo=args.repo))
        image = last_image = None
        if args.image or args.last_image:
            from PIL import Image
            image = Image.open(args.image) if args.image else None
            last_image = Image.open(args.last_image) if args.last_image else None

        res = engine.generate_video(
            prompt=args.prompt, variant=args.variant,
            image=image, last_image=last_image,
            width=args.width, height=args.height,
            num_frames=args.frames, num_steps=args.steps, seed=args.seed)
        dt = time.time() - t0

        if not res.ok:
            print(f"FAIL: {res.error.error if res.error else 'no frames'}")
            return 1
        assert res.audio is not None and res.sample_rate, "no soundtrack returned"
        assert res.fps == 24 and res.num_frames == len(res.frames), \
            f"metadata mismatch: fps={res.fps} num_frames={res.num_frames} got {len(res.frames)}"

        stem = outdir / f"h3_{'i2v' if (image or last_image) else 't2v'}_{args.seed}"
        res.frames[len(res.frames) // 2].save(f"{stem}_frame.png")
        try:
            import torch
            from diffusers.utils.export_utils import encode_video
            # MediaResult carries a portable numpy waveform; encode_video
            # torch.clip()s its audio, so hand it a tensor.
            encode_video(res.frames, fps=res.fps, output_path=f"{stem}.mp4",
                         audio=torch.from_numpy(res.audio),
                         audio_sample_rate=res.sample_rate)
            print(f"wrote {stem}.mp4 (with soundtrack) + {stem}_frame.png")
        except Exception as e:  # noqa: BLE001 — mux is not what we validate
            print(f"note: mp4 mux failed ({e}); frames + audio still validated")

        print(f"PASS: {len(res.frames)} frames @ {res.fps} fps "
              f"({res.width}x{res.height}), audio {res.audio.shape} @ {res.sample_rate} Hz, "
              f"seed {res.seeds[0]}, {dt:.0f}s total")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
