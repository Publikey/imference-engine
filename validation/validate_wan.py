#!/usr/bin/env python
"""GPU validation harness for the Wan 2.2 VIDEO engine (separate from validate.py).

Wan is not one of the image `Engine` backends — it's its own `WanEngine`
(GGUF-MoE video, generate_video). This harness loads a builtin variant, renders a
short clip, exports an mp4 + a sample frame, and reports pass/fail.

Usage (GPU instance, after `pip install -e ".[wan,dev]"`):

    # list the builtin variants (no torch)
    python validation/validate_wan.py --list

    # text-to-video (the quick default: 4-step Lightning)
    python validation/validate_wan.py --variant wan22-t2v-lightning

    # image-to-video (needs an input image)
    python validation/validate_wan.py --variant wan22-i2v-lightning --image /path/in.png

    # lighter/faster smoke: fewer frames
    python validation/validate_wan.py --frames 33

    # load the shared base + GGUF experts from an R2/CDN mirror instead of HF
    # (WanRuntimeConfig.from_env picks up WAN_MODEL_CDN / WAN_MODEL_CACHE):
    WAN_MODEL_CDN=https://.../wan22 python validation/validate_wan.py --variant wan22-i2v-lightning --image in.png

Heavy: the A14B experts are ~15 GB GGUF each (×2 high/low) + the shared UMT5/VAE
(~11.5 GB). `WAN_PROFILE=auto` picks the GGUF quant from your VRAM/RAM; offload
keeps VRAM ≈ one expert (~17 GB). Exit code is non-zero on failure.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Allow running from a fresh clone without `pip install -e` (e.g. a quick --list).
sys.path.insert(0, str(HERE.parent))
DEFAULT_OUTDIR = HERE / "renders"
DEFAULT_PROMPT = (
    "a red fox trotting through a snowy forest at golden hour, "
    "cinematic, smooth camera motion"
)


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU validation harness for the Wan video engine")
    ap.add_argument("--variant", default="wan22-t2v-lightning",
                    help="builtin variant name (see --list)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--image", default=None, help="input image for i2v variants")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--profile", default="auto",
                    help="auto|gguf_q8|gguf_q6|gguf_q5|gguf_q4")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--list", action="store_true", help="list builtin variants and exit (no torch)")
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress engine INFO logs")
    args = ap.parse_args()

    # Surface engine INFO logs by default (CDN manifest, UMT5 re-tie, LoRA-applied,
    # and the DIAG lines) — this is a debug harness, so the build story is the point.
    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.list:
        from imference_engine.wan.presets import BUILTIN_VARIANTS
        print("Builtin Wan variants:")
        for name, v in sorted(BUILTIN_VARIANTS.items()):
            print(f"  {name:24s} mode={v.mode}  base={v.base_repo}")
        return 0

    from imference_engine.wan import WanEngine, WanRuntimeConfig

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    print(f"=== wan: {args.variant} ===", flush=True)
    t0 = time.time()
    try:
        # Build from the env contract so WAN_MODEL_CDN / WAN_MODEL_CACHE reach the
        # engine (shared UMT5/VAE base + GGUF experts then load from the R2 mirror,
        # not HuggingFace). --device / --profile override on top.
        cfg = WanRuntimeConfig.from_env()
        cfg.device = args.device
        cfg.memory_profile = args.profile
        engine = WanEngine(runtime=cfg).load()

        img = None
        if args.image:
            from PIL import Image
            img = Image.open(args.image)

        res = engine.generate_video(
            variant=args.variant,
            prompt=args.prompt,
            image=img,
            width=args.width,
            height=args.height,
            num_frames=args.frames,
            num_steps=args.steps,
            fps=args.fps,
            seed=args.seed,
        )
        secs = round(time.time() - t0, 1)

        if not res.ok:
            err = res.error.error if res.error else "no frames returned"
            print(f"  [FAIL] {secs}s  {err}", flush=True)
            return 1

        from diffusers.utils import export_to_video
        mp4 = Path(args.outdir) / f"wan_{args.variant}_seed{args.seed}.mp4"
        export_to_video(res.frames, str(mp4), fps=res.fps)
        frame0 = Path(args.outdir) / f"wan_{args.variant}_seed{args.seed}.png"
        res.frames[0].save(frame0)
        print(f"  [OK ] {secs}s  {mp4}  ({len(res.frames)} frames)  sample: {frame0}",
              flush=True)
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {round(time.time() - t0, 1)}s  {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
