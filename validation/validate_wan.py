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
    ap.add_argument("--flow-shift", type=float, default=None,
                    help="override the variant's flow_shift (i2v often wants 5.0 vs t2v 3.0)")
    ap.add_argument("--guidance", type=float, default=None, help="guidance_scale (high-noise expert)")
    ap.add_argument("--guidance2", type=float, default=None, help="guidance_scale_2 (low-noise expert)")
    ap.add_argument("--no-lora", action="store_true",
                    help="drop the variant's Lightning LoRA (test the base experts; pair with "
                         "--steps 20 --guidance 3.5 for the non-distilled recipe)")
    ap.add_argument("--gguf-repo", default=None,
                    help="override the variant's gguf_repo (test another GGUF provider without "
                         "touching presets, e.g. bullerwins/Wan2.2-I2V-A14B-GGUF)")
    ap.add_argument("--gguf-high-template", default=None,
                    help="override gguf_high_template ({quant} placeholder; a literal filename "
                         "with no {quant} also works)")
    ap.add_argument("--gguf-low-template", default=None, help="override gguf_low_template")
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
        # --profile is a plain string; "auto" stays a string (engine resolves it
        # from VRAM/RAM at load), a concrete quant must become the MemoryProfile
        # enum (a bare "gguf_q6" string is rejected by _setup).
        if args.profile == "auto":
            cfg.memory_profile = "auto"
        else:
            from imference_engine.wan import MemoryProfile
            cfg.memory_profile = MemoryProfile(args.profile)
        engine = WanEngine(runtime=cfg).load()

        img = None
        if args.image:
            from PIL import Image
            img = Image.open(args.image)

        # --flow-shift: re-register the builtin variant with an overridden shift so
        # we can sweep it without editing presets.py (register_variant replaces by
        # name). Cheap: only the WanVariant recipe changes, not the cached weights.
        _wan_overrides = (args.flow_shift is not None or args.no_lora
                          or args.gguf_repo or args.gguf_high_template
                          or args.gguf_low_template)
        if _wan_overrides:
            import dataclasses

            from imference_engine.wan.presets import BUILTIN_VARIANTS
            base_v = BUILTIN_VARIANTS.get(args.variant)
            if base_v is not None:
                overrides: dict = {}
                if args.flow_shift is not None:
                    overrides["flow_shift"] = args.flow_shift
                if args.no_lora:
                    overrides["loras"] = []  # base experts, no Lightning distillation
                if args.gguf_repo:
                    overrides["gguf_repo"] = args.gguf_repo
                # override the templates and clear explicit names so the template wins
                if args.gguf_high_template:
                    overrides["gguf_high_template"] = args.gguf_high_template
                    overrides["gguf_high_name"] = None
                if args.gguf_low_template:
                    overrides["gguf_low_template"] = args.gguf_low_template
                    overrides["gguf_low_name"] = None
                engine.register_variant(dataclasses.replace(base_v, **overrides))
                print(f"  variant override: {overrides}", flush=True)

        gv = dict(
            variant=args.variant, prompt=args.prompt, image=img,
            width=args.width, height=args.height, num_frames=args.frames,
            num_steps=args.steps, fps=args.fps, seed=args.seed,
        )
        if args.guidance is not None:
            gv["guidance_scale"] = args.guidance
        if args.guidance2 is not None:
            gv["guidance_scale_2"] = args.guidance2
        res = engine.generate_video(**gv)
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
