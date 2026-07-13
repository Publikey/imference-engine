#!/usr/bin/env python
"""Sweep guidance_scale for one engine — load the model ONCE, render the same
prompt/seed at several CFG values, save one PNG each for side-by-side comparison.

Used to pin an engine's default guidance (e.g. Chroma trends oversaturated at
high CFG). Reads the engine's repo/prompt/params from base_models.yaml, so it
stays consistent with the validation harness.

    python validation/sweep_cfg.py                       # chroma, 2.5/3.0/3.5/4.0
    python validation/sweep_cfg.py --engine chroma --values 2,2.5,3,3.5,4
    python validation/sweep_cfg.py --engine flux --values 2.5,3,3.5,4

Weights are downloaded once (reused if already present). Free disk first for the
big ones — Chroma is ~18 GB + base.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # run from a clone without install
sys.path.insert(0, str(HERE))          # import resolve_weights from validate.py


def main() -> int:
    from validate import load_config, resolve_weights

    ap = argparse.ArgumentParser(description="Sweep guidance_scale for one engine")
    ap.add_argument("--engine", default="chroma")
    ap.add_argument("--values", default="2.5,3.0,3.5,4.0",
                    help="comma list of guidance_scale values")
    ap.add_argument("--config", default=str(HERE / "base_models.yaml"))
    ap.add_argument("--outdir", default=str(HERE / "renders"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=str(HERE / "weights"))
    args = ap.parse_args()

    config = load_config(Path(args.config))
    if args.engine not in config:
        print(f"error: unknown engine {args.engine!r}; have {list(config)}", file=sys.stderr)
        return 2
    entry = config[args.engine]
    values = [float(v) for v in args.values.split(",") if v.strip()]

    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(
        device=args.device, enable_offload=bool(entry.get("offload", False)))).load()
    weights = resolve_weights(args.engine, entry, Path(args.cache_dir))
    engine.register_model(args.engine, backend=args.engine, weights_path=weights,
                          base_model=entry.get("base_model"))

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    seed = entry.get("seed", 42)
    gen = dict(
        model=args.engine, prompt=entry["prompt"],
        width=entry.get("width", 1024), height=entry.get("height", 1024),
        num_steps=entry.get("num_steps", 28), seed=seed, batch=1,
    )
    if entry.get("negative_prompt") is not None:
        gen["negative_prompt"] = entry["negative_prompt"]
    if entry.get("backend_options"):
        gen["backend_options"] = entry["backend_options"]

    print(f"Sweeping {args.engine} guidance_scale over {values} "
          f"(same prompt/seed={seed}) — model loads once.")
    for cfg in values:
        result = engine.generate(guidance_scale=cfg, **gen)
        if result.errors:
            print(f"  cfg={cfg}: FAIL {result.errors[0].error}")
            continue
        out = Path(args.outdir) / f"{args.engine}_cfg{cfg}_seed{seed}.png"
        result.images[0].save(out)
        print(f"  cfg={cfg}: {out}")

    print("\nCompare the PNGs and pick the CFG with the truest colours; "
          "tell me the value and I'll bake it as the engine default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
