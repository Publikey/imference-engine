"""LoRA check through the Engine: render without, with (twice), and without again.

    python validation/e2e_lora.py --backend B --lora PATH[:weight] [--lora ...] --tag T
        [--weights W] [--vram GiB] [--offload m] [--source IMG] [--prompt P] [--steps N]

Pass (``"ok": true`` in the ``LORA {json}`` line) when the LoRA changes the image,
the second request reuses the cached adapter with an identical result, and the
resident pipe renders the original image again once the LoRA is deactivated.
Backends with a build table only (flux2-klein, lumina2).
"""
import argparse
import importlib
import json
import logging
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True)
ap.add_argument("--weights", default="auto")
ap.add_argument("--lora", action="append", required=True)
ap.add_argument("--vram", type=float, default=None)
ap.add_argument("--offload", default=None)
ap.add_argument("--source", default=None)
ap.add_argument("--prompt", default="a red fox in a snowy forest, golden hour")
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="validation/renders")
a = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
for n in ("httpx", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

from imference_engine import Engine, RuntimeConfig  # noqa: E402 — after logging setup
from imference_engine.managers import builds as bmod  # noqa: E402

if a.vram is not None:
    bmod.cuda_vram_gb = lambda: a.vram
pkg = {"flux2-klein": "flux2klein", "lumina2": "lumina2"}[a.backend]
build = importlib.import_module(f"imference_engine.{pkg}.builds").recommend_build(bmod.cuda_vram_gb())
cfg = RuntimeConfig.from_env()
if a.offload:
    cfg.enable_offload, cfg.offload_mode = True, a.offload
elif a.weights == "auto":
    for k, v in build.runtime_kwargs().items():
        setattr(cfg, k, v)
eng = Engine(runtime=cfg).load()
eng.register_model("m", backend=a.backend, weights_path=a.weights)
kw = dict(model="m", seed=42, prompt=a.prompt)
if a.steps:
    kw["num_steps"] = a.steps
if a.source:
    from PIL import Image
    kw["source_image"] = Image.open(a.source).convert("RGB")
else:
    kw.update(width=1024, height=1024)
loras = []
for spec in a.lora:
    # "path" or "path:weight" — the weight is the part after the LAST colon,
    # and only when it is a number (a Windows path has its own "C:").
    path, _, w = spec.rpartition(":")
    try:
        weight = float(w) if path else 1.0
    except ValueError:
        path, weight = "", 1.0
    loras.append({"source": path or spec, "weight": weight})

out = f"{a.out}/{a.tag}"
os.makedirs(out, exist_ok=True)


def run(name, **extra):
    t = time.time()
    r = eng.generate(**kw, **extra)
    assert r.ok, r.errors
    r.images[0].save(f"{out}/{name}.png")
    return np.asarray(r.images[0].convert("RGB")).astype(np.float64), round(time.time() - t, 1)


def psnr(x, y):
    mse = ((x - y) ** 2).mean()
    return 99.0 if mse == 0 else round(float(10 * np.log10(255 ** 2 / mse)), 1)


base, t0 = run("0_base")
torch.cuda.reset_peak_memory_stats()
with_lora, t1 = run("1_lora", loras=loras)
with_lora2, t2 = run("2_lora_again", loras=loras)
after, t3 = run("3_base_again")
res = dict(tag=a.tag, build=build.name if a.weights == "auto" else None,
           offload=cfg.offload_mode if cfg.enable_offload else None,
           psnr_lora_vs_base=psnr(with_lora, base), psnr_lora_repeat=psnr(with_lora2, with_lora),
           psnr_base_restored=psnr(after, base), t_base=t0, t_lora_first=t1, t_lora_cached=t2,
           peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
res["ok"] = bool(res["psnr_lora_vs_base"] < 35 and res["psnr_lora_repeat"] > 40 and res["psnr_base_restored"] > 40)
json.dump(res, open(f"{out}/result.json", "w"))
print("LORA", json.dumps(res))
