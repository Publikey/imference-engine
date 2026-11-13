"""Engine-level run of ONE image backend — the harness used to validate the
backends added in November 2026 (one configuration per process, so every run
starts from a cold Engine and its peak-VRAM reading is its own).

    python validation/e2e_backend.py --backend B --weights W --tag T
        [--base REPO] [--offload none|model|group] [--vram GiB]
        [--steps N --cfg G --neg TEXT --prompt TEXT --size WxH --batch N]
        [--opt key=value ...] [--source IMG] [--out DIR] [--runs 2]

- ``--weights auto`` + ``--vram`` exercise a backend's build table: ``--vram``
  overrides the detected VRAM (GiB), so every tier can be run on one card. The
  build's offload mode is applied to the RuntimeConfig unless ``--offload`` is given.
- ``--opt`` passes ``backend_options`` (JSON values: ``--opt layers=3``).
- ``--source`` makes it an img2img / edit request.
- The environment contract applies (``IMAGE_MODEL_CDN``, ``IMAGE_MODEL_CACHE``,
  ``HF_HUB_OFFLINE=1`` for a cold load that must never touch HuggingFace).

Prints one ``E2E {json}`` line: build, offload mode, cold-load seconds, seconds per
call (best of the warm runs), peak allocated / reserved VRAM, output size. Images
(and layers, for backends that return some) land in ``<out>/<tag>/``.
"""
import argparse
import importlib
import json
import logging
import os
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True)
ap.add_argument("--weights", default="auto")
ap.add_argument("--base", default=None)
ap.add_argument("--offload", default=None)
ap.add_argument("--vram", type=float, default=None)
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--cfg", type=float, default=None)
ap.add_argument("--neg", default=None)
ap.add_argument("--prompt", default="a photo of a red fox in a snowy forest, golden hour, sharp focus, 85mm")
ap.add_argument("--size", default="1024x1024")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--opt", action="append", default=[])
ap.add_argument("--source", default=None)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default="validation/renders")
ap.add_argument("--runs", type=int, default=2)
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
for n in ("httpx", "urllib3", "filelock"):
    logging.getLogger(n).setLevel(logging.WARNING)

from imference_engine import Engine, RuntimeConfig  # noqa: E402 — after logging setup

PKG = {"flux2-klein": "flux2klein", "ernie-image": "ernieimage", "lumina2": "lumina2",
       "qwenimage": "qwenimage", "qwenimage-layered": "qwenimage", "longcat": "longcat",
       "longcat-edit": "longcat", "ovis-image": "ovisimage", "kandinsky5": "kandinsky5",
       "kandinsky5-edit": "kandinsky5", "hidream": "hidream", "joyimage-edit": "joyimage"}
build = None
cfg = RuntimeConfig.from_env()
pkg = PKG.get(a.backend)
if pkg:
    try:
        mod = importlib.import_module(f"imference_engine.{pkg}.builds")
        bmod = importlib.import_module("imference_engine.managers.builds")
        if a.vram is not None:
            bmod.cuda_vram_gb = lambda: a.vram
        fn = getattr(mod, "recommend_layered_build" if a.backend.endswith("layered") else "recommend_build")
        build = fn(bmod.cuda_vram_gb())
    except (ImportError, AttributeError):
        build = None
if a.offload == "none":
    cfg.enable_offload = False            # force everything resident
elif a.offload:
    cfg.enable_offload, cfg.offload_mode = True, a.offload
elif build is not None and a.weights == "auto":
    for k, v in build.runtime_kwargs().items():
        setattr(cfg, k, v)

out = f"{a.out}/{a.tag}"
os.makedirs(out, exist_ok=True)
t0 = time.time()
eng = Engine(runtime=cfg).load()
eng.register_model("m", backend=a.backend, weights_path=a.weights, base_model=a.base)
w, h = (int(x) for x in a.size.split("x"))
kw = dict(model="m", seed=42, batch=a.batch, prompt=a.prompt)
if a.steps:
    kw["num_steps"] = a.steps
if a.cfg is not None:
    kw["guidance_scale"] = a.cfg
if a.neg:
    kw["negative_prompt"] = a.neg
opts = {}
for o in a.opt:
    k, v = o.split("=", 1)
    try:
        v = json.loads(v)
    except ValueError:
        pass
    opts[k] = v
if opts:
    kw["backend_options"] = opts
if a.source:
    from PIL import Image
    kw["source_image"] = Image.open(a.source).convert("RGB")
else:
    kw.update(width=w, height=h)

times = []
for i in range(a.runs):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    r = eng.generate(**kw)
    times.append(time.time() - t)
    if i == 0:
        first = time.time() - t0
    assert r.ok, r.errors
for j, img in enumerate(r.images):
    img.save(f"{out}/{j}.png")
layers = getattr(r, "layers", None)
if layers:
    for j, group in enumerate(layers):
        for k, im in enumerate(group):
            im.save(f"{out}/{j}_layer{k}.png")
pipe = eng._models._gpu["m"]
te = getattr(pipe, "text_encoder", None)
res = dict(tag=a.tag, build=build.name if (build is not None and a.weights == "auto") else None,
           offload=(cfg.offload_mode if cfg.enable_offload else None),
           cold_s=round(first, 1), s_per_call=round(min(times[1:] or times), 2),
           peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
           reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
           size=r.images[0].size, n=len(r.images), layers=[len(g) for g in layers] if layers else None,
           pipe=type(pipe).__name__, te=str(te.dtype) if te is not None else "released")
json.dump(res, open(f"{out}/result.json", "w"))
print("E2E", json.dumps(res))
