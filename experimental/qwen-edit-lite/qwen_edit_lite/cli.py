"""qwen-edit-lite command line.

    qwen-edit-lite doctor                       # GPU / RAM / libs, and the preset to start with
    qwen-edit-lite presets                      # built-in presets
    qwen-edit-lite files                        # GGUF quants + Lightning files on the Hub
    qwen-edit-lite edit --preset q6-l4 --image in.png --instruction "..." [--ref r.png] [-o out.png]
    qwen-edit-lite bench --image in.png [--presets q6-l4,q5-l4,...] [--out bench/]
    qwen-edit-lite bench --cases cases.yaml --presets auto
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path


def recommend(vram_gb: float, ram_gb: float, *, nvidia: bool = True) -> list:
    """Ordered presets to try on this machine (lightest-safe first by default)."""
    if vram_gb <= 0:
        return []
    if ram_gb >= 60 and vram_gb >= 20:
        first = ["bf16-l4", "q8-l4", "q6-l4"]
    elif ram_gb >= 30 and vram_gb >= 14:
        first = ["q6-l4", "q5-l4", "q8-l4"]
    elif ram_gb >= 22 and vram_gb >= 10:
        first = ["q5-l4", "q4-l4", "q5-l4-disk"]
    elif ram_gb >= 22:
        first = ["q4-l4", "q5-l4-disk"]
    else:
        first = ["q5-l4-disk", "q4-l4"]
    return first


def _machine() -> dict:
    info: dict = {"python": platform.python_version(), "os": platform.platform()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["ram_free_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
    except ImportError:
        info["ram_gb"] = None
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        info["rocm"] = bool(getattr(torch.version, "hip", None))
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(p.total_memory / 1e9, 1)
            info["compute_capability"] = f"{p.major}.{p.minor}"
    except ImportError:
        info["torch"] = None
    for mod in ("diffusers", "transformers", "accelerate", "peft", "gguf", "bitsandbytes", "torchao", "nunchaku", "lpips"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = None
    return info


def cmd_doctor(a) -> int:
    info = _machine()
    print(json.dumps(info, indent=2))
    problems = []
    if not info.get("cuda"):
        problems.append("No CUDA/ROCm GPU visible to torch: these presets need one.")
    if info.get("diffusers") and not str(info["diffusers"]).startswith("0.4"):
        problems.append(f"diffusers {info['diffusers']}: GGUF loading of Qwen-Image needs >= 0.37 (tested target 0.40.0).")
    if not info.get("gguf"):
        problems.append("gguf missing: pip install 'qwen-edit-lite[gguf]' for the gguf:* presets.")
    if not info.get("bitsandbytes"):
        problems.append("bitsandbytes missing: pip install 'qwen-edit-lite[nf4]' for the NF4 encoder (all lite presets).")
    cc = info.get("compute_capability")
    if cc and float(cc) < 8.9:
        problems.append(f"compute capability {cc}: fp8-torchao needs >= 8.9 (fp8-layerwise works everywhere).")
    for p in problems:
        print("! " + p)
    rec = recommend(info.get("vram_gb") or 0, info.get("ram_gb") or 0)
    if rec:
        print(f"\nStart with: qwen-edit-lite bench --image <photo> --presets {','.join(rec)}")
    return 0


def cmd_presets(a) -> int:
    from qwen_edit_lite.presets import BUILTIN
    for p in BUILTIN.values():
        print(f"{p.name:12} transformer={p.transformer:14} encoder={p.encoder:5} lightning={p.lightning:5} "
              f"offload={p.offload:11} steps={p.effective_steps:<3} cfg={p.effective_cfg}\n             {p.description}")
    return 0


def cmd_files(a) -> int:
    from qwen_edit_lite import hub
    for repo in (hub.GGUF_REPO, hub.LIGHTNING_REPO):
        print(f"== {repo}")
        try:
            for f in hub.list_files(repo):
                if f.endswith((".gguf", ".safetensors")):
                    print("  " + f)
        except Exception as e:
            print(f"  ! {type(e).__name__}: {e}")
    return 0


def _preset_from_args(a):
    from qwen_edit_lite.presets import get
    p = get(a.preset)
    return p.with_overrides(transformer=a.transformer, encoder=a.encoder, lightning=a.lightning, offload=a.offload,
                            steps=a.steps, cfg=a.cfg,
                            unload_encoder=(False if a.keep_encoder else None),
                            fuse_lora=(False if a.no_fuse else None))


def cmd_edit(a) -> int:
    from PIL import Image

    from qwen_edit_lite.engine import Engine
    from qwen_edit_lite.metrics import Meter

    preset = _preset_from_args(a)
    if a.dry_run:
        print(json.dumps(preset.to_dict(), indent=2))
        return 0
    meter = Meter().start()
    eng = Engine(preset, meter=meter)
    try:
        out = eng.edit(Image.open(a.image), a.instruction, references=[Image.open(r) for r in a.ref or []],
                       seed=a.seed)
    finally:
        meter.stop()
    out.save(a.output)
    print(json.dumps({"output": a.output, "preset": preset.to_dict(), "memory": meter.to_dict(), "notes": eng.notes},
                     indent=2, default=str))
    return 0


def cmd_bench(a) -> int:
    from qwen_edit_lite import bench
    from qwen_edit_lite.presets import from_dict, get

    names = a.presets
    if names == "auto":
        info = _machine()
        names = ",".join(recommend(info.get("vram_gb") or 0, info.get("ram_gb") or 0))
        if not names:
            raise SystemExit("no GPU detected")
        print(f"auto presets: {names}")
    presets = []
    for n in names.split(","):
        n = n.strip()
        if n.endswith((".yaml", ".yml", ".json")):
            import yaml
            for d in yaml.safe_load(Path(n).read_text()):
                presets.append(from_dict(d))
        else:
            presets.append(get(n))
    cases = bench.load_cases(a.cases, a.image)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in presets:
        print(f"\n=== {p.name}: {p.description}", flush=True)
        fields = {k: v for k, v in p.to_dict().items() if k in p.__dataclass_fields__}
        row = bench.run_preset_subprocess(fields, cases, out, a.seed, a.timeout)
        rows.append(row)
        status = row.get("error") or f"ok, {len(row.get('cases', []))} edits"
        mem = row.get("memory", {})
        print(f"--- {p.name}: {status} | VRAM {mem.get('vram_peak_gb')} GB | RAM {mem.get('ram_peak_gb')} GB", flush=True)
        (out / "results.json").write_text(json.dumps(rows, indent=2, default=str))
    reference = a.reference or next((n for n in ("bf16-l4", "q8-l4") if n in [p.name for p in presets]), presets[0].name)
    print("\n" + bench.summarize(rows, out, reference))
    print(f"\nResults: {out / 'results.md'} (+ results.json, grid-*.png)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qwen-edit-lite", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    sub.add_parser("presets")
    sub.add_parser("files")

    def preset_args(p):
        p.add_argument("--preset", default="q6-l4")
        p.add_argument("--transformer")
        p.add_argument("--encoder", choices=["bf16", "nf4", "int8"])
        p.add_argument("--lightning")
        p.add_argument("--offload")
        p.add_argument("--steps", type=int)
        p.add_argument("--cfg", type=float)
        p.add_argument("--keep-encoder", action="store_true", help="keep the encoder loaded between edits")
        p.add_argument("--no-fuse", action="store_true", help="keep the Lightning LoRA as an adapter")

    e = sub.add_parser("edit")
    preset_args(e)
    e.add_argument("--image", required=True)
    e.add_argument("--instruction", required=True)
    e.add_argument("--ref", action="append", help="reference image (image 2, image 3)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("-o", "--output", default="edit.png")
    e.add_argument("--dry-run", action="store_true")

    b = sub.add_parser("bench")
    b.add_argument("--presets", default="auto", help="comma list of preset names / presets.yaml, or 'auto'")
    b.add_argument("--image")
    b.add_argument("--cases")
    b.add_argument("--out", default="bench")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--reference", help="preset used as the PSNR/LPIPS reference")
    b.add_argument("--timeout", type=float, default=3600)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return {"doctor": cmd_doctor, "presets": cmd_presets, "files": cmd_files, "edit": cmd_edit,
            "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
