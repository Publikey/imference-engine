#!/usr/bin/env python
"""GPU validation harness — load each engine's base model, render, report.

Runs ONE backend at a time in an isolated Engine (per-engine cpu-offload from the
config), so a failure in one engine never blocks the others — exactly the
"validate the pipelines one by one" workflow. For each engine it downloads (or
reuses) the base model from ``validation/base_models.yaml``, generates an image,
saves the PNG, and records pass/fail + timing + any traceback.

Usage (on a GPU instance, after `pip install -e ".[<engine>,dev]"`):

    # everything the config lists
    python validation/validate.py

    # just a couple, to a chosen dir
    python validation/validate.py --engines sdxl,flux --outdir /tmp/renders

    # point at weights you already downloaded (edit base_models.yaml:weights_local
    # or pass a local path per engine there); gated FLUX needs `huggingface-cli login`

    # check the plan without loading torch / downloading anything
    python validation/validate.py --dry-run

    # tight-disk box: delete each engine's downloaded weights right after its
    # render, so the disk never holds more than one model at a time
    python validation/validate.py --rm-weights

    # load base components from an R2/CDN mirror instead of HuggingFace
    # (RuntimeConfig.from_env picks these up). NOTE: only the shared BASE is on the
    # CDN — the per-model checkpoint is still fetched via hf_hub_download, so for a
    # strict HF_HUB_OFFLINE=1 run point base_models.yaml:weights_local at a cached
    # single-file (or run online for the checkpoint). Anima is the exception: its
    # whole repo is on the CDN, so it runs fully offline from R2.
    IMAGE_MODEL_CDN=https://.../image python validation/validate.py --engines anima

Exit code is non-zero if any selected engine failed, so it doubles as a CI gate.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "base_models.yaml"
DEFAULT_OUTDIR = HERE / "renders"


def load_config(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_weights(engine: str, entry: dict, cache_dir: Path) -> str:
    """Return the weights_path to register: a local override, a downloaded
    single-file .safetensors, or a repo id (Anima / repo mode)."""
    local = entry.get("weights_local")
    if local:
        return str(local)

    mode = entry.get("mode", "single_file")
    if mode == "repo":
        return entry["repo"]  # ModularPipeline.from_pretrained takes the repo id/dir

    if mode != "single_file":
        raise ValueError(f"{engine}: unknown mode {mode!r} (expected single_file|repo)")

    filename = entry.get("filename")
    if not filename:
        raise ValueError(f"{engine}: single_file mode needs a 'filename' (or set weights_local)")

    from huggingface_hub import hf_hub_download
    dest = cache_dir / entry["repo"].replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {entry['repo']}/{filename} ...", flush=True)
    return hf_hub_download(entry["repo"], filename, local_dir=str(dest))


def run_one(engine: str, entry: dict, args) -> dict:
    """Load one engine's base model, render, save. Returns a result record."""
    from imference_engine import Engine, RuntimeConfig

    rec: dict = {"engine": engine, "status": "fail", "seconds": None,
                 "image": None, "error": None}
    t0 = time.time()
    eng = None
    try:
        # Build from the env contract so IMAGE_MODEL_CDN / IMAGE_MODEL_CACHE reach
        # the engine (base components then load from the R2 mirror, not HF). The
        # per-entry offload and the --device flag override on top.
        cfg = RuntimeConfig.from_env()
        cfg.device = args.device
        cfg.enable_offload = bool(entry.get("offload", False))
        eng = Engine(runtime=cfg).load()

        weights = resolve_weights(engine, entry, Path(args.cache_dir))
        eng.register_model(
            engine, backend=engine, weights_path=weights,
            base_model=entry.get("base_model"),
        )

        gen_kwargs = dict(
            model=engine,
            prompt=entry["prompt"],
            width=entry.get("width", 1024),
            height=entry.get("height", 1024),
            num_steps=entry.get("num_steps", 28),
            guidance_scale=entry.get("guidance_scale", 6.0),
            seed=entry.get("seed", 42),
            batch=1,
        )
        if entry.get("negative_prompt") is not None:
            gen_kwargs["negative_prompt"] = entry["negative_prompt"]
        if entry.get("backend_options"):
            gen_kwargs["backend_options"] = entry["backend_options"]
        # Optional LoRA stack (supported backends only — SDXL today). Entries:
        # {source: <path|url>, weight: 0.8}. Not shipped in the default config
        # (no stable public URL to pin) — add ad hoc when validating LoRAs.
        if entry.get("loras"):
            gen_kwargs["loras"] = entry["loras"]

        result = eng.generate(**gen_kwargs)

        if result.errors:
            rec["error"] = "; ".join(e.error for e in result.errors)
        img = result.images[0] if result.images else None
        if img is not None:
            out = Path(args.outdir) / f"{engine}_seed{gen_kwargs['seed']}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out)
            rec.update(status="ok", image=str(out))
        elif not rec["error"]:
            rec["error"] = "no image returned"
    except Exception as e:  # noqa: BLE001 — one engine's failure must not stop the rest
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    finally:
        rec["seconds"] = round(time.time() - t0, 1)
        del eng
        gc.collect()
        _free_cuda()
        if getattr(args, "rm_weights", False):
            _cleanup_weights(engine, entry, args)
    return rec


def _free_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _cleanup_weights(engine: str, entry: dict, args) -> None:
    """Delete ONLY what the harness downloaded for this engine, so the disk never
    holds more than one model at a time. Never touches a user's ``weights_local``
    or the rendered PNGs — only the harness's own caches (reproducible)."""
    import shutil

    if entry.get("weights_local"):
        return  # user-owned file — never delete
    removed = []
    # 1. the single-file download under --cache-dir
    if entry.get("mode", "single_file") == "single_file":
        d = Path(args.cache_dir) / entry["repo"].replace("/", "__")
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))
    # 2. the shared base-components in the flat offline tree (the whales:
    #    T5 / Qwen2.5-VL / Z-Image encoder). base_model is None for sdxl/sd15
    #    (they use a tiny config repo) and for Anima (repo mode) — nothing to free.
    base = entry.get("base_model")
    if base:
        try:
            from imference_engine.runtime.offline import flat_root
            bd = Path(flat_root(None, namespace="image")) / base
            if bd.exists():
                shutil.rmtree(bd, ignore_errors=True)
                removed.append(str(bd))
        except Exception:  # noqa: BLE001
            pass
    if removed:
        print(f"  rm-weights: freed {' + '.join(removed)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU validation harness for imference-engine backends")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="base_models.yaml")
    ap.add_argument("--engines", default="", help="comma list (default: all in config)")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="where to save renders + report.json")
    ap.add_argument("--device", default="auto", help="auto|cuda|cuda:N|mps|cpu")
    ap.add_argument("--cache-dir", default=str(HERE / "weights"), help="download cache for base weights")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, load nothing")
    ap.add_argument("--rm-weights", action="store_true",
                    help="delete each engine's downloaded weights after its render so the "
                         "disk never holds more than one model at a time (never touches "
                         "weights_local or the rendered PNGs)")
    args = ap.parse_args()

    config = load_config(Path(args.config))
    selected = [e.strip() for e in args.engines.split(",") if e.strip()] or list(config)
    unknown = [e for e in selected if e not in config]
    if unknown:
        print(f"error: unknown engine(s) {unknown}; config has {list(config)}", file=sys.stderr)
        return 2

    print(f"Validating {len(selected)} engine(s): {', '.join(selected)}")
    if args.dry_run:
        for e in selected:
            entry = config[e]
            src = entry.get("weights_local") or (
                entry["repo"] if entry.get("mode") == "repo"
                else f"{entry['repo']}/{entry.get('filename')}")
            print(f"  {e:11s} <- {src}  (offload={entry.get('offload', False)})")
        return 0

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    records = []
    for e in selected:
        print(f"\n=== {e} ===", flush=True)
        rec = run_one(e, config[e], args)
        status = "OK " if rec["status"] == "ok" else "FAIL"
        print(f"  [{status}] {rec['seconds']}s  {rec.get('image') or rec.get('error')}", flush=True)
        records.append(rec)

    report = Path(args.outdir) / "report.json"
    report.write_text(json.dumps(records, indent=2), encoding="utf-8")

    ok = sum(r["status"] == "ok" for r in records)
    print("\n" + "=" * 60)
    print(f"Summary: {ok}/{len(records)} passed   (report: {report})")
    for r in records:
        mark = "✓" if r["status"] == "ok" else "✗"
        detail = r.get("image") if r["status"] == "ok" else r.get("error")
        print(f"  {mark} {r['engine']:11s} {str(r['seconds']) + 's':>7}  {detail}")

    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
