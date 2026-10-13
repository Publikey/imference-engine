"""Benchmark: run presets x cases, one fresh process per preset.

Each preset runs in its OWN subprocess so the peak-RAM / peak-VRAM numbers are
not polluted by the previous preset (and an OOM kills one row, not the run).
Outputs under ``<out>/``:

    <preset>/<case>.png       edited images
    results.json              every measurement + notes (errors, LoRA mode, ...)
    results.md                comparison table (+ PSNR / LPIPS vs the reference)
    grid-<case>.png           side by side, for eyeballing
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

DEFAULT_CASES = [
    {"id": "sky", "instruction": "Replace the sky with a dramatic orange sunset, keep everything else unchanged."},
    {"id": "text", "instruction": "Change the text on the sign to 'OPEN 24/7' in the same font and color."},
    {"id": "remove", "instruction": "Remove the person on the left and fill the background naturally."},
    {"id": "style", "instruction": "Turn this photo into a watercolor painting."},
]


def load_cases(path: Optional[str], image: Optional[str]) -> list:
    """cases.yaml: [{id, image, instruction, references?: [...], seed?}]"""
    if path:
        import yaml
        cases = yaml.safe_load(Path(path).read_text())
        base = Path(path).parent
        for c in cases:
            c["image"] = str((base / c["image"]).resolve()) if not Path(c["image"]).is_absolute() else c["image"]
            c["references"] = [str((base / r).resolve()) if not Path(r).is_absolute() else r
                               for r in c.get("references", [])]
        return cases
    if not image:
        raise SystemExit("give --cases cases.yaml or --image <file> (the default instructions are applied to it)")
    return [dict(c, image=image, references=[]) for c in DEFAULT_CASES]


def run_preset_inprocess(preset_dict: dict, cases: list, out: Path, seed: int) -> dict:
    """Body of one preset run (executed in a child process)."""
    from PIL import Image

    from qwen_edit_lite.engine import Engine
    from qwen_edit_lite.metrics import Meter
    from qwen_edit_lite.presets import from_dict

    preset = from_dict(preset_dict)
    meter = Meter().start()
    eng = Engine(preset, meter=meter)
    row: dict = {"preset": preset.to_dict(), "cases": [], "error": None}
    t0 = time.perf_counter()
    try:
        eng.load()
        for c in cases:
            img = Image.open(c["image"])
            refs = [Image.open(r) for r in c.get("references", [])]
            t = time.perf_counter()
            res = eng.edit(img, c["instruction"], references=refs, seed=c.get("seed", seed))
            dest = out / preset.name / f"{c['id']}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            res.save(dest)
            row["cases"].append({"id": c["id"], "seconds": round(time.perf_counter() - t, 2), "path": str(dest)})
    except Exception as e:  # noqa: BLE001 — a failed preset is a result
        import traceback
        row["error"] = f"{type(e).__name__}: {e}"[:1000]
        row["traceback"] = traceback.format_exc()[-4000:]
        msg = str(e).lower()
        row["oom"] = "out of memory" in msg or isinstance(e, MemoryError)
    finally:
        meter.stop()
        eng.close()
    row["total_seconds"] = round(time.perf_counter() - t0, 1)
    row["memory"] = meter.to_dict()
    row["notes"] = eng.notes
    return row


def run_preset_subprocess(preset_dict: dict, cases: list, out: Path, seed: int, timeout: float) -> dict:
    job = out / f".job-{preset_dict['name']}.json"
    job.write_text(json.dumps({"preset": preset_dict, "cases": cases, "out": str(out), "seed": seed}))
    res_path = out / f".res-{preset_dict['name']}.json"
    cmd = [sys.executable, "-m", "qwen_edit_lite.bench", str(job), str(res_path)]
    try:
        proc = subprocess.run(cmd, timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        return {"preset": preset_dict, "error": f"timeout after {timeout:.0f}s", "cases": []}
    if res_path.is_file():
        return json.loads(res_path.read_text())
    # Killed (Linux OOM killer: -9) before writing a result.
    return {"preset": preset_dict, "cases": [], "oom": code in (-9, 137),
            "error": f"process exited with code {code} (killed: likely out of host RAM)" if code in (-9, 137)
            else f"process exited with code {code}"}


def summarize(rows: list, out: Path, reference: Optional[str]) -> str:
    from PIL import Image

    from qwen_edit_lite.quality import lpips, psnr

    ref = next((r for r in rows if r["preset"]["name"] == reference), None)
    ref_paths = {c["id"]: c["path"] for c in (ref or {}).get("cases", [])}
    lines = ["| preset | transformer | encoder | steps | cfg | offload | load s | s/edit | VRAM peak GB | RAM peak GB | "
             f"PSNR vs {reference or '-'} | LPIPS | notes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r["preset"]
        mem = r.get("memory", {})
        load_s = sum(s["seconds"] for s in mem.get("stages", []) if s["name"].startswith(("load", "setup")))
        per = [c["seconds"] for c in r.get("cases", [])]
        ps, lp = [], []
        for c in r.get("cases", []):
            if c["id"] in ref_paths and r is not ref:
                a, b = Image.open(ref_paths[c["id"]]), Image.open(c["path"])
                ps.append(psnr(a, b))
                v = lpips(a, b)
                if v is not None:
                    lp.append(v)
        notes = r.get("error") or ", ".join(f"{k}={v}" for k, v in (r.get("notes") or {}).items()
                                            if k in ("lora", "offload_fallback", "group_offload_error", "lora_fuse_error"))
        fmt = lambda xs, n=2: f"{sum(xs) / len(xs):.{n}f}" if xs else "-"  # noqa: E731
        lines.append(f"| {p['name']} | {p['transformer']} | {p['encoder']} | {p['effective_steps']} | {p['effective_cfg']} | "
                     f"{p['offload']} | {load_s:.0f} | {fmt(per, 1)} | {mem.get('vram_peak_gb', '-')} | "
                     f"{mem.get('ram_peak_gb', '-')} | {fmt(ps)} | {fmt(lp, 3)} | {str(notes)[:160]} |")
    table = "\n".join(lines)
    (out / "results.md").write_text("# qwen-edit-lite benchmark\n\n" + table + "\n")
    _grids(rows, out)
    return table


def _grids(rows: list, out: Path) -> None:
    from PIL import Image, ImageDraw

    ids = sorted({c["id"] for r in rows for c in r.get("cases", [])})
    for cid in ids:
        tiles = [(r["preset"]["name"], Image.open(c["path"]).convert("RGB"))
                 for r in rows for c in r.get("cases", []) if c["id"] == cid]
        if not tiles:
            continue
        h = 384
        tiles = [(n, im.resize((max(1, im.width * h // im.height), h))) for n, im in tiles]
        grid = Image.new("RGB", (sum(im.width for _, im in tiles), h + 24), "white")
        x = 0
        d = ImageDraw.Draw(grid)
        for n, im in tiles:
            grid.paste(im, (x, 24))
            d.text((x + 4, 4), n, fill="black")
            x += im.width
        grid.save(out / f"grid-{cid}.png")


def main_child(job_path: str, res_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    row = run_preset_inprocess(job["preset"], job["cases"], Path(job["out"]), job["seed"])
    Path(res_path).write_text(json.dumps(row, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main_child(sys.argv[1], sys.argv[2]))
