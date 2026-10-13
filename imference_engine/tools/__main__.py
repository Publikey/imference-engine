"""Command line for the tool catalog.

    python -m imference_engine.tools list
    python -m imference_engine.tools show upscale
    python -m imference_engine.tools export [--full] > tools.json
    python -m imference_engine.tools run upscale -i image=in.png -p scale=2 -o out/
    python -m imference_engine.tools weights [tool ...]      # what is cached / missing

``run`` prints the ToolResult as JSON; progress goes to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys


def _kv(items, multi_ok=False):
    out: dict = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"expected name=value, got {it!r}")
        if k in out:
            out[k] = out[k] if isinstance(out[k], list) else [out[k]]
            out[k].append(v)
        else:
            out[k] = v
    return out


def main(argv=None) -> int:
    from imference_engine.tools import catalog, run_tool, weights
    from imference_engine.tools.errors import ToolError

    ap = argparse.ArgumentParser(prog="python -m imference_engine.tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("tool")
    e = sub.add_parser("export")
    e.add_argument("--full", action="store_true", help="include weight paths")
    r = sub.add_parser("run")
    r.add_argument("tool")
    r.add_argument("-i", "--input", action="append", help="name=path (repeat for multiple)")
    r.add_argument("-p", "--param", action="append", help="name=value")
    r.add_argument("-o", "--out", default=".")
    r.add_argument("--device", default="auto")
    w = sub.add_parser("weights")
    w.add_argument("tools", nargs="*")
    w.add_argument("--fetch", action="store_true", help="download what is missing")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        for m in catalog.all():
            where = m.runtimes.cloud_queue if m.cloud_allowed else "local only"
            print(f"{m.id:20} {m.category:10} {where:12} {m.credits:>4g} cr  {m.name}")
        return 0
    if a.cmd == "show":
        print(json.dumps(catalog.builtin().get(a.tool).to_dict(), indent=2))
        return 0
    if a.cmd == "export":
        print(json.dumps(catalog.export(public=not a.full), indent=2))
        return 0
    if a.cmd == "weights":
        ms = [catalog.get(t) for t in a.tools] if a.tools else catalog.all()
        for m in ms:
            for wt in m.weights:
                there = weights.is_present(wt)
                print(f"{m.id:20} {wt.path:50} {'ok' if there else 'MISSING'}")
                if a.fetch and not there:
                    try:
                        weights.ensure([wt])
                    except ToolError as err:
                        print(f"  -> {err.code}: {err}", file=sys.stderr)
        return 0

    inputs = _kv(a.input)
    params = _kv(a.param)
    try:
        res = run_tool(a.tool, inputs, params, output_dir=a.out, device=a.device,
                       on_progress=lambda st, f: print(f"[{st}] {f:.0%}", file=sys.stderr))
    except ToolError as err:
        print(json.dumps({"error": err.to_dict()}), file=sys.stdout)
        return 2
    print(json.dumps(res.to_dict(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
