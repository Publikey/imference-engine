"""Stage the editing-tool weights on the R2 mirror (``TOOLS_MODEL_CDN``).

For every weight of every tool manifest: fetch its ``source`` (direct .onnx
URL, a .pth to convert to ONNX, or a HuggingFace repo to tar), compute the
sha256, upload it to ``<bucket>/<prefix>/<path>`` and optionally write
``sha256`` / ``size_mb`` back into the manifests (commit them: the engine then
verifies every download).

    pip install -e ".[stage,tools]" torch huggingface_hub   # torch only for .pth conversion
    export R2_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=...
    python validation/stage_tools_r2.py --prefix tools --update-manifests
    python validation/stage_tools_r2.py --tools upscale,remove-background --no-upload
    python validation/stage_tools_r2.py --source rife/rife-v4.6.onnx=/path/rife46.onnx

Then set ``TOOLS_MODEL_CDN=https://<cdn host>/<prefix>`` on workers / Desktop
(or rely on the ``$IMAGE_MODEL_CDN/tools`` fallback by using prefix
``<image prefix>/tools``).

Sources that are not a direct file (RIFE "exported to ONNX") need
``--source <path>=<url or local file>``: there is no single official ONNX
export, and the converter for RIFE is out of scope here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "imference_engine" / "tools" / "manifests"
sys.path.insert(0, str(ROOT))


# ------------------------------------------------------------------ fetchers

def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    print(f"  GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "imference-stage"})
    with urllib.request.urlopen(req, timeout=120) as r, open(str(dest) + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 4 << 20)
    os.replace(str(dest) + ".part", dest)
    head = dest.read_bytes()[:200]
    if head.startswith(b"version https://git-lfs"):
        raise SystemExit(f"{url} returned a git-lfs pointer, not the file")
    return dest


def rrdbnet(num_block: int):
    """Real-ESRGAN's RRDBNet (basicsr), scale x4 — enough to load the .pth."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    class RDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            nf = 64
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(nf) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return Net()


def convert_realesrgan(pth: Path, dest: Path) -> Path:
    import torch

    sd = torch.load(pth, map_location="cpu", weights_only=True)
    sd = sd.get("params_ema") or sd.get("params") or sd
    n = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("body."))
    net = rrdbnet(n).eval()
    net.load_state_dict(sd, strict=True)
    x = torch.rand(1, 3, 64, 64)
    print(f"  export RRDBNet({n} blocks) -> {dest.name}", flush=True)
    torch.onnx.export(net, x, str(dest), input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "H", 3: "W"}}, dynamo=False)
    return dest


def tar_hf_repo(repo: str, dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    d = Path(snapshot_download(repo, local_dir=str(dest.parent / (dest.stem + "-src"))))
    print(f"  tar {repo} -> {dest.name}", flush=True)
    with tarfile.open(dest, "w") as tf:
        for p in sorted(d.rglob("*")):
            if p.is_file() and ".cache" not in p.parts:
                tf.add(p, arcname=str(p.relative_to(d)))
    return dest


def build(w: dict, work: Path, overrides: dict) -> Path:
    """Produce the local file for weight ``w`` (manifest dict)."""
    dest = work / w["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = overrides.get(w["path"], w.get("source", ""))
    if not src:
        raise SystemExit(f"{w['path']}: no source; pass --source {w['path']}=<url|file>")
    if os.path.isfile(src):
        if src.endswith(".pth"):
            return convert_realesrgan(Path(src), dest)
        shutil.copyfile(src, dest)
        return dest
    url = src.split(" ")[0]
    if url.startswith("https://huggingface.co/") and "/resolve/" not in url:
        return tar_hf_repo(url.split("huggingface.co/")[1].strip("/"), dest)
    if url.endswith(".pth") and "realesrgan" in w["path"].lower():
        pth = download(url, work / "_src" / os.path.basename(url))
        return convert_realesrgan(pth, dest)
    if " " in src or not url.startswith("http") or url.endswith(".pth"):
        raise SystemExit(f"{w['path']}: source {src!r} is not a direct file; pass --source {w['path']}=<url|file>")
    return download(url, dest)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", help="comma-separated tool ids (default: all with weights)")
    ap.add_argument("--work", default="tools-stage", help="local working dir")
    ap.add_argument("--prefix", default="tools", help="object key prefix in the bucket")
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET"))
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--update-manifests", action="store_true", help="write sha256 + size_mb into manifests")
    ap.add_argument("--source", action="append", default=[], help="<weight path>=<url or local file>")
    a = ap.parse_args()

    overrides = dict(s.split("=", 1) for s in a.source)
    wanted = set(a.tools.split(",")) if a.tools else None
    work = Path(a.work)
    s3 = None
    if not a.no_upload:
        if not a.bucket:
            raise SystemExit("set R2_BUCKET or --bucket (or --no-upload)")
        from stage_r2 import r2_client  # same env contract as the image mirror
        s3 = r2_client()

    for mf in sorted(MANIFESTS.glob("*.json")):
        doc = json.loads(mf.read_text())
        if not doc.get("weights") or (wanted and doc["id"] not in wanted):
            continue
        print(f"[{doc['id']}]", flush=True)
        changed = False
        for w in doc["weights"]:
            try:
                local = build(w, work, overrides)
            except SystemExit as e:
                print(f"  SKIP {e}", flush=True)
                continue
            digest, size = sha256(local), local.stat().st_size
            print(f"  {w['path']}  {size / 1e6:.1f} MB  sha256={digest}", flush=True)
            if s3 is not None:
                key = f"{a.prefix.strip('/')}/{w['path']}"
                try:
                    if s3.head_object(Bucket=a.bucket, Key=key)["ContentLength"] == size:
                        print("  already uploaded", flush=True)
                    else:
                        raise KeyError
                except Exception:
                    s3.upload_file(str(local), a.bucket, key)
                    print(f"  uploaded s3://{a.bucket}/{key}", flush=True)
            if a.update_manifests and (w.get("sha256") != digest or w.get("size_mb") != round(size / 1e6, 1)):
                w["sha256"], w["size_mb"] = digest, round(size / 1e6, 1)
                changed = True
        if changed:
            mf.write_text(json.dumps(doc, indent=2) + "\n")
            print(f"  updated {mf.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    sys.exit(main())
