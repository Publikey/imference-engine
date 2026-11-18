#!/usr/bin/env python
"""Build (and optionally upload) the slim Lumina-Image 2.0 base for the CDN mirror.

Upstream ``Alpha-VLLM/Lumina-Image-2.0`` stores the Gemma-2-2b text encoder in
fp32 (10.4 GB in three shards). The pipeline runs it in bf16, so the mirror
carries a derived base, ``imference/Lumina-Image-2.0-bf16`` — the SAME shared
components with the encoder re-saved in bf16 (5.2 GB, one file):

    model_index.json  scheduler/  tokenizer/  vae/  transformer/config.json
    text_encoder/{config.json, model.safetensors}      <- bf16
    LICENSE            Apache-2.0 (Lumina-Image 2.0)
    NOTICE             what was changed, where each part comes from
    GEMMA_TERMS.md     Gemma Terms of Use + Prohibited Use Policy (the encoder)

``Lumina2Backend`` uses it whenever ``IMAGE_MODEL_CDN`` serves it and falls
back to the upstream repo otherwise. It is a mirror-only artifact: it never
goes to HuggingFace.

The Gemma Terms of Use allow redistribution of Gemma and its derivatives
provided the terms travel with them and modified files carry a notice — that is
what NOTICE and GEMMA_TERMS.md are for. Pass the two texts saved from
https://ai.google.dev/gemma/terms and
https://ai.google.dev/gemma/prohibited_use_policy.

Two steps, because the build needs torch and the upload needs the R2
credentials, which are rarely on the same machine:

    # on a box with the engine runtime installed (no credentials needed)
    python validation/stage_lumina_slim.py build --out /data/slim \\
        --gemma-terms terms.txt --gemma-policy policy.txt

    # where the R2_* env vars live (pure I/O)
    python validation/stage_lumina_slim.py upload --dir /data/slim --prefix image
"""
from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

UPSTREAM = "Alpha-VLLM/Lumina-Image-2.0"
SLIM = "imference/Lumina-Image-2.0-bf16"

APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"

NOTICE = """\
Lumina-Image 2.0 — slim base components (mirror-only artifact)
================================================================

Built {date} from {upstream} (revision {revision}).

Contents and origin
-------------------
- model_index.json, scheduler/, tokenizer/, vae/, transformer/config.json:
  copied UNCHANGED from {upstream}. Lumina-Image 2.0 is released by
  Alpha-VLLM under the Apache License 2.0 (see LICENSE). The VAE is the FLUX.1
  autoencoder as redistributed in that repository.
- text_encoder/: Google's Gemma-2-2b encoder as redistributed in {upstream}.
  MODIFIED: the weights were converted from float32 to bfloat16 and re-saved as
  a single safetensors file (config `torch_dtype` updated accordingly). No other
  change. Gemma is provided under and subject to the Gemma Terms of Use found
  at ai.google.dev/gemma/terms — a copy is in GEMMA_TERMS.md, together with the
  Gemma Prohibited Use Policy, which applies to every use of this encoder.

No transformer weights are included: they come from the checkpoint being run.
"""


def build(args) -> int:
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import Gemma2Model

    from imference_engine.runtime.offline import write_manifest

    out = Path(args.out) / SLIM
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.upstream_dir) if args.upstream_dir else Path(snapshot_download(
        UPSTREAM, allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                                  "text_encoder/*", "vae/*", "transformer/config.json"]))
    for rel in ("model_index.json", "scheduler", "tokenizer", "vae"):
        s, d = src / rel, out / rel
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    (out / "transformer").mkdir(exist_ok=True)
    shutil.copy2(src / "transformer" / "config.json", out / "transformer" / "config.json")

    print("converting the Gemma-2-2b encoder fp32 -> bf16 ...", flush=True)
    te = Gemma2Model.from_pretrained(str(src / "text_encoder"), dtype=torch.bfloat16)
    te.save_pretrained(str(out / "text_encoder"), max_shard_size="20GB")

    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(
            APACHE_URL, headers={"User-Agent": "imference-engine"})) as r:
        (out / "LICENSE").write_bytes(r.read())
    try:
        revision = HfApi().model_info(UPSTREAM).sha
    except Exception:  # noqa: BLE001 — offline build: unknown revision
        revision = "unknown"
    (out / "NOTICE").write_text(NOTICE.format(
        date=datetime.date.today().isoformat(), upstream=UPSTREAM, revision=revision),
        encoding="utf-8")
    terms = Path(args.gemma_terms).read_text(encoding="utf-8")
    policy = Path(args.gemma_policy).read_text(encoding="utf-8")
    (out / "GEMMA_TERMS.md").write_text(
        "# Gemma Terms of Use\n\nSource: https://ai.google.dev/gemma/terms\n\n" + terms
        + "\n\n# Gemma Prohibited Use Policy\n\n"
          "Source: https://ai.google.dev/gemma/prohibited_use_policy\n\n" + policy + "\n",
        encoding="utf-8")

    files = write_manifest(str(out))
    size = sum((out / f).stat().st_size for f in files)
    print(f"built {out}: {len(files)} files, {size / 1e9:.2f} GB")
    return 0


def upload(args) -> int:
    import stage_r2

    from imference_engine.runtime.offline import write_manifest

    local_dir = str(Path(args.dir) / SLIM)
    files = write_manifest(local_dir)
    bucket = args.bucket or os.environ.get("R2_BUCKET", "")
    if not bucket:
        print("error: set --bucket or R2_BUCKET", file=sys.stderr)
        return 2
    n = stage_r2.upload_dir(stage_r2.r2_client(), bucket, args.prefix, SLIM, local_dir, files)
    print(f"uploaded {n}/{len(files) + 1} objects to {stage_r2.object_key(args.prefix, SLIM, '')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Slim Lumina-Image 2.0 base (bf16 Gemma encoder)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the slim base dir (needs torch + transformers)")
    b.add_argument("--out", required=True, help=f"output root; the base lands in <out>/{SLIM}")
    b.add_argument("--upstream-dir", default=None,
                   help=f"local copy of {UPSTREAM} (default: download the needed files)")
    b.add_argument("--gemma-terms", required=True, help="text file: Gemma Terms of Use")
    b.add_argument("--gemma-policy", required=True, help="text file: Gemma Prohibited Use Policy")
    b.set_defaults(fn=build)
    u = sub.add_parser("upload", help="upload a built base to R2 (needs the R2_* env vars)")
    u.add_argument("--dir", required=True, help="the --out root given to build")
    u.add_argument("--prefix", default="image", help="object-key prefix (match IMAGE_MODEL_CDN)")
    u.add_argument("--bucket", default="", help="R2 bucket (or env R2_BUCKET)")
    u.set_defaults(fn=upload)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
