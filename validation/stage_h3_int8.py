#!/usr/bin/env python
"""One-shot job: quantize MiniMax-H3 to int8 and stage the mirror onto R2.

The decision this implements: cold loads should pull PRE-QUANTIZED weights from
the CDN mirror (fast, ~50 GB) instead of downloading the ~90 GB bf16 FL2VA half
and re-quantizing on every cold load. This script runs ONCE on a big-RAM box
(no GPU needed — int8 weight-only quantization runs on CPU; budget ~80 GB free
RAM, the two large components are processed sequentially):

1. Pull the FL2VA half of the official repo (transformer/, text_encoder/, the
   VAEs, tokenizer/processor/schedulers, modular_model_index.json).
2. Load transformer + Qwen3-VL through torchao ``Int8WeightOnlyConfig(version=2)``
   (the exact upstream consumer-card recipe, same modules-to-not-convert) and
   ``save_pretrained`` them — the quantization config is serialized into each
   component's config.json, so ``minimax_h3/loader.py`` detects the tree as
   pre-quantized and loads it as-is, no torchao re-quantization pass.
3. Copy the small shared components verbatim, write the ``.manifest.json`` the
   CDN reader expects, and (with --bucket) upload idempotently to R2.

Then point a variant at the mirror repo id and set H3_MODEL_CDN:

    # models.yml
    - name: h3-int8
      kind: video
      engine: minimax_h3
      mode: t2v
      repo: imference/MiniMax-H3-int8        # = --mirror-repo

    H3_MODEL_CDN=https://cdn.example/video python -c "..."

Requires the [minimax-h3] extra + the PR #14355 diffusers build (see
imference_engine/minimax_h3/README.md) + [stage] for the upload.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Small components copied verbatim (weights untouched); the two big ones are
# re-serialized quantized. transformer_ref/ and the original checkpoint folders
# are never pulled.
_COPY_SUBDIRS = ("vae", "audio_vae", "tokenizer", "processor",
                 "scheduler", "audio_scheduler", "video_processor")
_COPY_FILES = ("modular_model_index.json",)


def _pull_source(src: str, cache_dir: str | None) -> str:
    """Return a local dir holding the FL2VA half of ``src`` (repo id or dir)."""
    if os.path.isdir(src):
        return src
    from huggingface_hub import snapshot_download

    from imference_engine.minimax_h3.loader import _H3_PATTERNS
    from imference_engine.runtime.offline import flat_root

    d = os.path.join(flat_root(cache_dir, namespace="video"), src)
    print(f"  pulling {src} (FL2VA half, ~90 GB on first run) ...", flush=True)
    snapshot_download(src, allow_patterns=list(_H3_PATTERNS), local_dir=d)
    return d


def _save_quantized(model, out_dir: str) -> None:
    """Serialize a torchao-quantized component. Tensor-subclass weights may not
    be safetensors-serializable on every torchao version — fall back to the
    torch.save format, which diffusers/transformers load fine."""
    try:
        model.save_pretrained(out_dir, safe_serialization=True)
    except Exception as e:  # noqa: BLE001
        print(f"  safetensors serialization failed ({type(e).__name__}: {e}); "
              f"falling back to torch.save format", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        model.save_pretrained(out_dir, safe_serialization=False)


def quantize_tree(src_dir: str, out_dir: str) -> None:
    import torch

    from imference_engine.minimax_h3.loader import (_TEXT_ENCODER_SKIP,
                                                    _TRANSFORMER_SKIP,
                                                    require_h3_support)
    require_h3_support()
    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    os.makedirs(out_dir, exist_ok=True)

    # Sequential on purpose: each component peaks at its own bf16 size (~62 GB)
    # in host RAM; loading both at once would need ~128 GB.
    t0 = time.time()
    print("  quantizing transformer/ (int8 weight-only v2) ...", flush=True)
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        src_dir, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(version=2), modules_to_not_convert=_TRANSFORMER_SKIP),
        low_cpu_mem_usage=False,
    )
    _save_quantized(transformer, os.path.join(out_dir, "transformer"))
    del transformer
    gc.collect()
    print(f"  transformer done ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    print("  quantizing text_encoder/ (Qwen3-VL, int8 weight-only v2) ...", flush=True)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        src_dir, subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2), modules_to_not_convert=_TEXT_ENCODER_SKIP),
    )
    _save_quantized(text_encoder, os.path.join(out_dir, "text_encoder"))
    del text_encoder
    gc.collect()
    print(f"  text_encoder done ({time.time() - t0:.0f}s)", flush=True)

    for sub in _COPY_SUBDIRS:
        src_sub = os.path.join(src_dir, sub)
        if os.path.isdir(src_sub):
            shutil.copytree(src_sub, os.path.join(out_dir, sub), dirs_exist_ok=True)
    for fn in _COPY_FILES:
        src_fn = os.path.join(src_dir, fn)
        if os.path.isfile(src_fn):
            shutil.copy2(src_fn, os.path.join(out_dir, fn))

    # Sanity: the loader's pre-quantized detection must fire on this tree.
    with open(os.path.join(out_dir, "transformer", "config.json"), encoding="utf-8") as f:
        assert "quantization_config" in json.load(f), \
            "transformer/config.json lacks quantization_config — serialization bug?"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Quantize MiniMax-H3 to int8 and stage the mirror onto R2")
    ap.add_argument("--src", default=None,
                    help="source repo id or local dir (default: the official repo)")
    ap.add_argument("--mirror-repo", default="imference/MiniMax-H3-int8",
                    help="repo id the mirror is staged AS (what variants' `repo` points at)")
    ap.add_argument("--cache-dir", default=None, help="flat-tree root override")
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", ""),
                    help="R2 bucket (or env R2_BUCKET); omit to only build the local tree")
    ap.add_argument("--prefix", default="", help="object-key prefix (match H3_MODEL_CDN)")
    ap.add_argument("--skip-quantize", action="store_true",
                    help="tree already built — only (re)write the manifest + upload")
    args = ap.parse_args()

    from imference_engine.minimax_h3.presets import OFFICIAL_REPO
    from imference_engine.runtime.offline import flat_root, write_manifest
    src = args.src or OFFICIAL_REPO
    out_dir = os.path.join(flat_root(args.cache_dir, namespace="video"),
                           args.mirror_repo)

    print(f"MiniMax-H3 int8 staging: {src} -> {out_dir} (as {args.mirror_repo})")
    if not args.skip_quantize:
        src_dir = _pull_source(src, args.cache_dir)
        quantize_tree(src_dir, out_dir)

    files = write_manifest(out_dir)
    print(f"  manifest: {len(files)} files")

    if not args.bucket:
        print("no --bucket: local tree ready; upload later with --skip-quantize --bucket ...")
        return 0

    from stage_r2 import r2_client, upload_dir  # sibling module — R2 helpers
    s3 = r2_client()
    uploaded = upload_dir(s3, args.bucket, args.prefix, args.mirror_repo, out_dir, files)
    print(f"  uploaded {uploaded} object(s) to r2://{args.bucket}/"
          f"{args.prefix or ''}{'/' if args.prefix else ''}{args.mirror_repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
