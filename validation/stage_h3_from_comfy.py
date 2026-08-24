#!/usr/bin/env python
"""One-shot job: build a MiniMax-H3 diffusers tree from ComfyUI single-files.

The community (Comfy-Org/MiniMax-H3, mirrored on civitai) ships H3 as four
single ``.safetensors`` files — DiT partition, 50-layer-truncated Qwen3-VL,
video VAE, audio VAE — ~67 GB at int8 ConvRot versus ~124 GB for the official
bf16 repository. Desktop users running ComfyUI often have them on disk already.
This script converts that stack into the modular diffusers tree
``imference_engine.minimax_h3`` loads, with zero engine changes:

1. Pull the official repo's CONFIG skeleton only (config.json per component,
   tokenizer/processor/scheduler files, modular_model_index.json — a few MB).
2. Stream-convert each Comfy file (``minimax_h3/comfy_convert.py``): ConvRot
   int8 -> bf16 dequant, original-layout -> diffusers key mapping, audio-VAE
   weight-norm resynthesis. Peak RAM stays near one tensor, not one file.
3. ``--profile bf16``: done — the tree is ~125 GB on disk.
   ``--profile int8`` (default): re-quantize transformer + text encoder through
   the exact ``stage_h3_int8.quantize_tree`` recipe (torchao int8 weight-only
   v2, serialized) and drop the intermediate bf16 components — final tree
   ~35 GB, loaded via the loader's pre-quantized fast path. NOTE this is a
   second quantization on top of ConvRot's (~0.5-2% weight error per layer);
   validate output quality e2e before trusting a mirror built this way.

Files wanted (non-"pruned"! the pruned DiT repackages use a low-rank AdaLN
architecture diffusers cannot load — the script detects and refuses them):

    minimax_h3_fl2va_int8_convrot.safetensors        (~34 GB)
    qwen3vl_32b_minimax_h3_int8_convrot.safetensors  (~27 GB)
    minimax_h3_video_vae_fp16.safetensors            (~5 GB)
    minimax_h3_audio_vae_fp32.safetensors            (~0.6 GB)

Then point a variant at the result:

    # models.yml
    - name: h3-comfy-int8
      kind: video
      engine: minimax_h3
      mode: t2v
      repo: imference/MiniMax-H3-comfy-int8     # = --mirror-repo

Requires the [minimax-h3] extra (pins diffusers==0.40.0, the release carrying
PR #14355 — see imference_engine/minimax_h3/README.md); ``--profile int8`` also
needs torchao.
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

# Config-only skeleton of the official repo: everything the tree needs that is
# not weights. A few MB total.
_SKELETON_PATTERNS = [
    "modular_model_index.json",
    "transformer/config.json",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "tokenizer/*",
    "processor/*",
    "vae/config.json",
    "audio_vae/config.json",
    "scheduler/*",
    "audio_scheduler/*",
    "video_processor/*",
]

_COPY_SUBDIRS = ("tokenizer", "processor", "scheduler", "audio_scheduler",
                 "video_processor")


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pull_skeleton(src: str, cache_dir: str | None) -> str:
    """Return a local dir holding the official repo's config skeleton."""
    if os.path.isdir(src):
        return src
    from huggingface_hub import snapshot_download

    from imference_engine.runtime.offline import flat_root

    d = os.path.join(flat_root(cache_dir, namespace="video"), src + "-skeleton")
    print(f"  pulling config skeleton of {src} (a few MB) ...", flush=True)
    snapshot_download(src, allow_patterns=_SKELETON_PATTERNS, local_dir=d)
    return d


def _audio_vae_expected_keys(config: dict) -> set[str]:
    """State-dict keys of a freshly built AutoencoderKLMiniMaxH3Audio (what the
    weight-norm resynthesis validates against). Meta device when possible."""
    import torch

    from imference_engine.minimax_h3.loader import require_h3_support
    require_h3_support()
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
        AutoencoderKLMiniMaxH3Audio)

    kwargs = {k: v for k, v in config.items() if not k.startswith("_")}
    try:
        with torch.device("meta"):
            model = AutoencoderKLMiniMaxH3Audio(**kwargs)
    except Exception:  # noqa: BLE001 — meta tensors trip some param'trizations
        model = AutoencoderKLMiniMaxH3Audio(**kwargs)
    return set(model.state_dict())


def build_bf16_tree(args, skel: str, out_dir: str) -> None:
    from imference_engine.minimax_h3 import comfy_convert as cc

    max_shard = int(args.max_shard_gb * 1024 ** 3)

    t0 = time.time()
    print(f"  converting transformer {args.transformer} ...", flush=True)
    cc.convert_transformer_file(
        args.transformer, os.path.join(out_dir, "transformer"),
        _load_json(os.path.join(skel, "transformer", "config.json")),
        max_shard_bytes=max_shard,
        progress=lambda m: print(f"    {m}", flush=True))
    shutil.copy2(os.path.join(skel, "transformer", "config.json"),
                 os.path.join(out_dir, "transformer", "config.json"))
    print(f"  transformer done ({time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    print(f"  converting text_encoder {args.text_encoder} ...", flush=True)
    patched = cc.convert_text_encoder_file(
        args.text_encoder, os.path.join(out_dir, "text_encoder"),
        _load_json(os.path.join(skel, "text_encoder", "config.json")),
        max_shard_bytes=max_shard,
        progress=lambda m: print(f"    {m}", flush=True))
    with open(os.path.join(out_dir, "text_encoder", "config.json"),
              "w", encoding="utf-8") as f:
        json.dump(patched, f, indent=2)
    gen_cfg = os.path.join(skel, "text_encoder", "generation_config.json")
    if os.path.isfile(gen_cfg):
        shutil.copy2(gen_cfg, os.path.join(out_dir, "text_encoder",
                                           "generation_config.json"))
    print(f"  text_encoder done ({time.time() - t0:.0f}s)", flush=True)

    print(f"  converting video VAE {args.video_vae} ...", flush=True)
    vae_config = _load_json(os.path.join(skel, "vae", "config.json"))
    cc.convert_video_vae_file(args.video_vae, os.path.join(out_dir, "vae"),
                              vae_config, max_shard_bytes=max_shard)
    shutil.copy2(os.path.join(skel, "vae", "config.json"),
                 os.path.join(out_dir, "vae", "config.json"))

    print(f"  converting audio VAE {args.audio_vae} ...", flush=True)
    audio_config = _load_json(os.path.join(skel, "audio_vae", "config.json"))
    cc.convert_audio_vae_file(args.audio_vae, os.path.join(out_dir, "audio_vae"),
                              _audio_vae_expected_keys(audio_config))
    shutil.copy2(os.path.join(skel, "audio_vae", "config.json"),
                 os.path.join(out_dir, "audio_vae", "config.json"))

    for sub in _COPY_SUBDIRS:
        src_sub = os.path.join(skel, sub)
        if os.path.isdir(src_sub):
            shutil.copytree(src_sub, os.path.join(out_dir, sub), dirs_exist_ok=True)
    shutil.copy2(os.path.join(skel, "modular_model_index.json"),
                 os.path.join(out_dir, "modular_model_index.json"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a MiniMax-H3 diffusers tree from ComfyUI single-files")
    ap.add_argument("--transformer", required=True,
                    help="minimax_h3_fl2va_int8_convrot.safetensors (or bf16; NOT 'pruned')")
    ap.add_argument("--text-encoder", required=True,
                    help="qwen3vl_32b_minimax_h3_int8_convrot.safetensors (or bf16)")
    ap.add_argument("--video-vae", required=True,
                    help="minimax_h3_video_vae_fp16.safetensors")
    ap.add_argument("--audio-vae", required=True,
                    help="minimax_h3_audio_vae_fp32.safetensors")
    ap.add_argument("--official", default=None,
                    help="official repo id or local dir for the config skeleton "
                         "(default: the official repo)")
    ap.add_argument("--mirror-repo", default=None,
                    help="repo id the tree is staged AS (what variants' `repo` "
                         "points at; default imference/MiniMax-H3-comfy-<profile>)")
    ap.add_argument("--profile", choices=("int8", "bf16"), default="int8",
                    help="int8 (default): torchao-requantize the two big "
                         "components (~35 GB tree); bf16: dequantized tree as-is "
                         "(~125 GB)")
    ap.add_argument("--cache-dir", default=None, help="flat-tree root override")
    ap.add_argument("--max-shard-gb", type=float, default=5.0,
                    help="max output shard size (GB)")
    ap.add_argument("--keep-bf16", action="store_true",
                    help="with --profile int8, keep the intermediate bf16 tree")
    args = ap.parse_args()

    from imference_engine.minimax_h3.presets import OFFICIAL_REPO
    from imference_engine.runtime.offline import flat_root, write_manifest

    official = args.official or OFFICIAL_REPO
    mirror = args.mirror_repo or f"imference/MiniMax-H3-comfy-{args.profile}"
    root = flat_root(args.cache_dir, namespace="video")
    out_dir = os.path.join(root, mirror)

    for path in (args.transformer, args.text_encoder, args.video_vae, args.audio_vae):
        if not os.path.isfile(path):
            print(f"error: {path} is not a file", file=sys.stderr)
            return 2

    print(f"MiniMax-H3 Comfy staging: -> {out_dir} (as {mirror}, profile={args.profile})")
    skel = _pull_skeleton(official, args.cache_dir)

    if args.profile == "bf16":
        build_bf16_tree(args, skel, out_dir)
    else:
        bf16_dir = out_dir + "-bf16.tmp"
        build_bf16_tree(args, skel, bf16_dir)
        gc.collect()
        print("  re-quantizing to torchao int8 (stage_h3_int8 recipe) ...", flush=True)
        from stage_h3_int8 import quantize_tree  # sibling module
        quantize_tree(bf16_dir, out_dir)
        if not args.keep_bf16:
            shutil.rmtree(bf16_dir, ignore_errors=True)
            print(f"  removed intermediate {bf16_dir}", flush=True)

    files = write_manifest(out_dir)
    print(f"  manifest: {len(files)} files")
    print(f"done. Point a variant's `repo` at {mirror!r}; upload with "
          f"stage_h3_int8.py --skip-quantize --mirror-repo {mirror} --bucket ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
