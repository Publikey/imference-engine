#!/usr/bin/env python
"""Stage the Llama-3.1-8B-Instruct text encoder (HiDream-I1) onto the CDN mirror,
with its licence texts.

HiDream-I1 conditions on Llama-3.1-8B-Instruct, which is NOT in the HiDream
repos. The engine loads it from ``HIDREAM_LLAMA_REPO`` (default
``unsloth/Meta-Llama-3.1-8B-Instruct`` — an ungated copy of Meta's gated repo,
byte-identical weights). That copy ships no licence file, and the Llama 3.1
Community License requires the agreement to travel with any redistribution. So
the mirror gets the same weights PLUS:

    LICENSE         Llama 3.1 Community License Agreement
    USE_POLICY.md   Llama 3.1 Acceptable Use Policy
    NOTICE          where the files come from, and the "Built with Llama" duty

Pure I/O (no GPU, no model load). Needs the R2_* env vars and the two texts,
saved from Meta's llama-models repository:

    https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE
    https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/USE_POLICY.md

Usage:
    python validation/stage_llama_encoder.py --prefix image \\
        --license LICENSE.txt --use-policy USE_POLICY.md [--rm] [--dry-run]
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

REPO = "unsloth/Meta-Llama-3.1-8B-Instruct"

NOTICE = """\
Llama-3.1-8B-Instruct — text encoder for HiDream-I1 (mirror copy)
=================================================================

Mirrored {date} from {repo} (revision {revision}), an ungated copy of
meta-llama/Meta-Llama-3.1-8B-Instruct. The model files are UNCHANGED.

Added to this copy: LICENSE (Llama 3.1 Community License Agreement),
USE_POLICY.md (Llama 3.1 Acceptable Use Policy) and this NOTICE.

Llama 3.1 is licensed under the Llama 3.1 Community License,
Copyright (c) Meta Platforms, Inc. All Rights Reserved.

Products and services that use these weights must display "Built with Llama"
on a related website, user interface, blog post, about page or product
documentation, and must comply with the Acceptable Use Policy.
"""


def main() -> int:
    import stage_r2

    from imference_engine.hidream.backend import HiDreamBackend
    from imference_engine.runtime.offline import write_manifest

    ap = argparse.ArgumentParser(description="Stage the HiDream Llama encoder with its licence")
    ap.add_argument("--license", required=True, help="text file: Llama 3.1 Community License")
    ap.add_argument("--use-policy", required=True, help="text file: Llama 3.1 Acceptable Use Policy")
    ap.add_argument("--prefix", default="image", help="object-key prefix (match IMAGE_MODEL_CDN)")
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", ""))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--rm", action="store_true", help="delete the local copy after upload")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    patterns = [p for p in HiDreamBackend.LLAMA_PATTERNS
                if not p.startswith(("LICENSE", "USE_POLICY", "NOTICE"))]
    print(f"Staging {REPO} ({len(patterns)} patterns + licence texts)  "
          f"-> key {stage_r2.object_key(args.prefix, REPO, '<file>')}")
    if args.dry_run:
        return 0
    if not args.bucket:
        print("error: set --bucket or R2_BUCKET", file=sys.stderr)
        return 2

    local_dir = stage_r2.populate(REPO, patterns, args.cache_dir)
    shutil.copyfile(args.license, os.path.join(local_dir, "LICENSE"))
    shutil.copyfile(args.use_policy, os.path.join(local_dir, "USE_POLICY.md"))
    try:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(REPO).sha
    except Exception:  # noqa: BLE001
        revision = "unknown"
    with open(os.path.join(local_dir, "NOTICE"), "w", encoding="utf-8") as f:
        f.write(NOTICE.format(date=datetime.date.today().isoformat(), repo=REPO, revision=revision))

    files = write_manifest(local_dir)
    uploaded = stage_r2.upload_dir(stage_r2.r2_client(), args.bucket, args.prefix, REPO,
                                   local_dir, files)
    print(f"  [OK ] {uploaded}/{len(files) + 1} objects uploaded ({REPO})")
    if args.rm:
        shutil.rmtree(local_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
