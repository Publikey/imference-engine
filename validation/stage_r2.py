#!/usr/bin/env python
"""Stage each image backend's BASE components onto the R2 (S3-compatible) CDN mirror.

Pure I/O — no GPU, no model load. For each selected engine it:
  1. pulls ONLY the shared base components from HuggingFace into the flat offline
     tree — the EXACT patterns the backend uses at load (its ``BASE_PATTERNS`` /
     ``CONFIG_PATTERNS``, never the transformer weights the checkpoint replaces);
  2. writes the ``.manifest.json`` the CDN reader (``offline.py::_cdn_snapshot``)
     reads back to repopulate the tree from R2;
  3. uploads the repo dir to R2 under ``<prefix>/<repo>/<rel>``, idempotently
     (skips objects already present with the same size — resumable);
  4. optionally deletes the local dir after upload (``--rm``) so the box never
     holds more than one base at a time (the disk-tight streaming path).

Result: workers / imference-desktop with ``IMAGE_MODEL_CDN=<bucket-url>`` load
these bases straight from R2, never touching HuggingFace — immune to a repo going
gated or being removed.

Layout produced (matches ``offline.py::local_repo_dir`` / ``_cdn_snapshot``):
    <bucket>/<prefix>/<repo>/.manifest.json
    <bucket>/<prefix>/<repo>/<file>...
``<prefix>`` (``--prefix``, default empty) must line up with whatever
``IMAGE_MODEL_CDN`` points at: cdn_base + '/' + repo + '/' + rel is the read URL.

Repo staged per engine (from base_models.yaml + the backend, single source of truth):
    flux       black-forest-labs/FLUX.1-dev                 (GATED — hf auth login first)
    chroma     lodestones/Chroma1-HD
    sd15       stable-diffusion-v1-5/stable-diffusion-v1-5   (config/tokenizer only, ~MB)
    qwenimage  Qwen/Qwen-Image
    anima      circlestone-labs/Anima-Base-v1.0-Diffusers    (whole modular repo)
    krea2      krea/Krea-2-Turbo                             (GATED — hf auth login first)
    (sdxl/zimage work too if you pass them — already mirrored in your setup)

Credentials (env): R2_ENDPOINT (full S3 endpoint; R2_ENDPOINT_URL / R2_ACCOUNT_ID
also accepted), R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET.

Usage (on the remote instance, after ``pip install -e ".[runtime,stage]"``):
    hf auth login                     # once, for the gated FLUX base
    export R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
    export R2_ACCESS_KEY_ID=...  R2_SECRET_ACCESS_KEY=...  R2_BUCKET=gen-models

    python validation/stage_r2.py --prefix image                 # all CDN-wired bases -> gen-models/image/<repo>/...
    python validation/stage_r2.py --prefix image --engines flux,chroma
    python validation/stage_r2.py --prefix image --rm            # delete each local dir after upload
    python validation/stage_r2.py --prefix image --dry-run       # print the plan + resolved repos, touch nothing

    # Wan GGUF mode — stage individual expert files to <prefix>/<repo>/<file> (no
    # manifest; Wan reads GGUF as direct files). e.g. mirror bullerwins i2v Q8:
    python validation/stage_r2.py --prefix wan22 --rm \
        --wan-gguf bullerwins/Wan2.2-I2V-A14B-GGUF \
        --files wan2.2_i2v_high_noise_14B_Q8_0.gguf,wan2.2_i2v_low_noise_14B_Q8_0.gguf

Exit code is non-zero if any selected engine failed, so it doubles as a CI gate.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "base_models.yaml"

# Run-from-source: put the repo root on the path so `python validation/stage_r2.py`
# finds imference_engine without `pip install -e` (harmless when installed).
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

# Engines that serve their base from the CDN today (route through local_repo_dir).
DEFAULT_ENGINES = ["flux", "chroma", "sd15", "qwenimage", "anima", "krea2"]

# engine -> "module:ClassName". Imported lazily and torch-free (backends import
# torch inside methods), so we can read BASE_PATTERNS / CONFIG_PATTERNS without a
# GPU or a heavy import. The class is the single source of truth for the patterns.
_BACKENDS = {
    "flux": "imference_engine.flux:FluxBackend",
    "chroma": "imference_engine.chroma:ChromaBackend",
    "qwenimage": "imference_engine.qwenimage:QwenImageBackend",
    "zimage": "imference_engine.zimage:ZImageBackend",
    "sd15": "imference_engine.pipelines.sd15:SD15Backend",
    "sdxl": "imference_engine.pipelines.sdxl:SDXLBackend",
    "krea2": "imference_engine.krea2:Krea2Backend",
}


def load_config(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _backend_class(engine: str):
    ref = _BACKENDS.get(engine)
    if not ref:
        return None
    mod, cls = ref.split(":")
    return getattr(importlib.import_module(mod), cls)


def resolve_base(engine: str, entry: dict):
    """Return ``(repo, patterns)`` for the base components to stage, or ``None``
    when the engine has no CDN-served base (nothing to upload).

    - repo mode (anima)        -> (repo, None)         whole repo, all files
    - transformer-only base    -> (base_model, BASE_PATTERNS)     flux/chroma/qwenimage/zimage
    - single-file + config repo-> (CONFIG_REPO, CONFIG_PATTERNS)  sd15/sdxl
    """
    if entry.get("mode", "single_file") == "repo":
        return entry["repo"], None

    be = _backend_class(engine)
    base = entry.get("base_model")
    if base and be is not None and getattr(be, "BASE_PATTERNS", None):
        return base, list(be.BASE_PATTERNS)
    # single-file checkpoints whose layout comes from a config repo (no base_model)
    if be is not None and getattr(be, "CONFIG_REPO", None):
        return be.CONFIG_REPO, list(be.CONFIG_PATTERNS)
    return None


def object_key(prefix: str, repo: str, rel: str) -> str:
    """<prefix>/<repo>/<rel> with no leading/duplicate slashes — the exact suffix
    the CDN reader appends to IMAGE_MODEL_CDN."""
    return "/".join(p for p in (prefix.strip("/"), repo, rel) if p)


def populate(repo: str, patterns, cache_dir: str | None) -> str:
    """Pull the base into the flat offline tree and return its dir. Always online
    here (staging box), so we call snapshot_download directly — none of
    local_repo_dir's CDN/offline branching applies."""
    from huggingface_hub import snapshot_download

    from imference_engine.runtime.offline import flat_root

    d = os.path.join(flat_root(cache_dir, namespace="image"), repo)
    print(f"  pulling {repo} ({'all files' if patterns is None else f'{len(patterns)} patterns'}) ...",
          flush=True)
    snapshot_download(repo, allow_patterns=patterns, local_dir=d)
    return d


def r2_client():
    import boto3

    # R2_ENDPOINT is the worker convention (full S3 endpoint). R2_ENDPOINT_URL is
    # accepted as an alias; R2_ACCOUNT_ID is a fallback that builds the default
    # endpoint from the account id.
    endpoint = os.environ.get("R2_ENDPOINT") or os.environ.get("R2_ENDPOINT_URL")
    if not endpoint:
        acct = os.environ.get("R2_ACCOUNT_ID")
        if not acct:
            raise SystemExit("set R2_ENDPOINT (or R2_ENDPOINT_URL / R2_ACCOUNT_ID)")
        endpoint = f"https://{acct}.r2.cloudflarestorage.com"
    try:
        key_id = os.environ["R2_ACCESS_KEY_ID"]
        secret = os.environ["R2_SECRET_ACCESS_KEY"]
    except KeyError as e:
        raise SystemExit(f"missing env var {e}") from None
    # region_name='auto' is what R2 expects for SigV4.
    return boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=key_id, aws_secret_access_key=secret,
        region_name="auto",
    )


def upload_dir(s3, bucket: str, prefix: str, repo: str, local_dir: str, files: list) -> int:
    """Upload the manifest + every file it lists to R2. Idempotent: an object that
    already exists with the same byte size is skipped, so a killed run resumes.
    Returns the number of objects actually uploaded this call."""
    from botocore.exceptions import ClientError

    uploaded = 0
    # ".manifest.json" is not in its own list but the reader fetches it first, so
    # it MUST be uploaded. Hidden markers (.cdn_complete/.hf_complete/.cache) are
    # excluded by write_manifest and are reader-local — never uploaded.
    for rel in [*files, ".manifest.json"]:
        src = os.path.join(local_dir, *rel.split("/"))
        size = os.path.getsize(src)
        key = object_key(prefix, repo, rel)
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            if head["ContentLength"] == size:
                continue  # already there, same size -> skip
        except ClientError:
            pass  # 404 (or no head perm) -> upload
        s3.upload_file(src, bucket, key)  # multipart automatic for big shards
        uploaded += 1
    return uploaded


def stage_wan_gguf(repo: str, files: list, args, s3) -> int:
    """Stage individual Wan GGUF expert files to R2 under ``<prefix>/<repo>/<file>``.

    Unlike the image base components (multi-file repos + a manifest), Wan reads GGUF
    as DIRECT files (``<WAN_MODEL_CDN>/<repo>/<file>``, no manifest), so this just
    downloads each named file from HF and uploads it idempotently (skips an object
    already present with the same size). ``--rm`` frees each after upload.
    Returns non-zero if any file failed.
    """
    from botocore.exceptions import ClientError
    from huggingface_hub import hf_hub_download

    from imference_engine.runtime.offline import flat_root

    dest_root = os.path.join(flat_root(args.cache_dir, namespace="wan"), repo)
    failed = 0
    for fn in files:
        key = object_key(args.prefix, repo, fn)
        print(f"\n=== {fn} ===  -> key {key}", flush=True)
        t0 = time.time()
        try:
            print(f"  downloading {repo}/{fn} ...", flush=True)
            src = hf_hub_download(repo, fn, local_dir=dest_root)
            size = os.path.getsize(src)
            try:
                skip = s3.head_object(Bucket=args.bucket, Key=key)["ContentLength"] == size
            except ClientError:
                skip = False
            if skip:
                print(f"  [SKIP] already on R2 ({size} bytes)", flush=True)
            else:
                s3.upload_file(src, args.bucket, key)  # multipart automatic
                print(f"  [OK ] {round(time.time() - t0, 1)}s  uploaded {size} bytes", flush=True)
            if args.rm:
                os.remove(src)
                print(f"  rm: freed {src}", flush=True)
        except Exception as e:  # noqa: BLE001 — one file's failure must not stop the rest
            print(f"  [FAIL] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            failed += 1
    print("\n" + "=" * 60)
    print(f"Wan GGUF: {len(files) - failed}/{len(files)} staged to {repo}")
    return 1 if failed else 0


def stage_one(engine: str, entry: dict, args, s3) -> dict:
    rec = {"engine": engine, "status": "fail", "seconds": None,
           "repo": None, "files": None, "uploaded": None, "error": None}
    t0 = time.time()
    try:
        target = resolve_base(engine, entry)
        if target is None:
            rec.update(status="skip", error="no CDN-served base (nothing to stage)")
            return rec
        repo, patterns = target
        rec["repo"] = repo

        local_dir = populate(repo, patterns, args.cache_dir)

        from imference_engine.runtime.offline import write_manifest
        files = write_manifest(local_dir)
        rec["files"] = len(files)

        uploaded = upload_dir(s3, args.bucket, args.prefix, repo, local_dir, files)
        rec["uploaded"] = uploaded
        rec["status"] = "ok"

        if args.rm:
            shutil.rmtree(local_dir, ignore_errors=True)
            print(f"  rm: freed {local_dir}", flush=True)
    except Exception as e:  # noqa: BLE001 — one engine's failure must not stop the rest
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    finally:
        rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage image backend base components onto R2")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="base_models.yaml")
    ap.add_argument("--engines", default="", help=f"comma list (default: {','.join(DEFAULT_ENGINES)})")
    ap.add_argument("--cache-dir", default=None,
                    help="flat-tree root override (default: HF_HOME/image or ~/.cache/image)")
    ap.add_argument("--prefix", default="", help="object-key prefix under the bucket (match IMAGE_MODEL_CDN)")
    ap.add_argument("--bucket", default=os.environ.get("R2_BUCKET", ""), help="R2 bucket (or env R2_BUCKET)")
    ap.add_argument("--rm", action="store_true", help="delete each local base dir after upload (disk-tight)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + resolved repos, touch nothing")
    ap.add_argument("--wan-gguf", default=None, metavar="REPO",
                    help="Wan GGUF mode: upload --files of this repo to <prefix>/<repo>/<file> "
                         "(direct files, no manifest), e.g. bullerwins/Wan2.2-I2V-A14B-GGUF")
    ap.add_argument("--files", default="",
                    help="comma list of .gguf filenames to stage (--wan-gguf mode)")
    args = ap.parse_args()

    # Wan GGUF mode is a separate path (single files, no base_models.yaml, wan prefix).
    if args.wan_gguf:
        files = [f.strip() for f in args.files.split(",") if f.strip()]
        if not files:
            print("error: --wan-gguf needs --files (comma list of .gguf filenames)", file=sys.stderr)
            return 2
        print(f"Staging {len(files)} Wan GGUF file(s) from {args.wan_gguf}  (prefix {args.prefix!r})")
        if args.dry_run:
            for fn in files:
                print(f"  {fn}  -> key {object_key(args.prefix, args.wan_gguf, fn)}")
            return 0
        if not args.bucket:
            print("error: set --bucket or R2_BUCKET", file=sys.stderr)
            return 2
        return stage_wan_gguf(args.wan_gguf, files, args, r2_client())

    config = load_config(Path(args.config))
    selected = [e.strip() for e in args.engines.split(",") if e.strip()] or list(DEFAULT_ENGINES)
    unknown = [e for e in selected if e not in config]
    if unknown:
        print(f"error: unknown engine(s) {unknown}; config has {list(config)}", file=sys.stderr)
        return 2

    print(f"Staging {len(selected)} engine(s): {', '.join(selected)}")
    if args.dry_run:
        for e in selected:
            target = resolve_base(e, config[e])
            if target is None:
                print(f"  {e:11s} -- no CDN-served base (skip)")
            else:
                repo, patterns = target
                n = "all files" if patterns is None else f"{len(patterns)} patterns"
                print(f"  {e:11s} <- {repo}  ({n})  -> key {object_key(args.prefix, repo, '<file>')}")
        return 0

    if not args.bucket:
        print("error: set --bucket or R2_BUCKET", file=sys.stderr)
        return 2

    s3 = r2_client()
    records = []
    for e in selected:
        print(f"\n=== {e} ===", flush=True)
        rec = stage_one(e, config[e], args, s3)
        tag = {"ok": "OK ", "skip": "SKIP"}.get(rec["status"], "FAIL")
        detail = (f"{rec['uploaded']}/{rec['files']} uploaded ({rec['repo']})"
                  if rec["status"] == "ok" else rec.get("error"))
        print(f"  [{tag}] {rec['seconds']}s  {detail}", flush=True)
        records.append(rec)

    ok = sum(r["status"] == "ok" for r in records)
    skipped = sum(r["status"] == "skip" for r in records)
    failed = [r for r in records if r["status"] == "fail"]
    print("\n" + "=" * 60)
    print(f"Summary: {ok} staged, {skipped} skipped, {len(failed)} failed")
    for r in records:
        mark = {"ok": "✓", "skip": "–"}.get(r["status"], "✗")
        detail = (f"{r['uploaded']}/{r['files']} obj  {r['repo']}"
                  if r["status"] == "ok" else r.get("error"))
        print(f"  {mark} {r['engine']:11s} {str(r['seconds']) + 's':>7}  {detail}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
