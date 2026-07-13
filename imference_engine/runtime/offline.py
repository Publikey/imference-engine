"""Shared offline model-tree utilities (flat, symlink-free <root>/<repo>/<file>).

Promoted from ``wan/loader.py`` so the image side (SDXL / Z-Image) reuses the
exact same offline machinery the Wan engine validated: a flat, portable tree
that works fully offline once populated (``HF_HUB_OFFLINE=1``) and can be fed
from a CDN mirror (R2) on demand via the parallel downloader.

The only generalization over the Wan original is a ``namespace`` argument so
each engine family gets its own sub-tree under ``HF_HOME`` (``wan`` vs
``image``) without colliding. The Wan callers pass ``namespace="wan"`` through
thin shims in ``wan/loader.py`` — behaviour there is byte-for-byte unchanged.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Marker written into a repo dir once its CDN manifest pull fully completes. Lets a
# re-run tell "fully pulled" from "partial pull killed mid-way" (the sentinel can't).
_CDN_COMPLETE = ".cdn_complete"

# Same idea for the HuggingFace snapshot path: written only after snapshot_download
# returns. A killed pull (e.g. disk full) can leave the sentinel (model_index.json,
# downloaded early) while big shards are missing — the sentinel alone would then
# short-circuit to a BROKEN tree on the next load (crash at a missing shard).
_HF_COMPLETE = ".hf_complete"


def flat_root(cache_dir: Optional[str], *, namespace: str) -> str:
    """Root of the FLAT, symlink-free model tree (``<root>/<repo>/<file>``).

    Holds base configs AND on-demand weights (VAE, GGUF, LoRA, …). Defaults next
    to the HF cache on the big volume under a per-engine ``namespace`` sub-dir;
    overridable via ``cache_dir`` (e.g. ``IMAGE_MODEL_CACHE`` / ``WAN_MODEL_CACHE``).
    """
    if cache_dir:
        return cache_dir
    hf = os.environ.get("HF_HOME")
    if hf:
        return os.path.join(hf, namespace)
    return os.path.join(os.path.expanduser("~"), ".cache", namespace)


def local_repo_dir(
    repo: str,
    patterns: list,
    cache_dir: Optional[str],
    *,
    namespace: str,
    sentinel: str = "model_index.json",
    cdn_base: Optional[str] = None,
) -> str:
    """Ensure a repo's needed files are in a FLAT local dir and return it.

    Writes real files (NO symlinks) on every OS -> the tree is portable
    Windows<->Mac<->Linux, unlike the symlinked HF cache. Idempotent; offline
    once populated (``from_pretrained`` then just reads local files).

    Population source, in order:
      1. Already there (sentinel present) -> return, zero network. A pre-shipped
         tree works fully offline.
      2. ``cdn_base`` set -> pull the repo's file list from a CDN manifest
         (``<cdn_base>/<repo>/.manifest.json``) via the parallel downloader.
         Plain HTTP, so it coexists with ``HF_HUB_OFFLINE=1`` and never touches
         HuggingFace. If the CDN can't serve the repo (e.g. it isn't mirrored ->
         404 on the manifest) the pull falls back to HuggingFace, so a gap in the
         mirror degrades to a slower download instead of a hard crash.
      3. Otherwise -> ``snapshot_download`` from HuggingFace (dev / online).

    ``sentinel`` is the marker file that signals "already populated". Defaults to
    ``model_index.json`` (diffusers pipeline repos); pass e.g. ``config.json``
    for a single-component repo (a standalone VAE) with no ``model_index.json``.
    """
    d = os.path.join(flat_root(cache_dir, namespace=namespace), repo)
    if cdn_base:
        # The CDN path pulls EVERY file in the repo manifest. A `.cdn_complete`
        # marker — written only AFTER the full pull — is the skip condition, NOT
        # the sentinel: a killed download can leave `model_index.json` + a partial
        # tree, and short-circuiting on the sentinel would then load a broken repo
        # (missing shards/index). Without the marker, re-run _cdn_snapshot, which
        # is idempotent (skips already-complete files, re-fetches the rest).
        if not os.path.exists(os.path.join(d, _CDN_COMPLETE)):
            try:
                _cdn_snapshot(cdn_base, repo, d)
            except Exception as e:  # noqa: BLE001
                # CDN gap (repo not mirrored -> 404 on its manifest) or any other
                # CDN failure: fall back to HuggingFace so a hole in the mirror
                # degrades to a slower pull instead of crashing the load. Mark
                # complete afterwards so later loads skip both the CDN re-try and
                # HF. (A set HF_HUB_OFFLINE=1 would make this raise — intended:
                # the caller opted into strict offline.)
                logger.warning(
                    "CDN fetch failed for %s (%s); falling back to HuggingFace", repo, e)
                from huggingface_hub import snapshot_download
                snapshot_download(repo, allow_patterns=patterns, local_dir=d)
                open(os.path.join(d, _CDN_COMPLETE), "w").close()
        return d
    # Fast path: our own completed pull left the marker -> fully populated, skip.
    if os.path.exists(os.path.join(d, _HF_COMPLETE)):
        return d

    sentinel_present = os.path.exists(os.path.join(d, sentinel))
    # Offline: we can't (re)download, so trust a sentinel-bearing tree — this is
    # the base-tarball / pre-shipped contract (a truncated offline tree can't be
    # repaired without network anyway, so behaviour is unchanged there).
    if sentinel_present and os.environ.get("HF_HUB_OFFLINE") in ("1", "true", "True"):
        return d

    # Online: run snapshot_download even when the sentinel is present. It is
    # idempotent — it verifies each file and fills only the missing/incomplete
    # ones, so a killed pull that left the sentinel + partial shards is REPAIRED
    # here instead of loading broken. Write the completion marker only after it
    # returns, so the fast path above is safe on the next load.
    from huggingface_hub import snapshot_download
    snapshot_download(repo, allow_patterns=patterns, local_dir=d)
    open(os.path.join(d, _HF_COMPLETE), "w").close()
    return d


def write_manifest(repo_dir: str, *, name: str = ".manifest.json") -> list:
    """Write the CDN manifest for a flat repo dir and return the file list.

    The manifest is a JSON list of repo-relative, forward-slash file paths — the
    exact list ``_cdn_snapshot`` reads back to populate a tree from the CDN. Kept
    next to the reader so writer/reader can't drift. Excludes the manifest file
    itself, plus hidden dirs/files: ``snapshot_download(local_dir=...)`` drops a
    ``.cache/huggingface/`` tree (download metadata + locks) inside the repo dir,
    and repos ship a ``.gitattributes`` — none are real model components and they
    must not land in the manifest (the CDN doesn't host them -> 404 at cold load).
    """
    import json

    files = []
    for dirpath, dirnames, filenames in os.walk(repo_dir):
        # Prune hidden subtrees in place (skips .cache/huggingface, etc.).
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn == name or fn.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), repo_dir)
            files.append(rel.replace(os.sep, "/"))
    files.sort()
    with open(os.path.join(repo_dir, name), "w") as f:
        json.dump(files, f, indent=0)
    return files


def _cdn_threads() -> int:
    """Parallel HTTP streams for CDN pulls. ``IMAGE_CDN_THREADS`` (image side)
    aliases ``WAN_CDN_THREADS``; default 8."""
    return int(os.environ.get("IMAGE_CDN_THREADS")
               or os.environ.get("WAN_CDN_THREADS") or "8")


def _cdn_snapshot(cdn_base: str, repo: str, dest_dir: str) -> None:
    """Populate a flat local repo dir from a CDN mirror via its manifest.

    Reads ``<cdn_base>/<repo>/.manifest.json`` (a JSON list of repo-relative file
    paths) and pulls each file to ``<dest_dir>/<rel>`` with the parallel
    downloader. Idempotent: already-present files are skipped, so a partially
    warmed tree resumes cleanly. The manifest is emitted by the workers'
    ``prefetch.py`` when staging the R2 mirror.
    """
    import json
    from urllib.request import Request, urlopen

    from imference_engine.runtime.download import download_parallel

    base = f"{cdn_base.rstrip('/')}/{repo}"
    manifest_url = f"{base}/.manifest.json"
    logger.info("  CDN manifest %s", manifest_url)
    with urlopen(Request(manifest_url, headers={"User-Agent": "imference-engine"})) as r:
        files = json.load(r)

    threads = _cdn_threads()
    for rel in files:
        dest = os.path.join(dest_dir, *rel.split("/"))
        if os.path.exists(dest):
            continue
        download_parallel(f"{base}/{rel}", dest, threads)
    # Mark complete ONLY after every file landed — a killed pull leaves no marker,
    # so the next call re-enters here and finishes the partial tree.
    open(os.path.join(dest_dir, _CDN_COMPLETE), "w").close()


def cdn_download(
    cdn_base: str,
    repo: str,
    filename: str,
    cache_dir: Optional[str],
    *,
    namespace: str,
    threads_env: str = "WAN_CDN_THREADS",
) -> str:
    """Fetch ``<cdn_base>/<repo>/<filename>`` into the flat tree (skip if present).

    Uses the parallel multi-stream downloader (plain HTTP), so it works alongside
    ``HF_HUB_OFFLINE=1`` and saturates the CDN link.
    """
    dest = os.path.join(flat_root(cache_dir, namespace=namespace), repo, filename)
    if not os.path.exists(dest):
        from imference_engine.runtime.download import download_parallel
        url = f"{cdn_base.rstrip('/')}/{repo}/{filename}"
        logger.info("  CDN fetch %s", url)
        download_parallel(url, dest, int(os.environ.get(threads_env, "8")))
    return dest


def fetch(
    repo: str,
    filename: str,
    cdn_base: Optional[str],
    cache_dir: Optional[str],
    *,
    namespace: str,
) -> str:
    """Local path for a model file — into the FLAT tree, from the CDN if set else HF.

    Both paths land at ``<flat_root>/<repo>/<filename>`` (no symlinks), so the
    whole tree is portable + shippable + offline once present.
    """
    if cdn_base:
        return cdn_download(cdn_base, repo, filename, cache_dir, namespace=namespace)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo, filename,
        local_dir=os.path.join(flat_root(cache_dir, namespace=namespace), repo))
