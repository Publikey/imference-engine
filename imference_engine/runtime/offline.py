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
) -> str:
    """Ensure a repo's needed files are in a FLAT local dir and return it.

    Uses ``snapshot_download(local_dir=...)``, which writes real files (NO
    symlinks) on every OS -> the tree is portable Windows<->Mac<->Linux, unlike
    the symlinked HF cache. Idempotent; offline once populated (``from_pretrained``
    then just reads local files).

    ``sentinel`` is the marker file that signals "already populated, skip any
    network/snapshot call". Defaults to ``model_index.json`` (diffusers pipeline
    repos); pass e.g. ``config.json`` for a single-component repo (a standalone
    VAE) that has no ``model_index.json``.
    """
    d = os.path.join(flat_root(cache_dir, namespace=namespace), repo)
    # Already populated (shipped tree)? Return without any network/snapshot call,
    # so a pre-shipped tree works fully offline.
    if os.path.exists(os.path.join(d, sentinel)):
        return d
    from huggingface_hub import snapshot_download
    snapshot_download(repo, allow_patterns=patterns, local_dir=d)
    return d


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
