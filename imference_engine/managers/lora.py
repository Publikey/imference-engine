"""Image-side LoRA manager — resolve, cache, apply, deactivate.

Ported from the legacy ``sdxl-multimodel`` worker's LoRAManager (the design
production ran on) with the lessons the Wan loader learned since:

- **No fuse, ever**: adapters are applied with ``set_adapters`` and stay live
  over the base weights (same philosophy as ``wan/loader._apply_loras``, which
  dodges diffusers#12047). Crucial with the ModelManager's RESIDENT pipes — a
  fused LoRA would permanently alter a pipe that outlives the request.
- **Deactivate in finally**: after every request the active adapter set is
  cleared, so the next LoRA-less request gets clean output from the same
  resident pipe. Loaded adapters stay cached on the pipe for cheap reuse.
- **Offline-safe load form**: ``load_lora_weights(dir, weight_name=...)`` — a
  bare file path makes diffusers guess the name, which it refuses to do under
  ``HF_HUB_OFFLINE=1``.
- **Bookkeeping lives ON the resident pipe** (``pipe._imference_loras``):
  when the ModelManager evicts a pipe, the adapter cache dies with it — no
  stale per-model-name maps to reconcile. Always apply on the resident t2i
  pipe, never on the per-request img2img wrapper (which shares the same
  modules but is thrown away after the request).
- **Family check before load**: ``managers/lora_inspect`` reads the
  safetensors header and refuses a LoRA made for another family.

Accepted config shape (one dict per LoRA, list = stack)::

    {"source": "<local path | http(s) URL>", "weight": 0.8, "adapter_name": "style"}

``path`` and ``url`` are accepted as aliases of ``source`` (the legacy worker
payload used ``path``). ``weight`` defaults to 1.0; ``adapter_name`` is derived
from the file name when omitted. URL downloads land in ``cache_dir`` via the
engine's parallel downloader (multipart + atomic + retries) and are pruned
LRU-by-mtime beyond ``max_cached_files``.
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Attribute holding the per-pipe OrderedDict{adapter_name: source} cache.
_PIPE_ATTR = "_imference_loras"


class LoRAManager:
    def __init__(
        self,
        *,
        cache_dir: Optional[str] = None,
        max_adapters: int = 5,
        max_cached_files: int = 20,
    ) -> None:
        # Default the URL-download cache next to the offline model tree.
        if not cache_dir:
            from imference_engine.runtime.offline import flat_root
            cache_dir = os.path.join(flat_root(None, namespace="image"), "_loras")
        self._cache_dir = cache_dir
        self._max_adapters = max(1, max_adapters)
        self._max_cached_files = max(1, max_cached_files)

    # ------------------------------------------------------------------
    # Parsing / resolution
    # ------------------------------------------------------------------

    @staticmethod
    def parse(loras: Optional[list]) -> list[dict]:
        """Normalize request configs to ``[{source, weight, adapter_name}]``.
        Entries without a usable source are dropped (logged) rather than
        failing the whole request."""
        out: list[dict] = []
        for cfg in loras or []:
            if not isinstance(cfg, dict):
                logger.warning("Ignoring non-dict LoRA config: %r", cfg)
                continue
            source = cfg.get("source") or cfg.get("path") or cfg.get("url")
            if not source or not isinstance(source, str):
                logger.warning("Ignoring LoRA config without a source: %r", cfg)
                continue
            try:
                weight = float(cfg.get("weight", 1.0))
            except (TypeError, ValueError):
                logger.warning("Ignoring bad LoRA weight in %r; using 1.0", cfg)
                weight = 1.0
            name = cfg.get("adapter_name") or _derive_adapter_name(source)
            out.append({"source": source, "weight": weight, "adapter_name": name})
        return out

    def resolve(self, source: str) -> str:
        """Return a local ``.safetensors`` path for ``source`` — a local file
        as-is, an http(s) URL downloaded into the LRU-pruned cache dir."""
        if os.path.isfile(source):
            return source

        if source.startswith(("http://", "https://")):
            filename = _cache_filename(source)
            local = os.path.join(self._cache_dir, filename)
            if os.path.isfile(local):
                os.utime(local, None)  # touch: LRU freshness for the prune
                logger.info("LoRA cached locally: %s", local)
                return local
            os.makedirs(self._cache_dir, exist_ok=True)
            logger.info("Downloading LoRA %s -> %s", source, local)
            self._download(source, local)
            self._prune_cache()
            return local

        raise FileNotFoundError(
            f"LoRA source {source!r} is neither an existing file nor an http(s) URL"
        )

    def _download(self, url: str, dest: str) -> None:
        """Seam for tests; multipart + atomic + retries via the engine downloader."""
        from imference_engine.runtime.download import download_parallel
        download_parallel(url, dest)

    def _prune_cache(self) -> None:
        """Drop the oldest downloaded files beyond ``max_cached_files``."""
        try:
            entries = [
                (os.path.getmtime(p), p)
                for f in os.listdir(self._cache_dir)
                if (p := os.path.join(self._cache_dir, f)) and os.path.isfile(p)
            ]
        except OSError:
            return
        entries.sort()  # oldest first
        for _, path in entries[: max(0, len(entries) - self._max_cached_files)]:
            logger.info("Pruning LRU LoRA file: %s", path)
            try:
                os.remove(path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Apply / deactivate
    # ------------------------------------------------------------------

    def apply(self, pipe: Any, configs: list[dict], *, family: Optional[str] = None) -> None:
        """Load (or reuse) each adapter on the pipe and activate the stack.

        ``pipe`` must be the RESIDENT pipe (not a per-request img2img wrapper):
        the adapter bookkeeping lives on it, so it has to be the object the
        next request sees.

        ``family`` is the backend's engine id; when given, each file is checked
        with :func:`inspect_lora` and a LoRA positively identified for another
        family is refused with a readable error (instead of a diffusers key
        mismatch deep in ``load_lora_weights``).

        Raises on a failed load — a request that asked for a LoRA must not
        silently render without it. Caching: adapters already on the pipe are
        reused; beyond ``max_adapters`` the least-recently-used is deleted.
        """
        if not configs:
            return
        if len(configs) > self._max_adapters:
            # Loading more than the cache holds would evict adapters of this
            # very request before set_adapters activates them.
            raise ValueError(
                f"{len(configs)} LoRAs requested but at most {self._max_adapters} "
                f"can be active (MAX_CACHED_LORAS)"
            )
        loaded: "OrderedDict[str, str]" = getattr(pipe, _PIPE_ATTR, None) or OrderedDict()
        setattr(pipe, _PIPE_ATTR, loaded)

        names: list[str] = []
        weights: list[float] = []
        for cfg in configs:
            name, weight, source = cfg["adapter_name"], cfg["weight"], cfg["source"]
            names.append(name)
            weights.append(weight)

            if name in loaded:
                if loaded[name] == source:
                    loaded.move_to_end(name)
                    logger.info("LoRA %r already loaded on pipe; reusing (weight=%s)", name, weight)
                    continue
                # Same adapter name, different file: reusing would silently
                # render with the wrong weights — drop the old one and reload.
                logger.info("LoRA %r now points to %s; replacing the loaded adapter", name, source)
                del loaded[name]
                _delete_adapter(pipe, name)

            while len(loaded) >= self._max_adapters:
                evict, _ = loaded.popitem(last=False)
                logger.info("Evicting LRU LoRA adapter %r", evict)
                _delete_adapter(pipe, evict)

            path = self.resolve(source)
            if family is not None:
                self._check_family(name, path, family)
            logger.info("Loading LoRA %r from %s (weight=%s)", name, path, weight)
            # (dir, weight_name) form — offline-safe (see module docstring).
            try:
                pipe.load_lora_weights(
                    os.path.dirname(path),
                    weight_name=os.path.basename(path),
                    adapter_name=name,
                )
            except Exception:
                # A failed load can leave a half-injected adapter under this
                # name; without cleanup the next request would hit "adapter
                # name already in use".
                _delete_adapter(pipe, name)
                raise
            loaded[name] = source

        pipe.set_adapters(names, adapter_weights=weights)
        logger.info("Active LoRAs (not fused): %s", list(zip(names, weights)))

    @staticmethod
    def _check_family(name: str, path: str, family: str) -> None:
        from imference_engine.managers.lora_inspect import inspect_lora, is_compatible

        info = inspect_lora(path)  # ValueError on a non-safetensors file
        if not is_compatible(info.family, family):
            raise ValueError(
                f"LoRA {name!r} targets {info.family} (from {info.reason}) "
                f"but the model is {family}"
            )
        logger.info("LoRA %r family: %s (%s)", name, info.family or "unknown", info.reason)

    @staticmethod
    def deactivate(pipe: Any) -> None:
        """Clear the ACTIVE adapter set after a request (adapters stay cached on
        the pipe for reuse). Never raises — this runs in a finally."""
        try:
            pipe.set_adapters([])
        except Exception:  # noqa: BLE001
            try:
                pipe.disable_lora()
            except Exception:  # noqa: BLE001
                logger.warning("Could not deactivate LoRA adapters on pipe")


def _delete_adapter(pipe: Any, name: str) -> None:
    """Best-effort ``delete_adapters`` — eviction/cleanup must never mask the
    request's real outcome."""
    try:
        pipe.delete_adapters(name)
    except Exception as e:  # noqa: BLE001
        logger.warning("delete_adapters(%r) failed: %s", name, e)


def _derive_adapter_name(source: str) -> str:
    base = os.path.basename(source.split("?", 1)[0])
    name = os.path.splitext(base)[0]
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in name).strip("_").lower()
    return name or "lora"


def _cache_filename(url: str) -> str:
    """Stable cache file name for a URL: sanitized basename + a short hash so
    two different URLs with the same basename don't collide."""
    import hashlib

    base = os.path.basename(url.split("?", 1)[0]) or "lora"
    if not base.endswith(".safetensors"):
        base += ".safetensors"
    stem, ext = os.path.splitext(base)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
    return f"{safe}-{digest}{ext}"
