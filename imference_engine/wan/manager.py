"""Residency manager for Wan pipelines — RAM-LRU + shared components, no GPU tier.

P1g showed the design is simpler than the image-side ``ModelManager``:
- ``enable_model_cpu_offload`` keeps VRAM at ~one expert no matter how many pipes
  are resident, so there is **no GPU-LRU tier** — only CPU-RAM residency.
- The UMT5 text_encoder + Wan VAE are shared by every variant (loaded once).

So this manager just keeps up to ``max_resident`` built pipelines warm in RAM and
evicts least-recently-used. Sizing: ~31 GB RAM per GGUF-Q8 variant + ~30 GB
generation working set (measured P1g).
"""
from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from typing import Any, Callable, Optional

from imference_engine.wan.loader import (SharedComponents, build_pipeline,
                                         load_shared_components)

logger = logging.getLogger(__name__)


class ResidencyManager:
    """Keeps up to ``max_resident`` Wan pipelines warm in CPU RAM (LRU)."""

    def __init__(
        self,
        *,
        quant: str,
        device: str,
        enable_offload: bool,
        vae_tiling: bool,
        max_resident: int,
        cache_dir: Optional[str] = None,
        cdn_base: Optional[str] = None,
        shared_base_repo: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        on_evicted: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._quant = quant
        self._device = device
        self._enable_offload = enable_offload
        self._vae_tiling = vae_tiling
        self._max_resident = max(1, max_resident)
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base
        self._shared_base_repo = shared_base_repo
        self._on_evicted = on_evicted
        self._shared: Optional[SharedComponents] = None
        self._pipes: "OrderedDict[str, Any]" = OrderedDict()

    def _shared_components(self) -> SharedComponents:
        # One global set, reused by every variant (P1g: T2V & I2V share these).
        if self._shared is None:
            self._shared = load_shared_components(
                self._shared_base_repo, cache_dir=self._cache_dir)
        return self._shared

    def get_or_load(self, variant) -> Any:
        """Return a resident pipeline for ``variant``, building + caching if needed."""
        name = variant.name
        if name in self._pipes:
            self._pipes.move_to_end(name)  # mark MRU
            return self._pipes[name]

        pipe = build_pipeline(
            variant,
            quant=self._quant,
            shared=self._shared_components(),
            device=self._device,
            enable_offload=self._enable_offload,
            vae_tiling=self._vae_tiling,
            cache_dir=self._cache_dir,
            cdn_base=self._cdn_base,
        )
        self._pipes[name] = pipe
        self._pipes.move_to_end(name)
        self._evict_if_needed()
        return pipe

    def _evict_if_needed(self) -> None:
        while len(self._pipes) > self._max_resident:
            evicted_name, evicted_pipe = self._pipes.popitem(last=False)  # LRU
            logger.info("Evicting resident variant %r (LRU)", evicted_name)
            del evicted_pipe
            self._free_cache()
            if self._on_evicted:
                self._on_evicted(evicted_name)

    @staticmethod
    def _free_cache() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def resident(self) -> list[str]:
        return list(self._pipes.keys())
