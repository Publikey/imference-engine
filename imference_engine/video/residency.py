"""Generic residency manager for video pipelines — arch-agnostic (Phase 4c).

Keeps up to ``max_resident`` built pipelines warm in CPU RAM (LRU) and drives all
build/teardown/shared-component work through a ``VideoBackend``. No architecture
specifics live here — Wan, LTX, etc. each supply their own backend.

Video residency is simpler than the image ``ModelManager``: ``enable_offload``
keeps VRAM at ~one submodel no matter how many pipelines are resident, so there is
no GPU-LRU tier — only CPU-RAM residency.
"""
from __future__ import annotations

import gc
import logging
from collections import OrderedDict
from typing import Any, Callable, Optional

from imference_engine.video.backend import VideoBackend, VideoBuildContext

logger = logging.getLogger(__name__)


class ResidencyManager:
    """Keeps up to ``max_resident`` pipelines warm in CPU RAM (LRU), built via a
    ``VideoBackend``."""

    def __init__(
        self,
        *,
        backend: VideoBackend,
        ctx: VideoBuildContext,
        max_resident: int,
        on_evicted: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._backend = backend
        self._ctx = ctx
        self._max_resident = max(1, max_resident)
        self._on_evicted = on_evicted
        self._shared: Optional[Any] = None
        self._pipes: "OrderedDict[str, Any]" = OrderedDict()

    def _shared_components(self) -> Any:
        # One set per backend, reused by every variant of that arch.
        if self._shared is None:
            self._shared = self._backend.load_shared(self._ctx)
        return self._shared

    def get_or_load(self, spec) -> Any:
        """Return a resident pipeline for ``spec``, building + caching if needed."""
        name = spec.name
        if name in self._pipes:
            self._pipes.move_to_end(name)  # mark MRU
            return self._pipes[name]

        # Free RAM BEFORE building the new pipeline so a switch never holds two
        # variants resident at once (small-RAM OOM). Evict to (max-1), then build.
        self._evict_to(self._max_resident - 1)
        pipe = self._backend.build(spec, self._shared_components(), self._ctx)
        self._pipes[name] = pipe
        self._pipes.move_to_end(name)
        return pipe

    def _evict_to(self, target: int) -> None:
        """Evict LRU pipelines until at most ``target`` remain (frees RAM first)."""
        while len(self._pipes) > max(0, target):
            evicted_name, evicted_pipe = self._pipes.popitem(last=False)  # LRU
            logger.info("Evicting resident variant %r (LRU)", evicted_name)
            self._backend.teardown(evicted_pipe)
            evicted_pipe = None
            self._free_cache()
            if self._on_evicted:
                self._on_evicted(evicted_name)

    @staticmethod
    def _free_cache() -> None:
        gc.collect()
        gc.collect()  # second pass collects cycles freed by the first
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    @property
    def resident(self) -> list[str]:
        return list(self._pipes.keys())
