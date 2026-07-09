"""Shared engine base for the image and video engines.

Phase 3 of the unified-engine-core refactor (docs/unified-engine-core.md). Owns
the scaffolding both engines share — device resolution, CUDA tuning, the
idempotent ``load()`` lifecycle, seed + device-cache helpers — behind a
``_setup()`` hook each subclass fills with its backend/manager construction.

Public class names stay per modality (decision 3b): ``Engine`` (image) and
``WanEngine`` (video) subclass this; ``BaseEngine`` itself is internal to ``core``.
"""
from __future__ import annotations

import gc
import logging
import random
from typing import Optional

from imference_engine.core.config import BaseRuntimeConfig
from imference_engine.runtime.device import Device, resolve_device

logger = logging.getLogger(__name__)

# numpy int32 max — keeps seeds in a range schedulers accept.
MAX_SEED = 2 ** 31 - 1


class BaseEngine:
    """Common lifecycle for the modality engines. Single-threaded and stateful;
    concurrent callers must serialize."""

    MAX_SEED = MAX_SEED

    def __init__(self, *, runtime: BaseRuntimeConfig) -> None:
        self._runtime = runtime
        self._device: Optional[Device] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "BaseEngine":
        """Resolve the device, tune CUDA, then run the subclass ``_setup()``.
        Idempotent."""
        if self._loaded:
            return self
        self._device = resolve_device(self._runtime.device)
        logger.info("%s device: %s", type(self).__name__, self._device.torch_str)
        if self._device.kind == "cuda":
            self._tune_cuda()
        self._setup()
        self._loaded = True
        return self

    def _setup(self) -> None:
        """Subclass hook: build backends/managers + emit modality-specific device
        warnings. Runs inside ``load()`` with ``self._device`` resolved and
        ``self._loaded`` still False."""

    @property
    def device_str(self) -> Optional[str]:
        """Torch device string ("cuda:0" | "mps" | "cpu"), or None before load()."""
        return self._device.torch_str if self._device else None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tune_cuda() -> None:
        """Enable TF32 + flash/mem-efficient SDPA on CUDA (free perf on Ampere+)."""
        try:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception as e:  # noqa: BLE001
            logger.debug("CUDA tuning skipped: %s", e)

    def _make_seed(self, seed: Optional[int]) -> int:
        """Resolve a single seed (random when None)."""
        return seed if seed is not None else random.randint(0, self.MAX_SEED)

    def _free_cache(self) -> None:
        gc.collect()
        try:
            import torch
            if self._device and self._device.kind == "cuda":
                torch.cuda.empty_cache()
            elif (self._device and self._device.kind == "mps"
                  and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache")):
                torch.mps.empty_cache()
        except ImportError:
            pass
