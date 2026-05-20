"""Dynamic GPU batch sizing.

Profiles VRAM on the first inference per (engine, is_img2img) tuple. Subsequent
calls chunk the requested batch into GPU-optimal sub-batches. On OOM, halves
the cached max and retries.

This module is intentionally pure — no torch imports. Callers pass in
VRAM measurements they collect from their own torch.cuda calls. This keeps
the module unit-testable without a GPU and ready for future MPS / CPU paths
where VRAM measurement looks different.
"""
from __future__ import annotations
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)


class BatchSizer:
    """Per-process batch-size cache, keyed by (engine, is_img2img)."""

    VRAM_RESERVE_MB = int(os.getenv("BATCH_VRAM_RESERVE_MB", "512"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))

    def __init__(self) -> None:
        self._cache: dict[Tuple[str, bool], int] = {}
        self._oom_overrides: dict[Tuple[str, bool], int] = {}

    @staticmethod
    def _key(engine: str, is_img2img: bool) -> Tuple[str, bool]:
        return (engine, is_img2img)

    def get_max_batch_size(self, engine: str, is_img2img: bool = False) -> int:
        """Return cached max batch for (engine, mode); 0 means needs profiling."""
        key = self._key(engine, is_img2img)
        if key in self._oom_overrides:
            return self._oom_overrides[key]
        return self._cache.get(key, 0)

    def profile_and_compute(
        self,
        engine: str,
        is_img2img: bool,
        *,
        total_vram: int,
        baseline_vram: int,
        peak_vram: int,
    ) -> int:
        """After a single-image inference, compute and cache max batch size."""
        key = self._key(engine, is_img2img)

        per_image_vram = peak_vram - baseline_vram
        if per_image_vram <= 0:
            logger.warning(
                f"BatchSizer: per-image VRAM <= 0 ({per_image_vram}), defaulting to 1"
            )
            self._cache[key] = 1
            return 1

        reserve = self.VRAM_RESERVE_MB * 1024 * 1024
        available = total_vram - baseline_vram - reserve
        max_batch = 1 if available <= 0 else max(1, int(available / per_image_vram))
        max_batch = min(max_batch, self.MAX_BATCH_SIZE)

        self._cache[key] = max_batch
        logger.info(
            f"BatchSizer: engine={engine} img2img={is_img2img} "
            f"per_image={per_image_vram / 1e6:.0f}MB "
            f"baseline={baseline_vram / 1e6:.0f}MB "
            f"peak={peak_vram / 1e6:.0f}MB "
            f"total_vram={total_vram / 1e6:.0f}MB "
            f"max_batch={max_batch}"
        )
        return max_batch

    def report_oom(self, engine: str, is_img2img: bool, attempted_batch: int) -> int:
        """Record an OOM; halves the cached max for (engine, mode). Floor 1."""
        key = self._key(engine, is_img2img)
        new_max = max(1, attempted_batch // 2)
        self._oom_overrides[key] = new_max
        self._cache[key] = new_max
        logger.warning(
            f"BatchSizer OOM: engine={engine} batch={attempted_batch} -> {new_max}"
        )
        return new_max
