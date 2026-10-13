"""Peak memory and timing: host RSS sampled in a thread, CUDA peaks from torch."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Stage:
    name: str
    seconds: float = 0.0
    vram_peak_gb: float = 0.0
    ram_peak_gb: float = 0.0


@dataclass
class Meter:
    """Collects per-stage wall time, CUDA peak allocated/reserved and host RSS peak."""

    interval: float = 0.05
    stages: list = field(default_factory=list)
    ram_peak_gb: float = 0.0
    vram_peak_gb: float = 0.0
    vram_reserved_peak_gb: float = 0.0

    def __post_init__(self):
        self._stop = threading.Event()
        self._cur = 0.0
        self._thread = None

    def _rss(self) -> float:
        try:
            import psutil
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / 1e9
        except Exception:
            return 0.0

    def _sample(self):
        while not self._stop.is_set():
            r = self._rss()
            self._cur = max(self._cur, r)
            self.ram_peak_gb = max(self.ram_peak_gb, r)
            self._stop.wait(self.interval)

    def start(self) -> "Meter":
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    @contextmanager
    def stage(self, name: str):
        cuda = _cuda()
        if cuda:
            cuda.synchronize()
            cuda.reset_peak_memory_stats()
        self._cur = self._rss()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                cuda.synchronize()
            st = Stage(name=name, seconds=time.perf_counter() - t0, ram_peak_gb=max(self._cur, self._rss()))
            if cuda:
                st.vram_peak_gb = cuda.max_memory_allocated() / 1e9
                self.vram_peak_gb = max(self.vram_peak_gb, st.vram_peak_gb)
                self.vram_reserved_peak_gb = max(self.vram_reserved_peak_gb, cuda.max_memory_reserved() / 1e9)
            self.ram_peak_gb = max(self.ram_peak_gb, st.ram_peak_gb)
            self.stages.append(st)

    def to_dict(self) -> dict:
        return {
            "ram_peak_gb": round(self.ram_peak_gb, 2),
            "vram_peak_gb": round(self.vram_peak_gb, 2),
            "vram_reserved_peak_gb": round(self.vram_reserved_peak_gb, 2),
            "stages": [{"name": s.name, "seconds": round(s.seconds, 2), "vram_peak_gb": round(s.vram_peak_gb, 2),
                        "ram_peak_gb": round(s.ram_peak_gb, 2)} for s in self.stages],
        }


def _cuda():
    try:
        import torch
        return torch.cuda if torch.cuda.is_available() else None
    except Exception:
        return None


def free_memory() -> None:
    import gc
    gc.collect()
    c = _cuda()
    if c:
        c.empty_cache()
        try:
            c.ipc_collect()
        except Exception:
            pass
