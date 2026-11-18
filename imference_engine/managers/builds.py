"""Per-machine build selection shared by the backends that ship several builds.

A *build* is a combination measured as a whole: which transformer file
(bf16 / GGUF level), how the text encoder is stored, and which offload mode —
with the VRAM it needs. Each backend package keeps its own measured table
(``flux2klein/builds.py``, ``ernieimage/builds.py``); this module holds the
record and the picker.

Pure Python on purpose (no torch import, like ``managers/batch.py``): the
caller measures the VRAM and passes it in, so the tables are unit-testable
without a GPU and usable from a launcher that has not loaded torch yet (the
Desktop sidecar deciding what to download).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

# weights_path sentinel: let the backend pick + fetch the build for this machine.
AUTO = "auto"


@dataclass(frozen=True)
class ModelBuild:
    """One validated transformer / encoder / offload combination."""

    name: str
    repo: str
    """Repo holding the transformer file (HuggingFace id = CDN sub-path)."""
    filename: str
    encoder: str
    """Text-encoder storage: ``"bf16"`` or ``"fp8"`` (float8 storage, bf16 compute)."""
    offload: Optional[str]
    """``None`` (everything resident), ``"model"`` or ``"group"`` — the
    ``RuntimeConfig.offload_mode`` to run this build with."""
    min_vram_gb: float
    """Smallest total VRAM (GiB) this build is recommended for."""
    peak_vram_gb: float
    """Measured peak allocation at 1024x1024."""
    reason: str = ""
    transformer: str = ""
    """Transformer storage for backends whose builds share ONE weights file and
    differ in how it is held in memory: ``"bf16"`` or ``"fp8"`` (float8 storage,
    bf16 compute). Empty when the file itself is the choice (GGUF levels)."""

    def runtime_kwargs(self) -> dict:
        """The ``RuntimeConfig`` fields this build needs
        (``RuntimeConfig(**build.runtime_kwargs(), ...)``)."""
        if self.offload is None:
            return {"enable_offload": False}
        return {"enable_offload": True, "offload_mode": self.offload}


def pick_build(builds: Sequence[ModelBuild], vram_gb: Optional[float],
               fallback: ModelBuild) -> ModelBuild:
    """The first build of ``builds`` (ordered best-first) whose ``min_vram_gb``
    fits a card with ``vram_gb`` GiB of TOTAL VRAM.

    ``None`` (no CUDA device, or the caller could not measure) returns
    ``fallback``. Pass total, not free, VRAM: the tables already keep a margin,
    and the choice must not flip between two loads on the same machine.
    """
    if vram_gb is None:
        return fallback
    for b in builds:
        if vram_gb >= b.min_vram_gb:
            return b
    return builds[-1]


def build_named(builds: Sequence[ModelBuild], name: str) -> ModelBuild:
    for b in builds:
        if b.name == name:
            return b
    raise KeyError(f"unknown build {name!r}; known: {[b.name for b in builds]}")


def cuda_vram_gb() -> Optional[float]:
    """Total VRAM of the current CUDA device in GiB; None without CUDA (or
    without torch — the import is lazy, this module stays torch-free)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1 << 30)
    except Exception:  # noqa: BLE001 — unknown hardware: no opinion
        return None
