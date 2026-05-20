"""Device detection: cuda | mps | cpu.

Foundation for desktop / Apple Silicon support. The engine itself currently
assumes CUDA (workers run on Linux + NVIDIA); this module exists so the
desktop sidecar can swap in MPS / CPU without touching the rest of the engine.

Importing torch is deferred so this module can be loaded on machines without
torch installed (e.g. to inspect the catalog).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Device:
    kind: str  # "cuda" | "mps" | "cpu"
    index: int = 0

    @property
    def torch_str(self) -> str:
        """Render as a torch device string ('cuda:0', 'mps', 'cpu')."""
        if self.kind == "cuda":
            return f"cuda:{self.index}"
        return self.kind


def resolve_device(preference: str = "auto") -> Device:
    """Pick the best available device, honoring preference.

    preference: "auto" | "cuda" | "cuda:N" | "mps" | "cpu"
    Falls back gracefully when the preferred device isn't available.
    """
    pref = preference.strip().lower()

    try:
        import torch
    except ImportError:
        logger.warning("torch not installed; falling back to CPU")
        return Device(kind="cpu")

    def cuda_available() -> bool:
        return hasattr(torch, "cuda") and torch.cuda.is_available()

    def mps_available() -> bool:
        return (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

    if pref.startswith("cuda"):
        if cuda_available():
            idx = int(pref.split(":", 1)[1]) if ":" in pref else 0
            return Device(kind="cuda", index=idx)
        logger.warning("CUDA requested but unavailable; falling back to auto")
        pref = "auto"

    if pref == "mps":
        if mps_available():
            return Device(kind="mps")
        logger.warning("MPS requested but unavailable; falling back to auto")
        pref = "auto"

    if pref == "cpu":
        return Device(kind="cpu")

    # auto
    if cuda_available():
        return Device(kind="cuda")
    if mps_available():
        return Device(kind="mps")
    return Device(kind="cpu")
