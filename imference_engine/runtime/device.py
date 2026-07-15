"""Device detection: cuda (NVIDIA + AMD ROCm) | mps | cpu.

Foundation for desktop / Apple Silicon / AMD support. Workers run on Linux +
NVIDIA, but the same "cuda" kind transparently covers AMD GPUs: PyTorch's ROCm
build masquerades as CUDA (``torch.cuda.is_available()`` is True, device
strings stay ``cuda:N``), with ``torch.version.hip`` set as the tell. Use
``is_rocm()`` / ``Device.backend`` when a log line or UI needs to distinguish
the vendor — everything else can keep branching on ``kind == "cuda"``.

Importing torch is deferred so this module can be loaded on machines without
torch installed (e.g. to inspect the catalog).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def is_rocm() -> bool:
    """True when the installed torch is an AMD ROCm/HIP build.

    ROCm torch reports ``torch.version.hip`` (e.g. "6.4.x") while CUDA builds
    leave it None. False when torch isn't installed at all.
    """
    try:
        import torch
    except ImportError:
        return False
    return bool(getattr(getattr(torch, "version", None), "hip", None))


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

    @property
    def backend(self) -> str:
        """Vendor-aware label for logs/UI: 'rocm' on an AMD (HIP) torch build,
        otherwise same as ``kind``. Never feed this to torch APIs — the torch
        device string for ROCm is still ``cuda:N`` (use ``torch_str``)."""
        if self.kind == "cuda" and is_rocm():
            return "rocm"
        return self.kind


def resolve_device(preference: str = "auto") -> Device:
    """Pick the best available device, honoring preference.

    preference: "auto" | "cuda" | "cuda:N" | "mps" | "cpu"
    ("cuda" selects the GPU on both NVIDIA and AMD ROCm builds of torch.)
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

    def cuda_device(index: int = 0) -> Device:
        if is_rocm():
            logger.info("CUDA device is backed by ROCm/HIP (AMD GPU)")
        return Device(kind="cuda", index=index)

    if pref.startswith("cuda"):
        if cuda_available():
            idx = int(pref.split(":", 1)[1]) if ":" in pref else 0
            return cuda_device(idx)
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
        return cuda_device()
    if mps_available():
        return Device(kind="mps")
    return Device(kind="cpu")
