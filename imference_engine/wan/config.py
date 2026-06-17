"""Runtime configuration for the Wan video sub-package.

Kept separate from the image-side ``RuntimeConfig`` (in ``imference_engine.engine``)
because video residency works differently: the GPU never holds more than the
active MoE expert (``enable_model_cpu_offload`` handles that), so there is no
GPU-LRU tier — the LRU is purely about how many pipelines stay resident in CPU
RAM (validated by spike P1g).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class MemoryProfile(str, Enum):
    """How weights are loaded / where they live.

    GGUF_Q8 is the validated default (spike P1d/P1e): quality ≈ bf16, ~17 GB
    VRAM peak, low load-RAM (pre-quantized on disk). Lighter GGUF quants trade
    quality for VRAM; BF16 is the reference-quality fallback (needs ~75 GB RAM
    to load the A14B and ~30 GB VRAM).
    """

    GGUF_Q8 = "gguf_q8"
    GGUF_Q6 = "gguf_q6"   # Q6_K — smaller, slight quality loss
    GGUF_Q5 = "gguf_q5"   # Q5_K_M
    GGUF_Q4 = "gguf_q4"   # Q4_K_M — tightest cards
    BF16 = "bf16"         # reference quality, heavy

    @property
    def gguf_quant(self) -> Optional[str]:
        """The GGUF filename quant label, or None for non-GGUF profiles."""
        return {
            MemoryProfile.GGUF_Q8: "Q8_0",
            MemoryProfile.GGUF_Q6: "Q6_K",
            MemoryProfile.GGUF_Q5: "Q5_K_M",
            MemoryProfile.GGUF_Q4: "Q4_K_M",
        }.get(self)

    @property
    def is_gguf(self) -> bool:
        return self.gguf_quant is not None

    @classmethod
    def for_vram(cls, total_gb: float) -> "MemoryProfile":
        """Pick a GGUF quant for a GPU's total VRAM (measured P3 peaks + headroom).

        Q8 ~17 GB peak, Q6 ~13 GB, Q4 ~11 GB. Below Q6 the VRAM floor barely
        drops (activations/VAE dominate) while quality falls, so we jump Q6→Q4.
        """
        if total_gb >= 20:
            return cls.GGUF_Q8   # 24 GB+ — reference quality
        if total_gb >= 14:
            return cls.GGUF_Q6   # 16 GB — recommended, minimal quality loss
        return cls.GGUF_Q4       # 12 GB — best effort

    @classmethod
    def for_ram(cls, total_gb: float) -> "MemoryProfile":
        """Pick a GGUF quant for the host's CPU RAM.

        Under ``enable_model_cpu_offload`` the experts live in RAM, so RAM — not
        just VRAM — caps the quant. Measured: a resident variant + generation
        working set is ~61 GB at Q8, ~54 GB at Q6, ~48 GB at Q4 (P1g). A box with
        big VRAM but small RAM (e.g. 24 GB GPU / 50 GB RAM) OOMs on Q8, so the auto
        profile must clamp by RAM too. Thresholds keep working-set headroom and
        stay on uploaded quants (Q8/Q6/Q4, never Q5).
        """
        if total_gb >= 64:
            return cls.GGUF_Q8
        if total_gb >= 56:
            return cls.GGUF_Q6
        return cls.GGUF_Q4



@dataclass
class WanRuntimeConfig:
    """Runtime knobs for ``WanEngine``. All optional — defaults are safe."""

    device: str = "auto"
    """auto | cuda | cuda:N | cpu (mps untested for Wan)."""

    memory_profile: Union[MemoryProfile, str] = MemoryProfile.GGUF_Q8
    """Default weight format, or "auto" to pick the GGUF quant from the GPU's
    VRAM at load() (Q8 ≥20 GB, Q6 ≥14 GB, else Q4)."""

    enable_offload: bool = True
    """enable_model_cpu_offload on each pipe. Keeps VRAM at ~one expert (~17 GB)
    and is what makes multi-residency cheap (P1g). Disable only on a GPU big
    enough to hold a whole variant resident (rare)."""

    max_resident_variants: Optional[int] = 1
    """How many built pipelines to keep warm in CPU RAM (LRU). ~31 GB RAM per
    GGUF-Q8 variant + ~30 GB generation working set — size from the host's RAM.
    None or 1 = single-resident (desktop default)."""

    model_cache_dir: Optional[Union[str, Path]] = None
    """HF cache dir (maps to HF_HOME). Put on a big volume — GGUF experts are
    ~15 GB each. None = default HF cache."""

    model_cdn: Optional[str] = None
    """Base URL of a CDN mirroring `<repo>/<filename>` paths. When set, GGUF
    experts + LoRAs are fetched on demand from the CDN instead of HuggingFace
    (the shared base still loads from the cached HF repo). None = use HF."""

    vae_tiling: bool = True
    """Enable VAE tiling + slicing (cuts decode VRAM for long/large videos)."""
