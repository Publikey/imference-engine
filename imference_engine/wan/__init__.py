"""Wan2.2 video sub-package.

Self-contained engine for Wan2.2 text-to-video / image-to-video via diffusers,
using GGUF-quantized MoE experts + the Lightning 4-step LoRA (validated by the
spikes in ``spikes/wan22/``). Shares only ``imference_engine.runtime.device``
with the image side; does not touch the SDXL/Z-Image pipeline abstraction.

    from imference_engine.wan import WanEngine, WanRuntimeConfig, MemoryProfile

    engine = WanEngine(runtime=WanRuntimeConfig(max_resident_variants=4)).load()
    res = engine.generate_video(variant="wan22-t2v-lightning", prompt="a red fox...")
"""
from imference_engine.core.result import GenerationError, MediaResult
from imference_engine.wan.config import MemoryProfile, WanRuntimeConfig
from imference_engine.wan.engine import WanEngine
from imference_engine.wan.presets import WanLora, WanVariant

__all__ = [
    "WanEngine",
    "WanRuntimeConfig",
    "MemoryProfile",
    "WanVariant",
    "WanLora",
    "MediaResult",
    "GenerationError",
]
