"""Runtime configuration for the MiniMax-H3 video sub-package.

Separate from ``WanRuntimeConfig`` because the knobs differ fundamentally: H3 is
one dense 33B DiT + a 32B Qwen3-VL conditioner loaded through torchao int8 (or
bf16), not a GGUF-MoE — the memory levers are the quant profile and the *group
offload* granularity, not a GGUF filename quant. Same contract style as Wan:
every knob settable identically by env var (``H3_*``) or constructor param.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from imference_engine.core.config import BaseRuntimeConfig


class H3MemoryProfile(str, Enum):
    """How the two large components (transformer + Qwen3-VL) are loaded.

    INT8 is the intended default (target hardware: 24 GB VRAM / 64 GB RAM):
    torchao ``Int8WeightOnlyConfig(version=2)`` weight-only on both, either
    applied at load from the bf16 checkpoint or — much faster to cold-load —
    read from a pre-quantized mirror staged with ``validation/stage_h3_int8.py``.
    BF16 is reference quality and effectively multi-GPU-only (~124 GB weights).
    """

    INT8 = "int8"
    BF16 = "bf16"

    @classmethod
    def for_hardware(cls, vram_gb: float) -> "H3MemoryProfile":
        """The auto pick: int8 everywhere a single card is involved. bf16 is an
        explicit opt-in (two 80 GB cards), never auto-picked — even an 80 GB
        card runs int8 more comfortably than streamed bf16."""
        del vram_gb  # one branch today; the signature leaves room for an fp8 tier
        return cls.INT8


# Offload granularity (doc'd recipes from the upstream integration):
#   "block" — transformer group-offloaded block-by-block with a CUDA stream +
#             text encoder leaf-level; video/audio VAEs resident. 24-32 GB cards.
#   "leaf"  — same, plus the video VAE leaf-offloaded (no stream). 12-16 GB
#             cards with a reduced canvas (e.g. 960x544).
#   "none"  — everything resident on the device (multi-GPU / 80 GB+).
OFFLOAD_MODES = ("block", "leaf", "none")


@dataclass
class MiniMaxH3RuntimeConfig(BaseRuntimeConfig):
    """Runtime knobs for ``MiniMaxH3Engine``. Inherits ``device`` /
    ``model_cache_dir`` / ``model_cdn`` / ``enable_offload`` from
    ``BaseRuntimeConfig``; ``enable_offload`` is derived from ``offload_mode``
    at load() (kept only for base-class compatibility)."""

    memory_profile: Union[H3MemoryProfile, str] = H3MemoryProfile.INT8
    """"int8" | "bf16", or "auto" to resolve at load() (always int8 today)."""

    offload_mode: str = "auto"
    """"block" | "leaf" | "none", or "auto": "none" at >=80 GB VRAM, "leaf"
    below 20 GB, else "block". Weights live in host RAM under "block"/"leaf" —
    budget ~75 GB of it at int8."""

    max_resident_variants: Optional[int] = 1
    """How many built pipelines stay warm (LRU). One H3 pipeline is ~75 GB of
    host RAM at int8 — leave this at 1 unless the box is huge."""

    vae_tiling: bool = True
    """Enable video-VAE tiling/slicing when the loaded class supports it
    (best-effort; cuts decode VRAM on large canvases)."""

    attention_backend: Optional[str] = None
    """Optional diffusers attention backend for the transformer (e.g.
    "_flash_3_hub" on Hopper — kernels fetched from the Hub). None = SDPA."""

    @classmethod
    def from_env(cls) -> "MiniMaxH3RuntimeConfig":
        """Build a config from the ``H3_*`` env contract. Safe with no
        environment; workers override residency with cgroup-aware values, same
        as Wan."""
        from imference_engine.runtime.env import env_bool, env_int_or_none, env_str
        return cls(
            device=env_str("H3_DEVICE", "auto"),
            memory_profile=env_str("H3_PROFILE", "auto"),
            offload_mode=env_str("H3_OFFLOAD_MODE", "auto"),
            max_resident_variants=env_int_or_none("H3_MAX_RESIDENT") or 1,
            model_cache_dir=env_str("H3_MODEL_CACHE"),
            model_cdn=env_str("H3_MODEL_CDN"),
            vae_tiling=env_bool("H3_VAE_TILING", True),
            attention_backend=env_str("H3_ATTENTION_BACKEND"),
        )
