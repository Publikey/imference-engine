"""Video backend ABC — one subclass per video architecture (wan, ltx, ...).

Phase 4 of the unified-engine-core refactor (docs/unified-engine-core.md §8).
Mirrors the image ``PipelineBackend``: encapsulates how to assemble a pipeline
for a variant and how to call it for frames, so a new video architecture is a new
subclass rather than a fork of the engine. The generic residency manager (Phase
4c) drives these methods; the video engine routes to a backend by ``arch``.
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from imference_engine.core.backend import Backend


@dataclass
class VideoBuildContext:
    """The small builder-config a backend needs to assemble a pipeline —
    derived from ``WanRuntimeConfig`` by the engine, arch-agnostic."""

    quant: str
    device: str
    enable_offload: bool
    vae_tiling: bool
    cache_dir: Optional[str] = None
    cdn_base: Optional[str] = None
    text_encoder_quant: str = "none"
    # Optional arch-specific knobs (defaulted so existing backends are untouched).
    offload_mode: Optional[str] = None
    """Offload granularity for backends that distinguish more than on/off
    (MiniMax-H3: "block" | "leaf" | "none"). None -> the backend's default;
    ``enable_offload`` stays the coarse on/off signal for backends that only
    know that much (Wan)."""
    attention_backend: Optional[str] = None
    """Transformer attention backend override (diffusers
    ``set_attention_backend`` name, e.g. "_flash_3_hub" on Hopper). None -> keep
    the default SDPA path."""


class VideoBackend(Backend):
    """Assemble + call one video architecture. Extends the shared ``Backend``
    trunk (``engine`` id + ``engine_defaults``)."""

    @abstractmethod
    def load_shared(self, ctx: VideoBuildContext) -> Any:
        """Load the architecture's shared components (text encoder + VAE), reused
        by every variant. Called once and cached by the residency manager."""

    @abstractmethod
    def build(self, spec: Any, shared: Any, ctx: VideoBuildContext) -> Any:
        """Build a ready-to-run pipeline for ``spec`` (a variant), reusing the
        shared components."""

    @abstractmethod
    def build_call(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        num_frames: int,
        num_steps: int,
        guidance_scale: float,
        guidance_scale_2: Optional[float],
        image: Any,
        generator: Any,
    ) -> dict:
        """Build the kwargs for ``pipe(**call)`` — the frames call. ``image`` is
        non-None only for i2v; the backend preps it (resize/convert) as needed."""

    def teardown(self, pipe: Any) -> None:
        """Release a built pipeline's per-variant RAM before the next builds.
        Default no-op; backends with offload hooks / big modules override."""
