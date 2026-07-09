"""Shared runtime-config base for the image and video engines.

Phase 2 of the unified-engine-core refactor (docs/unified-engine-core.md). Holds
the fields both modalities share; ``RuntimeConfig`` (image) and
``WanRuntimeConfig`` (video) subclass this and add their modality-specific knobs.
``from_env()`` stays per-modality (different env prefixes: ``IMAGE_*`` / ``WAN_*``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class BaseRuntimeConfig:
    """Runtime knobs common to every engine. All optional — defaults adapt to the
    host. Subclasses extend with modality-specific fields."""

    device: str = "auto"
    """auto | cuda | cuda:N | mps | cpu"""

    model_cache_dir: Optional[Union[str, Path]] = None
    """Root of the flat, symlink-free offline model tree. Backends resolve shared
    base-components into it, so a cold load is fully offline under
    HF_HUB_OFFLINE=1 once populated (base tarball / prefetch / CDN)."""

    model_cdn: Optional[str] = None
    """Base URL of a CDN (R2) mirroring the same <repo>/<file> layout. When set,
    on-demand base-components download from the CDN instead of HuggingFace."""

    enable_offload: bool = False
    """Unified offload knob (image ``enable_cpu_offload`` + video ``enable_offload``
    merged here). When True the engine uses accelerate's
    ``enable_model_cpu_offload`` — individual submodels shuttle CPU<->GPU so peak
    VRAM drops to the largest single submodel, at ~10-30% throughput cost.
    Defaults False on the image side; the video subclass overrides the default to
    True (offload is the norm for the heavy A14B experts)."""
