"""Shared backend trunk for the image and video engines.

Phase 4 of the unified-engine-core refactor (docs/unified-engine-core.md §8).

Deliberately MINIMAL. Loading a single-file diffusion pipe (image
``PipelineBackend``) and assembling GGUF-MoE experts + LoRA (video
``VideoBackend``) share almost no method signatures, so forcing them into one
rich interface would be fake unification. Only the identity (``engine``) and the
layer-1 defaults hook live here; each modality's ABC extends this with its own
methods.
"""
from __future__ import annotations

from abc import ABC
from typing import ClassVar

from imference_engine.catalog.defaults import GenerationDefaults


class Backend(ABC):
    """Common trunk for every backend (image or video)."""

    engine: ClassVar[str]
    """Identifier matching a catalog row's ``engine`` ("sdxl" | "wan" | ...)."""

    def engine_defaults(self) -> GenerationDefaults:
        """Layer 1 of the defaults precedence chain: family-wide generation
        defaults for this engine, overridable by the model catalog and per
        request (see ``catalog/defaults.py``).

        Return only what is OPINIONATED for the family — e.g. SDXL's default
        scheduler, Z-Image's low guidance. Anything left ``None`` falls through
        to ``GLOBAL_DEFAULTS``. Engine INVARIANTS (dtype, attention) do NOT
        belong here — they stay hard-coded in the backend. Default: no opinion.
        """
        return GenerationDefaults()
