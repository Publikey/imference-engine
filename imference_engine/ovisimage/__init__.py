"""Ovis-Image sub-package — Alibaba AIDC's 7B text-to-image model (Apache-2.0).

Parallel to ``imference_engine.longcat`` / ``.ernieimage``: the backend on the
generic ``Engine`` machinery and the pinned diffusers 0.40 stack, with fp8
storage builds chosen per VRAM (``builds.py``). Text-to-image only; its
strength is legible text.
"""
from imference_engine.ovisimage.backend import OvisImageBackend
from imference_engine.ovisimage.builds import (
    AUTO,
    BUILDS,
    build_named,
    recommend_build,
)

__all__ = ["AUTO", "BUILDS", "OvisImageBackend", "build_named", "recommend_build"]
