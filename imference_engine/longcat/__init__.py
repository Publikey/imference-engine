"""LongCat-Image sub-package — Meituan's 6B model (Apache-2.0), text-to-image and editing.

Parallel to ``imference_engine.lumina2`` / ``.ernieimage``: two backends on the
generic ``Engine`` machinery and the pinned diffusers 0.40 stack — ``longcat``
(text-to-image) and ``longcat-edit`` (instruction editing, Edit-Turbo by
default) — sharing one mirrored copy of the Qwen2.5-VL encoder, with fp8
storage builds chosen per VRAM (``builds.py``).
"""
from imference_engine.longcat.backend import LongCatBackend, LongCatEditBackend
from imference_engine.longcat.builds import (
    AUTO,
    BUILDS,
    build_named,
    recommend_build,
)

__all__ = ["AUTO", "BUILDS", "LongCatBackend", "LongCatEditBackend", "build_named",
           "recommend_build"]
