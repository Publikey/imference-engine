"""Z-Image sub-package — Tongyi Z-Image, split out as a self-contained unit.

Parallel to ``imference_engine.wan``, but lighter: Z-Image shares the generic
``Engine`` / ``ModelManager`` / ``RuntimeConfig`` machinery and the same diffusers
0.39 stack as SDXL, so there is no separate ``ZImageEngine`` — just the backend,
its offline base-component loading, and (via the worker) a dedicated catalog.
"""
from imference_engine.zimage.backend import ZImageBackend

__all__ = ["ZImageBackend"]
