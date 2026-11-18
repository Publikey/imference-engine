"""ERNIE-Image sub-package — Baidu's 8B text-to-image model (Apache-2.0).

Parallel to ``imference_engine.flux2klein`` / ``.krea2``: rides the generic
``Engine`` / ``ModelManager`` / ``RuntimeConfig`` machinery on the pinned
diffusers 0.40 stack — the backend, its GGUF / single-file loading, and the
per-machine build table (``builds.py``). Built for ERNIE-Image-Turbo (8 steps,
guidance 1.0); text-to-image only.
"""
from imference_engine.ernieimage.backend import ErnieImageBackend
from imference_engine.ernieimage.builds import (
    AUTO,
    BUILDS,
    build_named,
    recommend_build,
)

__all__ = ["AUTO", "BUILDS", "ErnieImageBackend", "build_named", "recommend_build"]
