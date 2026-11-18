"""FLUX.2 [klein] sub-package — Black Forest Labs' 4B model (Apache-2.0).

Parallel to ``imference_engine.flux`` / ``.krea2``: klein rides the generic
``Engine`` / ``ModelManager`` / ``RuntimeConfig`` machinery on the pinned
diffusers 0.40 stack — the backend, its single-file loading (bf16, BFL scaled
fp8, GGUF), and the per-machine build table (``builds.py``: which transformer
quantization, encoder storage and offload mode for a given VRAM).
Text-to-image and reference editing through one pipeline.
"""
from imference_engine.flux2klein.backend import Flux2KleinBackend
from imference_engine.flux2klein.builds import (
    AUTO,
    BUILDS,
    KleinBuild,
    build_named,
    recommend_build,
)

__all__ = ["AUTO", "BUILDS", "Flux2KleinBackend", "KleinBuild", "build_named", "recommend_build"]
