"""Lumina-Image 2.0 sub-package — Alpha-VLLM's 2.6B model and its anime finetunes.

Parallel to ``imference_engine.ernieimage`` / ``.flux2klein``: rides the generic
``Engine`` / ``ModelManager`` / ``RuntimeConfig`` machinery on the pinned
diffusers 0.40 stack — the backend, its single-file loading (transformer-only,
ComfyUI all-in-one, GGUF) and the per-machine build table (``builds.py``).
Undistilled (true CFG, negative prompt honored); text-to-image only.
"""
from imference_engine.lumina2.backend import Lumina2Backend
from imference_engine.lumina2.builds import (
    AUTO,
    BUILDS,
    build_named,
    recommend_build,
)

__all__ = ["AUTO", "BUILDS", "Lumina2Backend", "build_named", "recommend_build"]
