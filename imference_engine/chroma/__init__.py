"""Chroma sub-package — FLUX-derived, de-distilled DiT, self-contained unit.

Parallel to ``imference_engine.flux``. Rides the generic Engine / ModelManager /
RuntimeConfig machinery on the same diffusers 0.39 stack; the split is a
packaging boundary. The 8.9B transformer is heavy — run with
``RuntimeConfig(enable_cpu_offload=True)`` on consumer VRAM.
"""
from imference_engine.chroma.backend import ChromaBackend

__all__ = ["ChromaBackend"]
