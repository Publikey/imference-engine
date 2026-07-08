"""FLUX sub-package — Black Forest Labs FLUX.1, split out as a self-contained unit.

Parallel to ``imference_engine.zimage`` / ``.wan``: FLUX rides the generic
``Engine`` / ``ModelManager`` / ``RuntimeConfig`` machinery and the same diffusers
0.38 stack as SDXL / Z-Image, so there is no separate ``FluxEngine`` — just the
backend, its offline base-component loading, and (via a catalog) per-model
defaults. The 12B transformer is heavy; run it with
``RuntimeConfig(enable_cpu_offload=True)`` on consumer VRAM.
"""
from imference_engine.flux.backend import FluxBackend

__all__ = ["FluxBackend"]
