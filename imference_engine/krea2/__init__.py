"""Krea 2 sub-package — Krea AI's Krea 2 (Turbo), split out as a self-contained unit.

Parallel to ``imference_engine.flux`` / ``.qwenimage``: Krea 2 rides the generic
``Engine`` / ``ModelManager`` / ``RuntimeConfig`` machinery and the same
diffusers 0.40 stack — no separate engine, just the backend, its in-memory
ComfyUI/native single-file normalization (``convert.py``: key remap + exact
scaled-fp8 dequant), and offline base-component loading. Built for the
civitai/ComfyUI **Krea 2 Turbo** finetune ecosystem (8 steps, guidance 0.0);
fp8 checkpoints stay ~13 GB resident via layerwise fp8 storage. The 12.9B
transformer in bf16 wants ``RuntimeConfig(enable_offload=True)`` on consumer
VRAM.
"""
from imference_engine.krea2.backend import Krea2Backend

__all__ = ["Krea2Backend"]
