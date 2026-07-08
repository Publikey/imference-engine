"""Qwen-Image sub-package — Alibaba Qwen-Image, self-contained unit.

Parallel to ``imference_engine.flux`` / ``.chroma``. Rides the generic Engine /
ModelManager / RuntimeConfig machinery on the same diffusers 0.39 stack; the
split is a packaging boundary. The 20B transformer is very heavy — run with
``RuntimeConfig(enable_cpu_offload=True)`` and prefer a quantized build on
consumer VRAM.
"""
from imference_engine.qwenimage.backend import QwenImageBackend

__all__ = ["QwenImageBackend"]
