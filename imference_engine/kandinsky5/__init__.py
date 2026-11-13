"""Kandinsky 5.0 Image Lite sub-package — a 6B model (MIT), text-to-image and editing.

Parallel to ``imference_engine.longcat``: two backends on the generic ``Engine``
machinery and the pinned diffusers 0.40 stack — ``kandinsky5`` (text-to-image)
and ``kandinsky5-edit`` (instruction editing) — sharing one mirrored copy of the
encoders, with fp8 storage builds chosen per VRAM (``builds.py``).
"""
from imference_engine.kandinsky5.backend import Kandinsky5Backend, Kandinsky5EditBackend
from imference_engine.kandinsky5.builds import (
    AUTO,
    BUILDS,
    build_named,
    recommend_build,
)

__all__ = ["AUTO", "BUILDS", "Kandinsky5Backend", "Kandinsky5EditBackend", "build_named",
           "recommend_build"]
