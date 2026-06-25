"""Stable Diffusion 1.5 sub-package — a deliberately tiny "for fun" engine.

Self-contained, single-model, no features. It loads ONE SD 1.5 checkpoint once
and renders text-to-image. No multi-model LRU, no offline tree, no CDN, no
img2img, no batching — that all lives in ``imference_engine.engine.Engine``.
This exists only to run an old model for a week or two without dragging the
whole production pipeline along.

    from imference_engine.sd15 import SD15Engine

    engine = SD15Engine().load()
    image = engine.generate(prompt="a girl laying on grass")
    image.save("out.png")
"""
from imference_engine.sd15.engine import DEFAULT_MODEL, SD15Engine

__all__ = ["SD15Engine", "DEFAULT_MODEL"]
