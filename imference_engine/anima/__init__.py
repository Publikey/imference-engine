"""Anima sub-package — CircleStone Labs / Comfy Org Anima (Modular Diffusers).

Unlike the other image backends, Anima ships only as a Modular Diffusers pipeline
(no standard AnimaPipeline, no from_single_file, no img2img). This backend adapts
that modular API onto the generic Engine machinery. Validated e2e (text-to-image)
on diffusers 0.39; see ``backend.py`` for details.
"""
from imference_engine.anima.backend import AnimaBackend

__all__ = ["AnimaBackend"]
