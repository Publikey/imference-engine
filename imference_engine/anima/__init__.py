"""Anima sub-package — CircleStone Labs / Comfy Org Anima (Modular Diffusers).

Unlike the other image backends, Anima ships only as a Modular Diffusers pipeline
(no standard AnimaPipeline, no from_single_file, no img2img). This backend adapts
that modular API onto the generic Engine machinery. See ``backend.py`` for the
verified-vs-unverified contract.
"""
from imference_engine.anima.backend import AnimaBackend

__all__ = ["AnimaBackend"]
