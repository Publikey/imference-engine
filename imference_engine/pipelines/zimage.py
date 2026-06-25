"""Back-compat shim — ZImageBackend now lives in ``imference_engine.zimage``.

Z-Image was split out of ``pipelines/`` into its own sub-package (parallel to
``wan``). This re-export keeps ``from imference_engine.pipelines.zimage import
ZImageBackend`` working for existing importers (e.g. ``engine.py``).
"""
from imference_engine.zimage.backend import ZImageBackend  # noqa: F401

__all__ = ["ZImageBackend"]
