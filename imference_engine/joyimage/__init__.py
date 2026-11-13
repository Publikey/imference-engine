"""JoyAI-Image-Edit-Plus sub-package — JD's 16B multi-image editing model (Apache-2.0).

Parallel to ``imference_engine.longcat``: the backend on the generic ``Engine``
machinery and the pinned diffusers 0.40 stack, with fp8 storage builds chosen
per VRAM (``builds.py``). Instruction editing from one to five reference images.
"""
from imference_engine.joyimage.backend import JoyImageEditBackend
from imference_engine.joyimage.builds import (
    AUTO,
    BUILDS,
    build_named,
    recommend_build,
)

__all__ = ["AUTO", "BUILDS", "JoyImageEditBackend", "build_named", "recommend_build"]
