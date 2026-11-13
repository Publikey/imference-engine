"""HiDream-I1 sub-package — a 17B text-to-image model (MIT) with four text encoders.

Parallel to ``imference_engine.ernieimage``: the backend on the generic
``Engine`` machinery and the pinned diffusers 0.40 stack, GGUF transformers and
fp8 encoder storage chosen per VRAM (``builds.py``). The Llama-3.1-8B encoder is
loaded from its own repo (Llama 3.1 Community License).
"""
from imference_engine.hidream.backend import HiDreamBackend
from imference_engine.hidream.builds import (
    AUTO,
    BUILDS,
    build_named,
    recommend_build,
)

__all__ = ["AUTO", "BUILDS", "HiDreamBackend", "build_named", "recommend_build"]
