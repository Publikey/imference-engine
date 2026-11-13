"""Which HiDream-I1 build to run on a given machine.

Pure Python (see ``managers/builds.py``). The bf16 transformer is 34 GB and
the four text encoders add 29 GB, so every build is a GGUF transformer run with
offload (the encoders are used once per request) and Llama + T5 stored in fp8.

**NOT MEASURED.** The GPU pod was lost before HiDream could be run. The floors
below are derived from file sizes only — a GGUF Q8_0 transformer is 18.7 GB and
a Q4_K_M 11.5 GB, and under model offload the peak is the transformer plus its
activations — and ``peak_vram_gb`` is 0.0 (unknown). Re-run
``validation/e2e_backend.py`` on a GPU and replace this table before relying on
``weights_path="auto"`` (see the package README).
"""
from __future__ import annotations

from typing import Optional

from imference_engine.managers import builds as _builds
from imference_engine.managers.builds import AUTO, ModelBuild

__all__ = ["AUTO", "BUILDS", "FALLBACK", "build_named", "recommend_build"]

GGUF_REPO = "city96/HiDream-I1-Fast-gguf"

# Ordered best-first; the first build whose min_vram_gb fits wins.
BUILDS: tuple = (
    ModelBuild("quality", GGUF_REPO, "hidream-i1-fast-Q8_0.gguf", "fp8", "model", 23.0, 0.0,
               "GGUF Q8_0 transformer + fp8 Llama/T5, model offload (not measured)"),
    ModelBuild("balanced", GGUF_REPO, "hidream-i1-fast-Q4_K_M.gguf", "fp8", "model", 15.0, 0.0,
               "GGUF Q4_K_M transformer + fp8 Llama/T5, model offload (not measured)"),
    ModelBuild("minimal", GGUF_REPO, "hidream-i1-fast-Q4_K_M.gguf", "fp8", "group", 0.0, 0.0,
               "GGUF Q4_K_M transformer + fp8 Llama/T5, group offload (not measured)"),
)

# No CUDA (float8 kernels need it): the small GGUF, bf16 encoders, model offload.
FALLBACK = ModelBuild("bf16-encoders", GGUF_REPO, "hidream-i1-fast-Q4_K_M.gguf", "bf16", "model",
                      0.0, 0.0, "no CUDA VRAM reading: GGUF Q4_K_M, bf16 encoders, model offload")


def build_named(name: str) -> ModelBuild:
    return _builds.build_named(BUILDS, name)


def recommend_build(vram_gb: Optional[float]) -> ModelBuild:
    """The build for a card with ``vram_gb`` GiB of TOTAL VRAM."""
    return _builds.pick_build(BUILDS, vram_gb, FALLBACK)
