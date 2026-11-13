"""How to hold JoyAI-Image-Edit-Plus in memory on a given machine.

Pure Python (see ``managers/builds.py``). One weights form usable with
diffusers 0.40 (the bf16 repo: a 32.5 GB transformer and a 17.5 GB Qwen3-VL
encoder), so a build is a storage choice — fp8 or bf16 — plus an offload mode.

**NOT MEASURED.** The GPU pod was lost before this backend could be run. In fp8
the transformer is about 16 GB and the encoder about 9 GB: they cannot both be
resident on 24 GB, so the only build offered is fp8 storage with model offload,
whose peak should be the transformer plus its activations (true CFG over the
target and up to five reference images). Whether that fits a 24 GB card is
exactly what has to be verified — it may turn out to need a 32 GB card or the
group-offload build. ``peak_vram_gb`` is 0.0 (unknown). Re-run
``validation/e2e_backend.py`` on a GPU and replace this table (see the README).
"""
from __future__ import annotations

from typing import Optional

from imference_engine.managers import builds as _builds
from imference_engine.managers.builds import AUTO, ModelBuild

__all__ = ["AUTO", "BUILDS", "FALLBACK", "build_named", "recommend_build"]

REPO = "jdopensource/JoyAI-Image-Edit-Plus-Diffusers"

# Ordered best-first; the first build whose min_vram_gb fits wins.
BUILDS: tuple = (
    ModelBuild("offload", REPO, "", "fp8", "model", 22.0, 0.0,
               "fp8 transformer + fp8 encoder, model offload (not measured)", transformer="fp8"),
    ModelBuild("minimal", REPO, "", "fp8", "group", 0.0, 0.0,
               "fp8 transformer + fp8 encoder, group offload (not measured)", transformer="fp8"),
)

# No CUDA (float8 kernels need it): plain bf16 with model offload.
FALLBACK = ModelBuild("bf16", REPO, "", "bf16", "model", 0.0, 0.0,
                      "no CUDA VRAM reading: bf16 weights, model offload", transformer="bf16")


def build_named(name: str) -> ModelBuild:
    return _builds.build_named(BUILDS, name)


def recommend_build(vram_gb: Optional[float]) -> ModelBuild:
    """The storage / offload combination for a card with ``vram_gb`` GiB of
    TOTAL VRAM (``None`` = no CUDA reading -> bf16 with model offload)."""
    return _builds.pick_build(BUILDS, vram_gb, FALLBACK)
