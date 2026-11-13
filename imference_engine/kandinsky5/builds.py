"""How to hold Kandinsky 5.0 Image Lite in memory on a given machine.

Pure Python (see ``managers/builds.py``). Like LongCat, Kandinsky 5 has one
weights form usable with diffusers 0.40 (the bf16 repo), so a build is a
storage choice — transformer and encoder in bf16 or fp8 — plus an offload mode.

**NOT MEASURED.** The GPU pod was lost before this backend's runs completed, so
this table is deliberately conservative rather than tuned: every CUDA card gets
fp8 storage with model offload. That combination peaked at 10.3 GB on
LongCat-Image, which has the same transformer size (6B) and the very same
Qwen2.5-VL encoder — an expectation, not a measurement. ``peak_vram_gb`` is 0.0
(unknown) throughout. Re-run ``validation/e2e_backend.py`` on a GPU and replace
this table before relying on ``weights_path="auto"`` (see the package README).
"""
from __future__ import annotations

from typing import Optional

from imference_engine.managers import builds as _builds
from imference_engine.managers.builds import AUTO, ModelBuild

__all__ = ["AUTO", "BUILDS", "FALLBACK", "build_named", "recommend_build"]

REPO = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"

# Ordered best-first; the first build whose min_vram_gb fits wins.
BUILDS: tuple = (
    ModelBuild("offload", REPO, "", "fp8", "model", 12.0, 0.0,
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
