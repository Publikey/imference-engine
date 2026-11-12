"""How to hold Ovis-Image-7B in memory on a given machine.

Pure Python (see ``managers/builds.py``). Like LongCat, Ovis-Image has one
weights form usable with diffusers 0.40 (the bf16 repo), so a build is a
storage choice — transformer and encoder in bf16 or fp8 (float8 storage, bf16
compute) — plus an offload mode. Measured on an RTX 3090 at 1024x1024,
diffusers 0.40.0 (2026-11-13); times are 30-step measurements scaled to the
50-step recipe:

==========  ===========  =======  =======  =========  ========
build       transformer  encoder  offload  peak VRAM  s/image
==========  ===========  =======  =======  =========  ========
quality     bf16         fp8      none     19.5 GB     88
resident    fp8          fp8      none     13.6 GB     93
offload     fp8          fp8      model     9.6 GB    103
minimal     fp8          fp8      group    not timed (host-bound)
==========  ===========  =======  =======  =========  ========

- fp8 storage is visually near-identical to bf16, legible text included.
- Model offload costs ~10 s per image (the encoder runs once per request).
- Plain bf16 with model offload peaks at 15.4 GB (16.1 GB reserved): too close
  to a 16 GB card's limit to recommend, hence fp8 below 22 GiB.
"""
from __future__ import annotations

from typing import Optional

from imference_engine.managers import builds as _builds
from imference_engine.managers.builds import AUTO, ModelBuild

__all__ = ["AUTO", "BUILDS", "FALLBACK", "build_named", "recommend_build"]

REPO = "AIDC-AI/Ovis-Image-7B"

# Ordered best-first; the first build whose min_vram_gb fits wins. Floors sit
# above the measured RESERVED memory plus what a desktop session holds.
BUILDS: tuple = (
    ModelBuild("quality", REPO, "", "fp8", None, 22.0, 19.5,
               "bf16 transformer + fp8 encoder, all resident", transformer="bf16"),
    ModelBuild("resident", REPO, "", "fp8", None, 18.0, 13.6,
               "fp8 transformer + fp8 encoder, all resident", transformer="fp8"),
    ModelBuild("offload", REPO, "", "fp8", "model", 11.0, 9.6,
               "fp8 transformer + fp8 encoder, model offload", transformer="fp8"),
    ModelBuild("minimal", REPO, "", "fp8", "group", 0.0, 9.6,
               "fp8 transformer + fp8 encoder, group offload", transformer="fp8"),
)

# No CUDA (float8 kernels need it): plain bf16 with model offload.
FALLBACK = ModelBuild("bf16", REPO, "", "bf16", "model", 0.0, 15.4,
                      "no CUDA VRAM reading: bf16 weights, model offload", transformer="bf16")


def build_named(name: str) -> ModelBuild:
    return _builds.build_named(BUILDS, name)


def recommend_build(vram_gb: Optional[float]) -> ModelBuild:
    """The storage / offload combination for a card with ``vram_gb`` GiB of
    TOTAL VRAM (``None`` = no CUDA reading -> bf16 with model offload)."""
    return _builds.pick_build(BUILDS, vram_gb, FALLBACK)
