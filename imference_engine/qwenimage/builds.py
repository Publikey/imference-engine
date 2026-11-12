"""Which Qwen-Image build to run on a given machine (lite builds only).

Pure Python (see ``managers/builds.py``). The bf16 transformer is 41 GB, so
every ``auto`` build is the lite recipe of ``lite.py`` — Qwen-Image-2512 GGUF
Q4_K_M (13.3 GB) + the Lightning 4-step LoRA + the NF4 encoder — and only the
offload mode changes with the card. Measured on an RTX 3090 at 1024x1024,
4 steps, diffusers 0.40.0 (2026-11-13):

==========  =======  =========  ========
build       offload  peak VRAM  s/image
==========  =======  =========  ========
resident    none     21.0 GB     9.2
offload     model    14.9 GB    17.2
minimal     group    (below)    host-bound
==========  =======  =========  ========

- ``resident``: transformer and VAE stay on the card; the NF4 encoder is kept
  between requests when the free VRAM allows it (it does on 24 GB at 1 MP).
- ``offload``: the 13.3 GB transformer is the peak, so it fits 16 GB cards;
  twice the time per image (the transformer shuttles every request).
- ``minimal``: block-streamed transformer for 8-12 GB cards — the recipe
  validated for Qwen-Image-Edit on an 8 GB laptop (6.5 GB peak). Its speed is
  set by the host (CPU and RAM-to-GPU copies), not by the card.
"""
from __future__ import annotations

from typing import Optional

from imference_engine.managers import builds as _builds
from imference_engine.managers.builds import AUTO, ModelBuild

__all__ = ["AUTO", "BUILDS", "FALLBACK", "build_named", "recommend_build"]

GGUF_REPO = "unsloth/Qwen-Image-2512-GGUF"
GGUF_FILE = "qwen-image-2512-Q4_K_M.gguf"

# Ordered best-first; the first build whose min_vram_gb fits wins.
BUILDS: tuple = (
    ModelBuild("resident", GGUF_REPO, GGUF_FILE, "nf4", None, 22.0, 21.0,
               "GGUF Q4_K_M + Lightning + NF4 encoder, transformer resident"),
    ModelBuild("offload", GGUF_REPO, GGUF_FILE, "nf4", "model", 15.5, 14.9,
               "GGUF Q4_K_M + Lightning + NF4 encoder, model offload"),
    ModelBuild("minimal", GGUF_REPO, GGUF_FILE, "nf4", "group", 0.0, 6.5,
               "GGUF Q4_K_M + Lightning + NF4 encoder, group offload"),
)

# No CUDA reading: group offload degrades to model offload off-CUDA.
FALLBACK = BUILDS[-1]


def build_named(name: str) -> ModelBuild:
    return _builds.build_named(BUILDS, name)


def recommend_build(vram_gb: Optional[float]) -> ModelBuild:
    """The Qwen-Image-2512 lite build for a card with ``vram_gb`` GiB of TOTAL
    VRAM (``None`` = no CUDA reading -> the minimal build)."""
    return _builds.pick_build(BUILDS, vram_gb, FALLBACK)
