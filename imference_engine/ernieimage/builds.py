"""Which ERNIE-Image-Turbo build to run on a given machine.

Pure Python (see ``managers/builds.py``). Measured on an RTX 3090 at 1024x1024,
8 steps, diffusers 0.40.0 (2026-11-12):

==========  =============  =======  =======  =========  ========
build       transformer    encoder  offload  peak VRAM  s/image
==========  =============  =======  =======  =========  ========
quality     GGUF Q8_0      bf16     none     19.4 GB    12.5
balanced    GGUF Q5_K_M    fp8      none     12.9 GB    14.5
offload     GGUF Q8_0      fp8      model     9.8 GB    19.7
compact     GGUF Q4_K_M    fp8      model     6.1 GB    16.6
minimal     GGUF Q4_K_M    fp8      group     2.7 GB    17.7
==========  =============  =======  =======  =========  ========

Why these:

- The bf16 transformer is 16 GB and the encoder 7.7 GB: bf16 does not fit a
  24 GB card resident, so even the top tier is a GGUF. Q8_0 is near-lossless
  against bf16 (LPIPS 0.05 on fixed seeds) and runs at the same speed as
  fp8 storage (the model is compute-bound: ~1.5 s/step).
- The fp8 encoder (float8 storage, bf16 compute) changes the composition more
  than on FLUX.2 klein (LPIPS ~0.24) but not the quality: text stays correct.
  It is what brings the resident footprint under 16 GB.
- Q8_0 under group offload is too slow (44 s — every streamed block is twice
  the bytes), so the lowest tier streams Q4_K_M instead.

There is no resident bf16 tier: it needs a 32 GB card, which was not available
to measure. The bf16 single file still loads when registered explicitly (model
offload: 16.9 GB peak).
"""
from __future__ import annotations

from typing import Optional

from imference_engine.managers import builds as _builds
from imference_engine.managers.builds import AUTO, ModelBuild

__all__ = ["AUTO", "BUILDS", "FALLBACK", "build_named", "recommend_build"]

GGUF_REPO = "unsloth/ERNIE-Image-Turbo-GGUF"
BF16_REPO = "Comfy-Org/ERNIE-Image"
BF16_FILE = "diffusion_models/ernie-image-turbo.safetensors"

# Ordered best-first; the first build whose min_vram_gb fits wins. Floors sit
# above the measured RESERVED memory (not just the peak allocation), plus what
# a desktop session holds on the card.
BUILDS: tuple = (
    ModelBuild("quality", GGUF_REPO, "ernie-image-turbo-Q8_0.gguf", "bf16", None, 22.0, 19.4,
               "GGUF Q8_0 transformer + bf16 encoder, all resident"),
    ModelBuild("balanced", GGUF_REPO, "ernie-image-turbo-Q5_K_M.gguf", "fp8", None, 15.0, 12.9,
               "GGUF Q5_K_M transformer + fp8 encoder, all resident"),
    ModelBuild("offload", GGUF_REPO, "ernie-image-turbo-Q8_0.gguf", "fp8", "model", 11.0, 9.8,
               "GGUF Q8_0 transformer + fp8 encoder, model offload"),
    ModelBuild("compact", GGUF_REPO, "ernie-image-turbo-Q4_K_M.gguf", "fp8", "model", 7.25, 6.1,
               "GGUF Q4_K_M transformer + fp8 encoder, model offload"),
    ModelBuild("minimal", GGUF_REPO, "ernie-image-turbo-Q4_K_M.gguf", "fp8", "group", 0.0, 2.7,
               "GGUF Q4_K_M transformer + fp8 encoder, group offload"),
)

# No CUDA (CPU / Apple MPS) or unknown VRAM: float8 storage needs CUDA kernels
# and GGUF is unvalidated there, so stay on plain bf16 weights.
FALLBACK = ModelBuild("bf16", BF16_REPO, BF16_FILE, "bf16", None, 0.0, 16.9,
                      "no CUDA VRAM reading: plain bf16 weights")


def build_named(name: str) -> ModelBuild:
    return _builds.build_named(BUILDS, name)


def recommend_build(vram_gb: Optional[float]) -> ModelBuild:
    """The best build for a card with ``vram_gb`` GiB of TOTAL VRAM (``None`` =
    no CUDA reading -> plain bf16). See ``managers.builds.pick_build``."""
    return _builds.pick_build(BUILDS, vram_gb, FALLBACK)
