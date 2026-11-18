"""Which Lumina-Image 2.0 build to run on a given machine.

Pure Python (see ``managers/builds.py``). Measured on an RTX 3090 at 1024x1024,
30 steps with CFG (60 transformer passes), diffusers 0.40.0 (2026-11-12):

==========  ===========  =======  =======  =========  ========
build       transformer  encoder  offload  peak VRAM  s/image
==========  ===========  =======  =======  =========  ========
full        bf16 5.2 GB  bf16     none     13.2 GB    28
balanced    bf16         fp8      none     11.2 GB    28
compact     GGUF Q8_0    fp8      none      8.8 GB    31
offload     GGUF Q8_0    fp8      model     3.4 GB    33
minimal     GGUF Q4_K_M  fp8      group     2.7 GB    58
==========  ===========  =======  =======  =========  ========

Why these:

- The model is small and compute-bound (true CFG doubles every step), so the
  quantized builds are not faster — GGUF is even slightly slower (on-the-fly
  dequantization over 60 passes). bf16 is used wherever it fits.
- The encoder runs once per request, so model offload costs ~2 s and cuts the
  peak to the transformer alone: it is the build for 6-10 GB cards, ahead of
  any smaller resident build.
- GGUF Q8_0 and Q4_K_M are near-identical to the bf16 file they were made
  from; the fp8 encoder changes details of the composition, not the quality.

The "bf16" transformer is the Comfy-Org repackage of the OFFICIAL checkpoint
(``consolidated.pth``). The ``transformer/`` folder of the upstream diffusers
repo holds different weights and is not used by any build.
"""
from __future__ import annotations

from typing import Optional

from imference_engine.managers import builds as _builds
from imference_engine.managers.builds import AUTO, ModelBuild

__all__ = ["AUTO", "BUILDS", "FALLBACK", "build_named", "recommend_build"]

BF16_REPO = "Comfy-Org/Lumina_Image_2.0_Repackaged"
BF16_FILE = "split_files/diffusion_models/lumina_2_model_bf16.safetensors"
GGUF_REPO = "calcuis/lumina-gguf"

# Ordered best-first; the first build whose min_vram_gb fits wins. Floors sit
# above the measured RESERVED memory, plus what a desktop session holds.
BUILDS: tuple = (
    ModelBuild("full", BF16_REPO, BF16_FILE, "bf16", None, 18.0, 13.2,
               "bf16 transformer + bf16 encoder, all resident"),
    ModelBuild("balanced", BF16_REPO, BF16_FILE, "fp8", None, 15.0, 11.2,
               "bf16 transformer + fp8 encoder, all resident"),
    ModelBuild("compact", GGUF_REPO, "lumina2-q8_0.gguf", "fp8", None, 11.25, 8.8,
               "GGUF Q8_0 transformer + fp8 encoder, all resident"),
    ModelBuild("offload", GGUF_REPO, "lumina2-q8_0.gguf", "fp8", "model", 5.5, 3.4,
               "GGUF Q8_0 transformer + fp8 encoder, model offload"),
    ModelBuild("minimal", GGUF_REPO, "lumina2-q4_k_m.gguf", "fp8", "group", 0.0, 2.7,
               "GGUF Q4_K_M transformer + fp8 encoder, group offload"),
)

# No CUDA (CPU / Apple MPS) or unknown VRAM: float8 storage needs CUDA kernels
# and GGUF is unvalidated there, so stay on plain bf16 weights.
FALLBACK = ModelBuild("full", BF16_REPO, BF16_FILE, "bf16", None, 0.0, 13.2,
                      "no CUDA VRAM reading: plain bf16 weights")


def build_named(name: str) -> ModelBuild:
    return _builds.build_named(BUILDS, name)


def recommend_build(vram_gb: Optional[float]) -> ModelBuild:
    """The best build for a card with ``vram_gb`` GiB of TOTAL VRAM (``None`` =
    no CUDA reading -> plain bf16). See ``managers.builds.pick_build``."""
    return _builds.pick_build(BUILDS, vram_gb, FALLBACK)
