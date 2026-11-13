"""Which FLUX.2 klein 4B build to run on a given machine.

Pure Python (no torch import, like ``managers/batch.py``): the caller measures
the VRAM and passes it in, so the table is unit-testable without a GPU and
usable from a launcher that has not loaded torch yet (the Desktop sidecar
picking what to download).

A *build* is three choices measured together on an RTX 3090 (1024x1024, 4
steps, diffusers 0.40.0, 2026-11-12):

==========  ============  =======  =======  =========  ========
build       transformer   encoder  offload  peak VRAM  s/image
==========  ============  =======  =======  =========  ========
full        bf16 7.75 GB  bf16     none     18.6 GB    2.7
balanced    GGUF Q8_0     fp8      none     11.1 GB    2.9
compact     GGUF Q4_K_M   fp8      none      9.4 GB    3.2
offload     GGUF Q8_0     fp8      model     5.1 GB    7.7
minimal     GGUF Q4_K_M   fp8      group     2.8 GB    6.7
==========  ============  =======  =======  =========  ========

Why these: the Qwen3-4B text encoder is as big as the transformer (8 GB bf16),
so quantizing the transformer alone never gets under ~13 GB — the fp8 encoder
(float8 storage, bf16 compute) is what opens the 12-16 GB cards. Q8_0 is
visually indistinguishable from bf16 (LPIPS 0.02 on fixed seeds); Q4_K_M and
the fp8 encoder change details of the composition but not the quality — legible
text included, where the 4-step model's own error rate dominates. BFL's
official fp8 transformer is no smaller than Q8_0 and further from bf16, and a
4-bit (NF4) encoder was visibly worse on text: neither is recommended (the fp8
file still loads when registered explicitly).

Under offload the encoder sets the peak, so the 8 GB tier keeps the better Q8_0
transformer; group offload is slower per step but streams the transformer in
blocks, which is what fits 4-6 GB cards.
"""
from __future__ import annotations

from typing import Optional

from imference_engine.managers import builds as _builds
from imference_engine.managers.builds import AUTO, ModelBuild

__all__ = ["AUTO", "BUILDS", "FALLBACK", "KleinBuild", "build_named", "recommend_build"]

BF16_REPO = "black-forest-labs/FLUX.2-klein-4B"
GGUF_REPO = "unsloth/FLUX.2-klein-4B-GGUF"

# The record is shared with the other multi-build backends (managers/builds.py).
KleinBuild = ModelBuild


# Ordered best-first; the first build whose min_vram_gb fits wins. Thresholds
# leave ~1.5-3 GiB over the measured peak for the allocator's reserve and what
# the desktop itself holds on the card.
BUILDS: tuple = (
    KleinBuild("full", BF16_REPO, "flux-2-klein-4b.safetensors", "bf16", None, 22.0, 18.6,
               "bf16 transformer + bf16 encoder, all resident"),
    KleinBuild("balanced", GGUF_REPO, "flux-2-klein-4b-Q8_0.gguf", "fp8", None, 14.0, 11.1,
               "GGUF Q8_0 transformer + fp8 encoder, all resident"),
    KleinBuild("compact", GGUF_REPO, "flux-2-klein-4b-Q4_K_M.gguf", "fp8", None, 11.0, 9.4,
               "GGUF Q4_K_M transformer + fp8 encoder, all resident"),
    KleinBuild("offload", GGUF_REPO, "flux-2-klein-4b-Q8_0.gguf", "fp8", "model", 7.0, 5.1,
               "GGUF Q8_0 transformer + fp8 encoder, model offload"),
    KleinBuild("minimal", GGUF_REPO, "flux-2-klein-4b-Q4_K_M.gguf", "fp8", "group", 0.0, 2.8,
               "GGUF Q4_K_M transformer + fp8 encoder, group offload"),
)

# No CUDA (CPU / Apple MPS) or unknown VRAM: float8 storage needs CUDA kernels
# and GGUF is unvalidated there, so stay on plain bf16 weights.
FALLBACK = KleinBuild("full", BF16_REPO, "flux-2-klein-4b.safetensors", "bf16", None, 0.0, 18.6,
                      "no CUDA VRAM reading: plain bf16 weights")


def build_named(name: str) -> ModelBuild:
    return _builds.build_named(BUILDS, name)


def recommend_build(vram_gb: Optional[float]) -> ModelBuild:
    """The best build for a card with ``vram_gb`` GiB of TOTAL VRAM (``None`` =
    no CUDA reading -> plain bf16). See ``managers.builds.pick_build``."""
    return _builds.pick_build(BUILDS, vram_gb, FALLBACK)
