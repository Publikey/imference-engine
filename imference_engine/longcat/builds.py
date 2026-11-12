"""How to hold LongCat-Image in memory on a given machine.

Pure Python (see ``managers/builds.py``). LongCat has ONE weights form usable
with diffusers 0.40 (the bf16 repo: no single-file / GGUF loader exists for its
transformer), so a build here is not a file but a storage choice — transformer
and encoder in bf16 or fp8 (float8 storage, bf16 compute) — plus an offload
mode. Measured on an RTX 3090 at 1024x1024, diffusers 0.40.0 (2026-11-13):

==========  ===========  =======  =======  =========  ===========  ==========
build       transformer  encoder  offload  peak VRAM  t2i 50 step  edit Turbo
==========  ===========  =======  =======  =========  ===========  ==========
resident    fp8          fp8      none     19.6 GB    77 s         14.8 s
quality     bf16         fp8      model    13.1 GB    93 s         —
offload     fp8          fp8      model    10.3 GB    92 s         23.6 s
minimal     fp8          fp8      group    not timed (host-bound)
==========  ===========  =======  =======  =========  ===========  ==========

(text-to-image times are 30-step measurements scaled to the 50-step recipe.)

- fp8 storage is visually identical to bf16 here, text included, and model
  offload costs ~10 s per image: the encoder runs once per request.
- Plain bf16 with model offload (the upstream recipe) peaks at 17.0 GB
  (18.4 GB for editing) — it only fits 24 GB cards, where fp8 resident is
  faster. So bf16 for the transformer is offered on the 16 GB tier, where it
  fits next to an fp8 encoder.
- ``minimal`` (block-streamed transformer) is wired but was not timed: group
  offload speed depends on the host, and the test host was not representative.
"""
from __future__ import annotations

from typing import Optional

from imference_engine.managers import builds as _builds
from imference_engine.managers.builds import AUTO, ModelBuild

__all__ = ["AUTO", "BUILDS", "FALLBACK", "build_named", "recommend_build"]

REPO = "meituan-longcat/LongCat-Image"

# Ordered best-first; the first build whose min_vram_gb fits wins. Floors sit
# above the measured RESERVED memory plus what a desktop session holds.
BUILDS: tuple = (
    ModelBuild("resident", REPO, "", "fp8", None, 22.0, 19.6,
               "fp8 transformer + fp8 encoder, all resident", transformer="fp8"),
    ModelBuild("quality", REPO, "", "fp8", "model", 15.0, 13.1,
               "bf16 transformer + fp8 encoder, model offload", transformer="bf16"),
    ModelBuild("offload", REPO, "", "fp8", "model", 11.5, 10.3,
               "fp8 transformer + fp8 encoder, model offload", transformer="fp8"),
    ModelBuild("minimal", REPO, "", "fp8", "group", 0.0, 10.3,
               "fp8 transformer + fp8 encoder, group offload", transformer="fp8"),
)

# No CUDA (float8 kernels need it): plain bf16 with model offload.
FALLBACK = ModelBuild("bf16", REPO, "", "bf16", "model", 0.0, 17.0,
                      "no CUDA VRAM reading: bf16 weights, model offload", transformer="bf16")


def build_named(name: str) -> ModelBuild:
    return _builds.build_named(BUILDS, name)


def recommend_build(vram_gb: Optional[float]) -> ModelBuild:
    """The storage / offload combination for a card with ``vram_gb`` GiB of
    TOTAL VRAM (``None`` = no CUDA reading -> bf16 with model offload)."""
    return _builds.pick_build(BUILDS, vram_gb, FALLBACK)
