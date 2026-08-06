"""MiniMax-H3 video sub-package — joint video + audio generation.

    from imference_engine.minimax_h3 import MiniMaxH3Engine, MiniMaxH3RuntimeConfig

    engine = MiniMaxH3Engine(runtime=MiniMaxH3RuntimeConfig(device="auto")).load()
    res = engine.generate_video(prompt="a red fox trotting through snow")
    # res.frames (PIL, 24 fps) + res.audio ((2, n) numpy) + res.sample_rate

UPSTREAM: needs a diffusers build with PR #14355 (unreleased) — see the
package README. The imports here are torch/diffusers-free so listing variants
and parsing configs works anywhere.
"""
from imference_engine.minimax_h3.config import (H3MemoryProfile,
                                                MiniMaxH3RuntimeConfig)
from imference_engine.minimax_h3.engine import MiniMaxH3Engine
from imference_engine.minimax_h3.presets import (BUILTIN_VARIANTS, H3Variant,
                                                 OFFICIAL_REPO)

__all__ = [
    "MiniMaxH3Engine",
    "MiniMaxH3RuntimeConfig",
    "H3MemoryProfile",
    "H3Variant",
    "BUILTIN_VARIANTS",
    "OFFICIAL_REPO",
]
