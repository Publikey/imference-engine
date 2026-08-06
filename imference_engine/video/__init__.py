"""Video sub-package — the video-architecture abstraction (parallel to image/).

Houses the ``VideoBackend`` ABC and its implementations (Wan and MiniMax-H3
today). The residency manager + a video engine drive backends through this ABC,
so a new architecture is a new ``VideoBackend`` subclass. The low-level builders
(GGUF/LoRA loading, presets, config for Wan; modular load + int8 quant for
MiniMax-H3) live under ``imference_engine.wan`` / ``imference_engine.minimax_h3``;
the backends here wrap them.
"""
from imference_engine.video.backend import VideoBackend, VideoBuildContext
from imference_engine.video.backends.minimax_h3 import MiniMaxH3Backend
from imference_engine.video.backends.wan import WanBackend

# Every video arch any engine knows. Engines validate catalog rows against THIS
# set (so a typo'd arch still fails loudly) and then keep only the rows for the
# backends they actually run — which lets one shared models.yml carry wan AND
# minimax_h3 rows without either engine rejecting the other's.
KNOWN_VIDEO_ARCHS = {WanBackend.engine, MiniMaxH3Backend.engine}

__all__ = ["VideoBackend", "VideoBuildContext", "WanBackend", "MiniMaxH3Backend",
           "KNOWN_VIDEO_ARCHS"]
