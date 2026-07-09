"""Video sub-package — the video-architecture abstraction (parallel to image/).

Houses the ``VideoBackend`` ABC and its implementations (Wan today; LTX-Video
next). The residency manager + video engine drive backends through this ABC, so a
new architecture is a new ``VideoBackend`` subclass. The low-level Wan builder
(GGUF/LoRA loading, presets, config) still lives under ``imference_engine.wan``;
``WanBackend`` wraps it.
"""
from imference_engine.video.backend import VideoBackend, VideoBuildContext
from imference_engine.video.backends.wan import WanBackend

__all__ = ["VideoBackend", "VideoBuildContext", "WanBackend"]
