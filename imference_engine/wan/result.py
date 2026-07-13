"""Result types — moved to ``imference_engine.core.result`` (shared image+video).

Kept as a re-export shim. ``WanVideoResult`` / ``WanGenerationError`` are gone —
``WanEngine.generate_video`` now returns the shared ``MediaResult``
(``kind="video"``, ``.frames`` alias, ``.fps`` / ``.num_frames`` metadata,
``.ok`` / ``.error`` helpers).
"""
from imference_engine.core.result import GenerationError, MediaResult

__all__ = ["GenerationError", "MediaResult"]
