"""Result types — moved to ``imference_engine.core.result`` (shared image+video).

Kept as a re-export shim so ``from imference_engine.types import ...`` keeps
resolving. ``GenerationResult`` is gone — use ``MediaResult`` (``.images`` still
works as an alias for the image batch).
"""
from imference_engine.core.result import GenerationError, MediaResult

__all__ = ["GenerationError", "MediaResult"]
