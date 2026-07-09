"""Shared engine core — machinery common to the image and video engines.

Phase 1 of the unified-engine-core refactor (see docs/unified-engine-core.md):
the leaf result types live here so both ``Engine`` (image) and ``WanEngine``
(video) return one ``MediaResult``. Later phases move the config base, the
``BaseEngine``, the ``Backend`` protocol, and the residency interface here too.
"""
from imference_engine.core.result import GenerationError, MediaResult

__all__ = ["GenerationError", "MediaResult"]
