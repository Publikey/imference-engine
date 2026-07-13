"""Back-compat shim — the residency manager moved to ``video/residency.py``.

Phase 4c of the unified-engine-core refactor generified the Wan residency into an
arch-agnostic ``ResidencyManager`` driven by a ``VideoBackend`` (Wan's builder
logic now lives in ``video/backends/wan.py`` / ``wan/loader.py``). This re-export
keeps ``from imference_engine.wan.manager import ResidencyManager`` working.
"""
from imference_engine.video.residency import ResidencyManager

__all__ = ["ResidencyManager"]
