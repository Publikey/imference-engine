"""Declarative model catalog.

Implemented:
  - defaults.py  — GenerationDefaults + the engine<model<request precedence chain
  - loader.py    — parse models.yml -> list[ModelConfig] (strict validation)

Planned (still stubbed):
  - disk_cache.py  — on-disk LRU for downloaded weights (lifted verbatim)
  - remote_sync.py — hot-reload models.yml from HTTP (URL as constructor arg)
"""
from imference_engine.catalog.defaults import GLOBAL_DEFAULTS, GenerationDefaults
from imference_engine.catalog.loader import CatalogError, ModelConfig, load, loads

__all__ = [
    "GenerationDefaults",
    "GLOBAL_DEFAULTS",
    "ModelConfig",
    "CatalogError",
    "load",
    "loads",
]
