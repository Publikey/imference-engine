"""Tiny env-var parsing helpers for the engine config contract.

Centralized so ``RuntimeConfig.from_env`` (image side) and
``WanRuntimeConfig.from_env`` (Wan) parse the documented env contract
identically — same truthy set, same int/``None`` handling — instead of each
config re-implementing ``os.getenv`` quirks. See the per-engine READMEs
(``pipelines/README.md``, ``wan/README.md``, ``zimage/README.md``) for the full
env↔param↔default tables these helpers back.
"""
from __future__ import annotations

import os
from typing import Optional

# Generous truthy set so a human-set "yes"/"on"/"true" all work; everything
# else (including the empty string) is falsy.
_TRUE = {"1", "true", "yes", "on"}


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Value of env var ``name``, or ``default``. Empty string -> ``default``."""
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var. Unset/empty -> ``default``; unknown -> ``False``."""
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in _TRUE


def env_int_or_none(name: str) -> Optional[int]:
    """Int value of env var, or ``None`` when unset/empty/``auto``/non-integer.

    ``auto`` (and unset) deliberately map to ``None`` so the engine keeps its
    safe default while a worker-side hardware detector (e.g.
    ``config/resource_detection.py``) fills the real number on top. This is the
    decoupling seam: the engine never does hardware detection itself.
    """
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip().lower()
    if val in ("", "auto"):
        return None
    try:
        return int(val)
    except ValueError:
        return None
