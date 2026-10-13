"""Named progress stages (never an opaque "generating")."""
from __future__ import annotations

from typing import Callable, Optional

DOWNLOADING_WEIGHTS = "downloading_weights"
LOADING = "loading"
RUNNING = "running"
ENCODING = "encoding"
STAGES = (DOWNLOADING_WEIGHTS, LOADING, RUNNING, ENCODING)

ProgressFn = Callable[[str, float], None]


class Progress:
    """Wraps a user callback: clamps fractions, never lets a broken callback
    kill the tool, and remembers the stage timeline for logs."""

    def __init__(self, fn: Optional[ProgressFn] = None) -> None:
        self._fn = fn
        self.timeline: list = []

    def __call__(self, stage: str, fraction: float = 0.0) -> None:
        fraction = 0.0 if fraction != fraction else max(0.0, min(1.0, float(fraction)))
        if not self.timeline or self.timeline[-1] != stage:
            self.timeline.append(stage)
        if self._fn is not None:
            try:
                self._fn(stage, fraction)
            except Exception:  # pragma: no cover - a UI callback must never fail a tool
                pass
