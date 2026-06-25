"""Result types for Wan video generation.

Mirrors the engine/transport split of the image side: ``WanEngine`` returns
frames + metadata; the caller (worker / sidecar) does ``export_to_video`` and
upload. No file I/O, no encoding here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class WanGenerationError:
    error: str
    seed: Optional[int] = None


@dataclass
class WanVideoResult:
    """Output of one ``WanEngine.generate_video`` call.

    ``frames`` is the list of PIL frames (caller encodes to mp4 at ``fps``).
    ``frames`` is None when ``error`` is set.
    """

    frames: Optional[list["Image"]]
    fps: int
    seed: int
    variant: str
    width: int
    height: int
    num_frames: int
    error: Optional[WanGenerationError] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.frames is not None
