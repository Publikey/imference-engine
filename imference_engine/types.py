"""Result types returned by Engine.generate()."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class GenerationError:
    error: str
    batch_index: int
    seed: Optional[int]


@dataclass
class GenerationResult:
    """Output of one Engine.generate() call.

    images[i] is None when errors contains an entry with batch_index=i.
    seeds[i] is the seed used (or attempted) for the i-th image.
    """
    images: list[Optional["Image"]]
    seeds: list[int]
    errors: list[GenerationError] = field(default_factory=list)
