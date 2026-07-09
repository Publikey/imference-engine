"""Unified result type for image AND video generation.

Replaces the image-side ``GenerationResult`` and the video-side
``WanVideoResult`` with one ``MediaResult``. Both modalities produce a list of
PIL frames per call — for images that list is the batch (N independent images,
one seed each); for video it is the frames of the single clip (one seed, plus
``fps`` / ``num_frames``). ``kind`` disambiguates, and ``.images`` / ``.frames``
read naturally per modality.

Engine/transport split unchanged: the engine returns frames + metadata; the
caller encodes/uploads. No file I/O here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass
class GenerationError:
    """One failed output. ``batch_index`` is the image batch position (None for
    video, which produces a single clip)."""

    error: str
    seed: Optional[int] = None
    batch_index: Optional[int] = None


@dataclass
class MediaResult:
    """Output of one ``generate`` (image) or ``generate_video`` (video) call.

    ``media`` is the produced frames: for ``kind="image"`` it's the batch of N
    images (``media[i] is None`` where ``errors`` has a matching ``batch_index``);
    for ``kind="video"`` it's the frames of the single clip (empty on failure).
    ``seeds`` is N seeds (image) or ``[seed]`` (video). ``fps`` / ``num_frames`` /
    ``width`` / ``height`` / ``variant`` are video metadata (None for image).
    """

    kind: str  # "image" | "video"
    media: list[Optional["Image"]]
    seeds: list[int]
    errors: list[GenerationError] = field(default_factory=list)
    # Video-only metadata (None for images).
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    variant: Optional[str] = None

    # Modality-appropriate aliases — both point at ``media``.
    @property
    def images(self) -> list[Optional["Image"]]:
        return self.media

    @property
    def frames(self) -> list[Optional["Image"]]:
        return self.media

    @property
    def error(self) -> Optional[GenerationError]:
        """First error, or None — convenient for the single-clip video path."""
        return self.errors[0] if self.errors else None

    @property
    def ok(self) -> bool:
        """No errors and at least one non-None frame/image produced."""
        return not self.errors and any(m is not None for m in self.media)
