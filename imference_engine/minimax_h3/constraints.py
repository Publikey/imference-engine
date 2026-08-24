"""MiniMax-H3 generation constraints — pure-Python mirrors of the upstream checks.

The modular pipeline (diffusers >= 0.40.0) enforces these itself, but only once
the ~90 GB of components are loaded and the call is in flight. The engine mirrors
the *checks* here so a bad request fails fast with a clear message, and so
``MediaResult.num_frames`` reports the frame count actually generated (the
pipeline snaps ``num_frames`` up internally and only warns).

Mirrored from ``diffusers/modular_pipelines/minimax_h3/packing.py`` at the
``minimax-h3`` branch — if upstream changes these numbers, change them here too
(the pipeline remains the source of truth; a drift makes us reject/announce
wrongly, never generate wrongly).
"""
from __future__ import annotations

# Fixed output rate — H3 has no fps knob; ``fps`` in MediaResult is always this.
MINIMAX_H3_FPS = 24

# The video VAE decodes frame counts of the form 17*n + 5 only.
_FRAMES_PER_CHUNK = 17
_FRAMES_REMAINDER = 5

# Duration window (seconds) the model generates, checked on the *aligned* count.
MIN_DURATION_S = 5.0
MAX_DURATION_S = 15.0

# Canvas rules: both dims multiples of 32; the model's own canvas (when
# height/width are omitted) uses a 768 short edge for the keyframe's (or 16:9)
# aspect ratio. We never compute that canvas here — omitted dims are passed
# through as None and the pipeline resolves them.
CANVAS_MULTIPLE = 32
SHORT_EDGE = 768

# Default frame count (5.17 s) — the upstream blocks' own default.
DEFAULT_NUM_FRAMES = 124


def align_num_frames(num_frames: int) -> int:
    """Snap a frame count up to the next ``17*n + 5`` the video VAE can decode."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    remainder = (num_frames - _FRAMES_REMAINDER) % _FRAMES_PER_CHUNK
    if remainder:
        num_frames += _FRAMES_PER_CHUNK - remainder
    return num_frames


def check_num_frames(num_frames: int) -> int:
    """Validate + snap ``num_frames``; returns the aligned count the model will
    actually generate. Raises ValueError when the aligned duration leaves the
    5-15 s window (the upstream rule: the *aligned* count is what must fit, so
    e.g. 346 is rejected — it would round up to 362 = 15.083 s)."""
    aligned = align_num_frames(num_frames)
    duration = aligned / MINIMAX_H3_FPS
    if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
        raise ValueError(
            f"MiniMax-H3 generates {MIN_DURATION_S:g}-{MAX_DURATION_S:g} s at "
            f"{MINIMAX_H3_FPS} fps; num_frames={num_frames} aligns up to {aligned} "
            f"frames = {duration:.3f} s, outside that window. Valid aligned counts "
            f"run 124 (5.17 s) to 345 (14.375 s) in steps of 17."
        )
    return aligned


def check_canvas(width, height) -> None:
    """Validate the requested canvas: both dims or neither, multiples of 32.
    (None, None) is valid — the pipeline derives the model's own 768-short-edge
    canvas from the first keyframe's aspect ratio, or 16:9 without one."""
    if (width is None) != (height is None):
        raise ValueError("width and height must be passed together, or neither")
    if width is None:
        return
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(
            f"width and height must be multiples of {CANVAS_MULTIPLE}, got "
            f"{width}x{height} (the model's native canvas uses a {SHORT_EDGE} "
            f"short edge; smaller multiples-of-32 canvases like 960x544 trade "
            f"quality for ~2x+ speed)"
        )
