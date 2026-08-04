"""MiniMax-H3 variant definitions.

Unlike Wan — where t2v and i2v are different checkpoints — one H3 checkpoint
serves both tasks: the FL2VA partition covers ``t2va`` (text only) and ``fl2va``
(first and/or last keyframe), and the modular blocks route by which inputs the
request carries. So a variant here is just *which repository* (official bf16, or
a pre-quantized int8 mirror staged with ``validation/stage_h3_int8.py``) plus
its step recipe, and the engine decides t2v-vs-i2v per request from ``image`` /
``last_image``. The Ref2VA partition (``transformer_ref/``, omni-references) is
deliberately out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass

# Official diffusers-format repository (both partitions + shared components;
# only the FL2VA half is ever fetched — see loader._H3_PATTERNS).
OFFICIAL_REPO = "MiniMaxAI/MiniMax-H3"

# H3 is guidance-distilled but NOT step-distilled; there is no documented step
# recipe yet, so we start at the flow-matching de-facto standard. NOTE: H3
# counts sigma grid points terminal-zero included, so this runs num_steps - 1
# model evaluations. Refine per-variant (catalog) once validated numbers exist.
DEFAULT_NUM_STEPS = 50


@dataclass
class H3Variant:
    """A loadable MiniMax-H3 model variant. Serves t2v AND i2v (see module doc)."""

    name: str
    repo: str = OFFICIAL_REPO       # modular diffusers repo id or local dir;
                                    # bf16 official or a pre-quantized int8 mirror
    arch: str = "minimax_h3"        # video backend key — routes to a VideoBackend
    num_steps: int = DEFAULT_NUM_STEPS

    def __post_init__(self) -> None:
        if not self.repo:
            raise ValueError(f"variant {self.name!r}: 'repo' must be non-empty")
        if self.num_steps < 2:
            # steps counts sigma grid points including the terminal 0 — below 2
            # there is no model evaluation at all.
            raise ValueError(
                f"variant {self.name!r}: num_steps must be >= 2, got {self.num_steps}")


BUILTIN_VARIANTS: dict[str, H3Variant] = {
    "minimax-h3": H3Variant(name="minimax-h3"),
}


def variant_from_catalog(cfg) -> H3Variant:
    """Build an ``H3Variant`` from a ``catalog.loader.VideoModelConfig`` (a
    ``kind: video`` row with ``engine: minimax_h3``). ``repo`` is optional and
    defaults to the official repository — a row exists mostly to point at a
    pre-quantized mirror or to pin a step recipe.

    The row's required ``mode`` field is accepted but not binding for H3 (one
    checkpoint serves both tasks); write ``mode: t2v`` by convention.

    Catalog row example::

        - name: h3-int8
          kind: video
          engine: minimax_h3
          mode: t2v
          repo: my-org/MiniMax-H3-int8
          num_steps: 40
    """
    s = cfg.spec
    return H3Variant(
        name=cfg.name,
        repo=s.get("repo", OFFICIAL_REPO),
        arch=cfg.arch,
        num_steps=s.get("num_steps", DEFAULT_NUM_STEPS),
    )
