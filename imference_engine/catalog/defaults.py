"""Generation defaults + the engine < model < request precedence chain.

Three layers carry generation params, resolved general -> fine:

  1. **Engine defaults** — ``PipelineBackend.engine_defaults()``. Per engine
     FAMILY (all SDXL -> EulerAncestral; all Z-Image -> flow matching).
  2. **Model defaults** — a catalog row's ``defaults:`` block. Per CHECKPOINT
     (this Z-Image-Turbo -> shift=3.0, steps=8).
  3. **Request params** — the explicit args to ``Engine.generate()``. Per CALL.

The effective value is the first non-``None`` walking ``request -> model ->
engine -> GLOBAL_DEFAULTS`` (finest wins). ``None`` means "not set at this
layer", so a lower layer can fill it — which is why ``Engine.generate()`` takes
``Optional`` params instead of concrete signature defaults (a concrete default
would look "explicit" and permanently shadow the model layer).

Engine INVARIANTS (dtype, attention, memory_format) are NOT part of this chain —
they stay hard-coded in the backend and cannot be overridden.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class GenerationDefaults:
    """Overridable generation params. Every field ``Optional``; ``None`` = unset.

    ``backend_options`` is a free dict for engine-specific knobs (e.g. Z-Image
    ``shift``) kept out of the cross-engine fields so the core stays agnostic.
    It merges KEY-WISE across layers rather than all-or-nothing.
    """

    num_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    scheduler: Optional[str] = None
    clip_skip: Optional[int] = None
    negative_prompt: Optional[str] = None
    strength: Optional[float] = None
    backend_options: dict = field(default_factory=dict)

    def merged_over(self, base: "GenerationDefaults") -> "GenerationDefaults":
        """Return a new defaults where ``self`` wins wherever it is set and
        ``base`` fills the rest. ``backend_options`` merges key-wise (``self``
        overrides ``base`` per key). Neither input is mutated."""
        out: dict = {}
        for f in fields(self):
            if f.name == "backend_options":
                out[f.name] = {**base.backend_options, **self.backend_options}
            else:
                sv = getattr(self, f.name)
                out[f.name] = sv if sv is not None else getattr(base, f.name)
        return GenerationDefaults(**out)


# The bottom layer: concrete fallbacks for the params that MUST be non-None by
# the time inference runs (a request may omit them and no engine/model layer may
# set them). Kept in ONE place so the magic numbers aren't duplicated across
# backends. scheduler / clip_skip / negative_prompt legitimately stay None here —
# the backends handle None downstream (SDXL falls back to EulerAncestral, prompt
# encoders treat None as "").
GLOBAL_DEFAULTS = GenerationDefaults(
    num_steps=28,
    guidance_scale=6.0,
    width=1024,
    height=1024,
    strength=0.75,
)
