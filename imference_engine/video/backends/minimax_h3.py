"""MiniMaxH3Backend — the MiniMax-H3 architecture as a ``VideoBackend``.

The second video architecture next to Wan, wrapping ``minimax_h3/loader.py``.
Structural differences from Wan worth knowing:

- **No shared components.** Wan splits a shared UMT5/VAE base from per-variant
  GGUF experts; H3 is one self-contained modular repository per variant, so
  ``load_shared`` returns None and ``build`` ignores it.
- **One pipeline serves t2v AND i2v.** The FL2VA blocks route by request inputs
  (``image`` / ``last_image``), so both tasks share a single resident pipeline —
  the reason H3 has one builtin variant, not a t2v/i2v pair.
- **Guidance-distilled.** ``negative_prompt`` and ``guidance_scale`` do not
  exist for H3; the ABC-mandated arguments are accepted and ignored (with a
  warning when a caller actually set them).
"""
from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

from imference_engine.video.backend import VideoBackend, VideoBuildContext

logger = logging.getLogger(__name__)


class MiniMaxH3Backend(VideoBackend):
    """MiniMax-H3 (33B joint video+audio DiT, modular diffusers, int8/bf16)."""

    engine: ClassVar[str] = "minimax_h3"

    def load_shared(self, ctx: VideoBuildContext) -> Any:
        return None  # self-contained modular repo — nothing shared across variants

    def build(self, spec: Any, shared: Any, ctx: VideoBuildContext) -> Any:
        from imference_engine.minimax_h3.loader import build_pipeline
        return build_pipeline(
            spec,
            profile=ctx.quant,
            device=ctx.device,
            offload_mode=ctx.offload_mode or "block",
            vae_tiling=ctx.vae_tiling,
            cache_dir=ctx.cache_dir,
            cdn_base=ctx.cdn_base,
            attention_backend=ctx.attention_backend,
        )

    def build_call(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: Optional[int],
        height: Optional[int],
        num_frames: int,
        num_steps: int,
        guidance_scale: Optional[float],
        guidance_scale_2: Optional[float],
        image: Any,
        generator: Any,
        last_image: Any = None,
    ) -> dict:
        """Build the kwargs for ``pipe(**call)``. ``width``/``height`` may be
        None — the pipeline then derives the model's native 768-short-edge
        canvas from the first keyframe's aspect ratio (or 16:9 for t2v).
        ``last_image`` is the H3-specific end keyframe (beyond the ABC
        signature; optional, so ABC-shaped callers are unaffected)."""
        if negative_prompt:
            logger.warning(
                "MiniMax-H3 is guidance-distilled — negative_prompt is ignored")
        if guidance_scale is not None and guidance_scale != 1.0:
            logger.warning(
                "MiniMax-H3 is guidance-distilled — guidance_scale=%s is ignored",
                guidance_scale)

        call = dict(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            generator=generator,
            output_type="pil",
        )
        if width is not None and height is not None:
            call["width"] = width
            call["height"] = height
        # Keyframes go through as-is: the setup block EXIF-transposes, converts
        # to RGB and puts them onto the canvas itself (stretch for the first,
        # cover-crop for the last) — pre-resizing here would double-resample.
        if image is not None:
            call["image"] = image
        if last_image is not None:
            call["last_image"] = last_image
        return call

    def teardown(self, pipe: Any) -> None:
        from imference_engine.minimax_h3.loader import teardown
        teardown(pipe)
