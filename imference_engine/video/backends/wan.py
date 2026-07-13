"""WanBackend — the Wan 2.2 video architecture as a ``VideoBackend``.

Wraps the existing ``wan/loader.py`` builder functions and the Wan-specific
teardown, so the residency manager (Phase 4c) can drive Wan through the generic
``VideoBackend`` interface — and a 2nd architecture (LTX-Video) plugs in as a
sibling subclass. Behaviour is identical to the pre-refactor inline path; this is
re-homing, not a rewrite.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from imference_engine.video.backend import VideoBackend, VideoBuildContext


class WanBackend(VideoBackend):
    """Wan 2.2 (GGUF-MoE, Lightning LoRA). One shared UMT5 + Wan VAE base."""

    engine: ClassVar[str] = "wan"

    # The shared UMT5 text_encoder + Wan VAE come from this diffusers repo (T2V &
    # I2V share them — P1g). Loaded once, reused by every variant.
    SHARED_BASE_REPO: ClassVar[str] = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    def load_shared(self, ctx: VideoBuildContext) -> Any:
        from imference_engine.wan.loader import load_shared_components
        return load_shared_components(
            self.SHARED_BASE_REPO, cache_dir=ctx.cache_dir,
            text_encoder_quant=ctx.text_encoder_quant, cdn_base=ctx.cdn_base)

    def build(self, spec: Any, shared: Any, ctx: VideoBuildContext) -> Any:
        from imference_engine.wan.loader import build_pipeline
        return build_pipeline(
            spec, quant=ctx.quant, shared=shared, device=ctx.device,
            enable_offload=ctx.enable_offload, vae_tiling=ctx.vae_tiling,
            cache_dir=ctx.cache_dir, cdn_base=ctx.cdn_base)

    def build_call(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        num_frames: int,
        num_steps: int,
        guidance_scale: float,
        guidance_scale_2: Optional[float],
        image: Any,
        generator: Any,
    ) -> dict:
        call = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            guidance_scale_2=(guidance_scale_2 if guidance_scale_2 is not None
                              else guidance_scale),
            generator=generator,
            output_type="pil",
        )
        if image is not None:
            # Preserve the image's aspect ratio within the width*height area budget,
            # aligned to the Wan latent grid (vae_scale_factor_spatial 8 * patch 2 =
            # 16) — the canonical diffusers i2v resize. A hard resize to
            # (width, height) squashed any aspect that didn't match the request.
            img = image.convert("RGB")
            mod = 16
            max_area = width * height
            ar = img.height / img.width
            h = max(mod, round((max_area * ar) ** 0.5) // mod * mod)
            w = max(mod, round((max_area / ar) ** 0.5) // mod * mod)
            call["image"] = img.resize((w, h))
            call["height"] = h  # generation dims follow the image aspect
            call["width"] = w
        return call

    def teardown(self, pipe: Any) -> None:
        """Release a built pipeline's RAM before the next variant builds.

        ``enable_model_cpu_offload`` leaves accelerate hooks + pinned host buffers
        that a bare ``del`` does NOT return, so a switch kept ~2 variants' weights
        in RAM -> OOM on small-RAM hosts even after eviction. Remove the hooks and
        drop the big per-variant MoE modules (the shared text_encoder/vae live on
        the SharedComponents and are deliberately left untouched)."""
        for fn in ("remove_all_hooks", "maybe_free_model_hooks"):
            try:
                getattr(pipe, fn)()
            except Exception:  # noqa: BLE001
                pass
        for attr in ("transformer", "transformer_2"):
            try:
                if getattr(pipe, attr, None) is not None:
                    setattr(pipe, attr, None)
            except Exception:  # noqa: BLE001
                pass
