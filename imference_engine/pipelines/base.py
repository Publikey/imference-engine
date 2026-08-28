"""Abstract base for pipeline backends.

One subclass per engine family (sdxl, zimage, ...). Encapsulates the
differences between Diffusers pipeline types: loading, prompt encoding,
scheduler selection, img2img construction, and inference kwargs.
"""
from __future__ import annotations
from abc import abstractmethod
from typing import Any, Optional

from imference_engine.core.backend import Backend


class PipelineBackend(Backend):
    """Image backend — one subclass per pipeline family (sdxl, zimage, ...).

    Extends the shared ``Backend`` trunk (``engine`` id + ``engine_defaults``)
    with the image-specific interface: single-file loading, prompt encoding,
    scheduler selection, img2img construction, and inference kwargs.
    """

    # Whether Engine.generate(loras=...) applies user LoRAs on this backend's
    # pipes (via managers/lora.py — set_adapters, never fused, deactivated per
    # request). Every diffusers 0.40 pipeline HAS a LoRA mixin; this flag is
    # about what the ENGINE has validated: SDXL first, the others flip as each
    # is proven (watch the interaction with offload/fp8 hooks per backend).
    supports_loras: bool = False

    @abstractmethod
    def load_pipeline(
        self,
        *,
        local_path: str,
        base_model: Optional[str] = None,
    ) -> Any:
        """Load and return a diffusers text-to-image Pipeline.

        base_model is the HF repo id for shared components (used by Z-Image
        finetunes that ship as a single transformer .safetensors file but
        need tokenizer/text_encoder/vae from the base repo). Ignored by SDXL.
        """

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        """Download this backend's shared base-components into the offline tree
        WITHOUT loading any model. Lets a worker warm the base at deploy time so
        a fresh pod is 'ready' with the base on disk, and the first request only
        pays for the checkpoint weights (which stay lazy). Idempotent and
        offline-safe once populated. Default: no-op."""
        return None

    @abstractmethod
    def make_img2img(self, t2i_pipe: Any) -> Any:
        """Build an img2img pipeline sharing weights with the t2i pipeline.

        SDXL shares unet; Z-Image shares transformer. The implementation must
        reuse the in-memory weights, not reload from disk.
        """

    @abstractmethod
    def get_compute_module(self, pipe: Any) -> Any:
        """Return the main compute module (`pipe.unet` or `pipe.transformer`).

        Used for device residency checks (is the model on GPU?).
        """

    @abstractmethod
    def encode_prompts(
        self,
        pipe: Any,
        prompt: str,
        negative_prompt: Optional[str],
    ) -> dict:
        """Return kwargs to inject into pipe(...) for prompt encoding.

        SDXL returns weighted embeds (overcomes 77-token limit, supports BREAK).
        Z-Image returns plain {"prompt": ..., "negative_prompt": ...}.
        """

    @abstractmethod
    def apply_scheduler(
        self,
        pipe: Any,
        scheduler: Optional[str],
        **kwargs: Any,
    ) -> None:
        """Configure the scheduler on the pipeline in place.

        SDXL: DPMSolverMultistep / EulerAncestral with optional Karras sigmas.
        Z-Image: FlowMatchEulerDiscreteScheduler with `shift` (auto 3.0 for Turbo).
        """

    def apply_guidance(self, pipe: Any, guidance_scale: float) -> None:
        """Configure classifier-free guidance on the pipeline in place.

        Default: no-op. Standard ``DiffusionPipeline`` backends take
        ``guidance_scale`` as a ``pipe(...)`` kwarg (see ``build_inference_kwargs``),
        so there is nothing to set here. Modular-Diffusers backends (Anima) instead
        hold a ``ClassifierFreeGuidance`` *guider* component whose scale is read off
        the object at denoise time — it is NOT a call kwarg — so those backends
        override this to set the scale on the guider before the call.
        """
        return None

    @abstractmethod
    def build_inference_kwargs(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],
        chunk_size: int,
        generator: Any,
        image: Any = None,
        strength: float = 0.75,
    ) -> dict:
        """Build positional kwargs for pipe(...) for one batch chunk.

        SDXL passes clip_skip; Z-Image does not. When ``image`` is not None the
        call is img2img: include ``image`` + ``strength`` and let the pipeline
        take the output size from the source image (omit width/height).
        """

    @abstractmethod
    def make_generator(self, seed: int, device: str) -> Any:
        """Build a torch.Generator on the right device for this backend.

        `device` is the engine's resolved torch device string ("cuda:0", "mps",
        "cpu"). Backends may ignore it (SDXL works fine with CPU generator) or
        honor it (Z-Image flow-matching needs the generator on the same device
        as the transformer).
        """
