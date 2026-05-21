"""Abstract base for pipeline backends.

One subclass per engine family (sdxl, zimage, ...). Encapsulates the
differences between Diffusers pipeline types: loading, prompt encoding,
scheduler selection, img2img construction, and inference kwargs.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional


class PipelineBackend(ABC):
    """One backend per ModelConfig.engine value."""

    engine: ClassVar[str]
    """Identifier matching ModelConfig.engine ("sdxl" | "zimage" | ...)."""

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
    ) -> dict:
        """Build positional kwargs for pipe(...) for one batch chunk.

        SDXL passes clip_skip; Z-Image does not.
        """

    @abstractmethod
    def make_generator(self, seed: int, device: str) -> Any:
        """Build a torch.Generator on the right device for this backend.

        `device` is the engine's resolved torch device string ("cuda:0", "mps",
        "cpu"). Backends may ignore it (SDXL works fine with CPU generator) or
        honor it (Z-Image flow-matching needs the generator on the same device
        as the transformer).
        """
