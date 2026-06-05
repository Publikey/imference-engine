"""SDXL backend — Stable Diffusion XL pipelines from single .safetensors files."""
from __future__ import annotations
import logging
from typing import Any, ClassVar, Optional

from imference_engine.pipelines.base import PipelineBackend
from imference_engine.prompting.weighted import encode_sdxl_weighted

logger = logging.getLogger(__name__)


class SDXLBackend(PipelineBackend):
    """Backend for SDXL checkpoints (.safetensors single-file).

    Knobs (set at construction time, propagated by Engine from RuntimeConfig):
      use_tiny_vae : swap the heavyweight VAE for TAESDxl (~5 MB, fp16-native).
                     Decode goes from ~20 s to ~2 s on tight-VRAM GPUs at the
                     cost of slight quality loss on photo-realistic prompts.
                     Tolerated well on anime/illustration models.
    """

    engine: ClassVar[str] = "sdxl"

    # HuggingFace repo id for the SDXL tiny autoencoder. Tiny (~5 MB), fp16-safe,
    # designed by madebyollin specifically as a fast drop-in for SDXL's VAE.
    TINY_VAE_REPO: ClassVar[str] = "madebyollin/taesdxl"

    def __init__(self, *, use_tiny_vae: bool = False) -> None:
        # Cache scheduler instances keyed by (id(pipe.scheduler.config), scheduler_name).
        # The id() key isolates per-pipe configs so multiple loaded models can each
        # have their own cached scheduler without cross-contamination.
        self._scheduler_cache: dict[tuple[int, str], Any] = {}
        self._use_tiny_vae = use_tiny_vae

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import StableDiffusionXLPipeline

        logger.info(f"Loading SDXL pipeline from {local_path}")
        pipe = StableDiffusionXLPipeline.from_single_file(
            local_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        # Some Illustrious-based fine-tunes leave bias params in float32 after
        # from_single_file, causing "Input type (Half) and bias type (float)"
        # mismatches at inference. Force every parameter to float16.
        pipe = pipe.to(torch.float16)

        # VAE handling — two mutually exclusive paths:
        if self._use_tiny_vae:
            # TAESDxl: ~5 MB fp16 autoencoder, ~10× faster decode at the cost of
            # slight quality loss (visible on photoreal, transparent on illust).
            # Replaces the lazy upcast_vae path entirely.
            from diffusers import AutoencoderTiny
            logger.info(f"Swapping VAE for TAESD ({self.TINY_VAE_REPO}) — faster decode, slight quality loss")
            pipe.vae = AutoencoderTiny.from_pretrained(
                self.TINY_VAE_REPO, torch_dtype=torch.float16
            )
        else:
            # Full VAE pre-cast to fp32. Diffusers' SDXL pipeline otherwise does
            # an implicit upcast at decode time (the source of the
            # `FutureWarning: upcast_vae is deprecated` log line we see every
            # generation), which allocates +300 MB transiently and adds latency.
            # Pre-casting once at load makes that allocation persistent (steady
            # state, no decode-time churn) and shaves a few seconds per gen.
            try:
                pipe.vae.to(torch.float32)
                logger.info("VAE pre-cast to float32 (bypasses upcast_vae lazy path)")
            except Exception as e:
                logger.warning(f"VAE fp32 pre-cast failed (non-fatal): {e}")

        # channels_last memory format on the U-Net: trades a few MB of metadata
        # for ~10-20% throughput on Ampere+ conv-heavy nets. Memory format is
        # preserved across .to(device) calls, so applying it here (still on
        # CPU) carries through to the GPU promotion in ModelManager.
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            logger.info("U-Net moved to channels_last memory format")
        except Exception as e:
            logger.warning(f"channels_last on U-Net failed (non-fatal): {e}")

        return pipe

    def make_img2img(self, t2i_pipe: Any) -> Any:
        from diffusers import StableDiffusionXLImg2ImgPipeline
        return StableDiffusionXLImg2ImgPipeline(
            vae=t2i_pipe.vae,
            text_encoder=t2i_pipe.text_encoder,
            text_encoder_2=t2i_pipe.text_encoder_2,
            tokenizer=t2i_pipe.tokenizer,
            tokenizer_2=t2i_pipe.tokenizer_2,
            unet=t2i_pipe.unet,
            scheduler=t2i_pipe.scheduler,
        )

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.unet

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        return encode_sdxl_weighted(pipe, prompt, negative_prompt)

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any
    ) -> None:
        from diffusers import (
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
        )

        config_id = id(pipe.scheduler.config)
        name = scheduler or "EulerAncestralDiscreteScheduler"

        if name == "DPMSolverMultistepScheduler":
            key = (config_id, "dpm_karras")
            if key not in self._scheduler_cache:
                self._scheduler_cache[key] = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config, use_karras_sigmas=True
                )
        elif name == "EulerAncestralDiscreteScheduler":
            key = (config_id, "euler_ancestral")
            if key not in self._scheduler_cache:
                self._scheduler_cache[key] = EulerAncestralDiscreteScheduler.from_config(
                    pipe.scheduler.config
                )
        else:
            # Unknown name → Euler ancestral + Karras (matches worker fallback)
            key = (config_id, "euler_ancestral_karras")
            if key not in self._scheduler_cache:
                self._scheduler_cache[key] = EulerAncestralDiscreteScheduler.from_config(
                    pipe.scheduler.config, use_karras_sigmas=True
                )

        pipe.scheduler = self._scheduler_cache[key]

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
        kwargs = {
            "width": width,
            "height": height,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
        }
        if clip_skip is not None and clip_skip > 0:
            kwargs["clip_skip"] = clip_skip
        return kwargs

    def make_generator(self, seed: int, device: str) -> Any:  # noqa: ARG002
        import torch
        # CPU generator works fine for SDXL and avoids device-pool contention.
        return torch.Generator().manual_seed(seed)
