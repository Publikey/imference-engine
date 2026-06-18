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

    # madebyollin/sdxl-vae-fp16-fix: SDXL VAE retrained to be numerically stable
    # in fp16 (no overflow → no NaNs). Its config sets force_upcast=False, so
    # diffusers decodes directly in fp16 with no Half/float dtype mismatch.
    FP16_VAE_REPO: ClassVar[str] = "madebyollin/sdxl-vae-fp16-fix"

    # Repo whose *config + tokenizers* drive from_single_file's layout inference.
    # The checkpoint supplies the unet/text_encoder/vae *weights*, but NOT the
    # model_index/scheduler config or the two CLIP tokenizers (vocab/merges) —
    # diffusers fetches those from this repo. Mirrored into the flat offline tree
    # so a cold load never touches HF. NEVER the weights (allow only config/tokenizer).
    CONFIG_REPO: ClassVar[str] = "stabilityai/stable-diffusion-xl-base-1.0"
    CONFIG_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "tokenizer_2/*",
        "text_encoder/config.json", "text_encoder_2/config.json",
        "vae/config.json",
    ]
    # A standalone VAE repo has only config.json + the weights file (no model_index).
    _VAE_PATTERNS: ClassVar[list] = ["config.json", "diffusion_pytorch_model.*"]

    def __init__(
        self, *, use_tiny_vae: bool = False,
        cache_dir: Optional[str] = None, cdn_base: Optional[str] = None,
    ) -> None:
        # Cache scheduler instances keyed by (id(pipe.scheduler.config), scheduler_name).
        # The id() key isolates per-pipe configs so multiple loaded models can each
        # have their own cached scheduler without cross-contamination.
        self._scheduler_cache: dict[tuple[int, str], Any] = {}
        self._use_tiny_vae = use_tiny_vae
        # Flat offline tree root (IMAGE_MODEL_CACHE) + optional CDN mirror — threaded
        # from RuntimeConfig by the Engine. The shared SDXL config + the fp16-fix VAE
        # are resolved into this tree so a cold load is fully offline (HF_HUB_OFFLINE=1).
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import StableDiffusionXLPipeline

        from imference_engine.runtime.offline import local_repo_dir

        # Pin the layout to a LOCAL config dir (+ local_files_only) so from_single_file
        # never reaches out to HF for model_index/scheduler/tokenizers.
        cfg_dir = local_repo_dir(
            self.CONFIG_REPO, self.CONFIG_PATTERNS, self._cache_dir, namespace="image")

        logger.info(f"Loading SDXL pipeline from {local_path} (config={cfg_dir})")
        pipe = StableDiffusionXLPipeline.from_single_file(
            local_path,
            config=cfg_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        # Some Illustrious-based fine-tunes leave bias params in float32 after
        # from_single_file, causing "Input type (Half) and bias type (float)"
        # mismatches at inference. Force every parameter to float16.
        pipe = pipe.to(torch.float16)

        # VAE handling — two mutually exclusive paths. Both resolve the VAE into
        # the flat offline tree first, then load from that LOCAL dir (no HF hit).
        if self._use_tiny_vae:
            # TAESDxl: ~5 MB fp16 autoencoder, ~10× faster decode at the cost of
            # slight quality loss (visible on photoreal, transparent on illust).
            # Replaces the lazy upcast_vae path entirely.
            from diffusers import AutoencoderTiny
            logger.info(f"Swapping VAE for TAESD ({self.TINY_VAE_REPO}) — faster decode, slight quality loss")
            vae_dir = local_repo_dir(
                self.TINY_VAE_REPO, self._VAE_PATTERNS, self._cache_dir,
                namespace="image", sentinel="config.json")
            pipe.vae = AutoencoderTiny.from_pretrained(
                vae_dir, torch_dtype=torch.float16, local_files_only=True
            )
        else:
            # Swap in the fp16-fix VAE. The previous `pipe.vae.to(torch.float32)`
            # left the VAE in fp32 while latents stayed fp16; diffusers 0.37 only
            # realigns latent dtype when needs_upcasting (vae fp16 + force_upcast)
            # is true, so a pre-cast VAE skipped that path and crashed at
            # post_quant_conv with "Input type (Half) and bias type (float)
            # should be the same". The fp16-fix VAE keeps everything in fp16 —
            # no upcast, no dtype mismatch, no decode-time churn.
            from diffusers import AutoencoderKL
            logger.info(f"Swapping VAE for fp16-fix ({self.FP16_VAE_REPO})")
            vae_dir = local_repo_dir(
                self.FP16_VAE_REPO, self._VAE_PATTERNS, self._cache_dir,
                namespace="image", sentinel="config.json")
            pipe.vae = AutoencoderKL.from_pretrained(
                vae_dir, torch_dtype=torch.float16, local_files_only=True
            )

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
