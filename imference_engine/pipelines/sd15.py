"""SD 1.5 backend — Stable Diffusion 1.5 pipelines from single .safetensors files.

Sibling of ``SDXLBackend`` (same ``pipelines/`` home). Structurally ~80 % the
same — from_single_file, scheduler cache, channels_last, img2img wrapper — but a
genuinely DIFFERENT diffusers pipeline and architecture, NOT a reuse of SDXL:

- ``StableDiffusionPipeline`` (vs XL), single CLIP-L text encoder + one tokenizer
  (vs SDXL's two), no pooled/time-id machinery, 512 native.
- Prompt encoding returns ``prompt_embeds`` only (no ``pooled_prompt_embeds``).

A shared ``_StableDiffusionBackendBase`` extraction between this and SDXL is a
sensible follow-up, deferred so the production SDXL load path stays untouched
(it can't be re-validated e2e without a GPU here).

SD 1.5 remains the deepest community-finetune / LoRA base and the lightest image
model to serve — a good desktop / low-VRAM default.
"""
from __future__ import annotations
import logging
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.pipelines.base import PipelineBackend
from imference_engine.prompting.weighted import encode_sd15_weighted

logger = logging.getLogger(__name__)


class SD15Backend(PipelineBackend):
    """Backend for Stable Diffusion 1.5 checkpoints (.safetensors single-file).

    Knobs (set at construction, propagated by Engine from RuntimeConfig):
      use_tiny_vae : swap the VAE for TAESD (~5 MB, fp16-native) — fast decode,
                     slight quality loss. SD 1.5's own tiny autoencoder
                     (madebyollin/taesd), distinct from SDXL's taesdxl.
    """

    engine: ClassVar[str] = "sd15"

    # SD 1.5 tiny autoencoder (NOT taesdxl). ~5 MB, fp16-safe.
    TINY_VAE_REPO: ClassVar[str] = "madebyollin/taesd"

    # Repo whose config + tokenizer drive from_single_file's layout inference.
    # The checkpoint supplies unet/text_encoder/vae WEIGHTS; diffusers reads the
    # model_index/scheduler config + the single CLIP tokenizer from here. The
    # original runwayml/stable-diffusion-v1-5 was delisted; this is the community
    # re-upload with the identical layout. Config/tokenizer only — never weights.
    CONFIG_REPO: ClassVar[str] = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    CONFIG_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*",
        # config.json of every sub-model (text_encoder, unet, vae). Glob on
        # config.json only → never pulls weights.
        "*/config.json",
        # The SD 1.5 model_index lists a feature_extractor (CLIPImageProcessor,
        # for the safety checker). Even with safety_checker=None, from_single_file
        # tries to load it and needs feature_extractor/preprocessor_config.json —
        # which */config.json does NOT match. Mirror it so the cold load resolves.
        "feature_extractor/*",
    ]
    _VAE_PATTERNS: ClassVar[list] = ["config.json", "diffusion_pytorch_model.safetensors"]

    def __init__(
        self, *, use_tiny_vae: bool = False,
        cache_dir: Optional[str] = None, cdn_base: Optional[str] = None,
    ) -> None:
        self._scheduler_cache: dict[tuple[int, str], Any] = {}
        self._use_tiny_vae = use_tiny_vae
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # SD 1.5 family norms: 512 native, CFG ~7, EulerAncestral. num_steps falls
        # through to GLOBAL_DEFAULTS (28 — fine for SD 1.5's 20-30 range).
        return GenerationDefaults(
            width=512, height=512, guidance_scale=7.0,
            scheduler="EulerAncestralDiscreteScheduler",
        )

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import StableDiffusionPipeline

        from imference_engine.runtime.offline import local_repo_dir

        cfg_dir = local_repo_dir(
            self.CONFIG_REPO, self.CONFIG_PATTERNS, self._cache_dir,
            namespace="image", cdn_base=self._cdn_base)

        logger.info(f"Loading SD 1.5 pipeline from {local_path} (config={cfg_dir})")
        pipe = StableDiffusionPipeline.from_single_file(
            local_path,
            config=cfg_dir,
            local_files_only=True,
            torch_dtype=torch.float16,
            use_safetensors=True,
            # SD 1.5 config ships a safety_checker + feature_extractor we don't
            # want to load or gate generation on. safety_checker=None disables the
            # NSFW gate; feature_extractor=None skips the CLIPImageProcessor it
            # feeds (only used by the safety checker). feature_extractor/* is also
            # mirrored (CONFIG_PATTERNS) as a fallback if the loader still probes it.
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(torch.float16)

        # VAE: SD 1.5 checkpoints ship a working fp16 VAE, so unlike SDXL (which
        # MUST swap in the fp16-fix VAE) the default keeps the checkpoint's own
        # VAE. use_tiny_vae swaps in TAESD for fast preview decode.
        if self._use_tiny_vae:
            from diffusers import AutoencoderTiny
            logger.info(f"Swapping VAE for TAESD ({self.TINY_VAE_REPO}) — faster decode, slight quality loss")
            vae_dir = local_repo_dir(
                self.TINY_VAE_REPO, self._VAE_PATTERNS, self._cache_dir,
                namespace="image", sentinel="config.json", cdn_base=self._cdn_base)
            pipe.vae = AutoencoderTiny.from_pretrained(
                vae_dir, torch_dtype=torch.float16, local_files_only=True
            )

        try:
            pipe.unet.to(memory_format=torch.channels_last)
            logger.info("U-Net moved to channels_last memory format")
        except Exception as e:
            logger.warning(f"channels_last on U-Net failed (non-fatal): {e}")

        return pipe

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        """Pull the SD 1.5 config repo (+ the tiny VAE if enabled) into the
        offline tree (no model load)."""
        from imference_engine.runtime.offline import local_repo_dir
        local_repo_dir(self.CONFIG_REPO, self.CONFIG_PATTERNS, self._cache_dir,
                       namespace="image", cdn_base=self._cdn_base)
        if self._use_tiny_vae:
            local_repo_dir(self.TINY_VAE_REPO, self._VAE_PATTERNS, self._cache_dir,
                           namespace="image", sentinel="config.json", cdn_base=self._cdn_base)
        logger.info("SD 1.5 base-components warmed (config=%s)", self.CONFIG_REPO)

    def make_img2img(self, t2i_pipe: Any) -> Any:
        from diffusers import StableDiffusionImg2ImgPipeline
        return StableDiffusionImg2ImgPipeline(
            vae=t2i_pipe.vae,
            text_encoder=t2i_pipe.text_encoder,
            tokenizer=t2i_pipe.tokenizer,
            unet=t2i_pipe.unet,
            scheduler=t2i_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.unet

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        return encode_sd15_weighted(pipe, prompt, negative_prompt)

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
        image: Any = None,
        strength: float = 0.75,
    ) -> dict:
        kwargs = {
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
        }
        if image is not None:
            kwargs["image"] = image
            kwargs["strength"] = strength
        else:
            kwargs["width"] = width
            kwargs["height"] = height
        if clip_skip is not None and clip_skip > 0:
            kwargs["clip_skip"] = clip_skip
        return kwargs

    def make_generator(self, seed: int, device: str) -> Any:  # noqa: ARG002
        import torch
        # CPU generator works fine for SD 1.5 and avoids device-pool contention.
        return torch.Generator().manual_seed(seed)
