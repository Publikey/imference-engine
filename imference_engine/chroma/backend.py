"""Chroma backend — FLUX-derived, de-distilled 8.9B flow-matching DiT.

Own sub-package parallel to ``imference_engine.flux``. Chroma is architecturally
a FLUX sibling (rectified-flow DiT, same VAE) but with two differences that
matter to this backend:

- **Single text encoder.** Chroma uses T5-XXL only — no CLIP-L, so no
  ``text_encoder_2`` / ``tokenizer_2`` (unlike FLUX's dual encoders).
- **Real classifier-free guidance.** Chroma is trained de-distilled, so
  ``guidance_scale`` is TRUE CFG and ``negative_prompt`` IS honored (unlike
  guidance-distilled FLUX.1, which ignores negatives). This backend therefore
  passes the negative prompt through, like Z-Image.

Apache-2.0 and a large "uncensored" community-finetune following make it a cheap,
high-value addition once FLUX is in: same diffusers 0.39 stack, same offline
base-component machinery, same heavy-transformer offload path.

Offline: base components (T5-XXL + VAE + tokenizer + scheduler) resolve into the
flat tree (``namespace="image"``) and load with ``local_files_only=True``. Run
the 8.9B transformer with ``RuntimeConfig(enable_offload=True)`` on consumer
VRAM (shared ModelManager path — no Chroma-specific code).
"""
from __future__ import annotations
import logging
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)


class ChromaBackend(PipelineBackend):
    """Backend for Chroma checkpoints (.safetensors single-file)."""

    engine: ClassVar[str] = "chroma"

    # Files needed from a Chroma base repo (diffusers format). T5-XXL text encoder
    # + its tokenizer + the VAE + scheduler are FULL weights/config. The
    # transformer is CONFIG-ONLY: from_single_file reads its layout here while the
    # WEIGHTS come from the checkpoint. NOTE: no text_encoder_2/tokenizer_2 —
    # Chroma has a single (T5) encoder.
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*",
        "vae/*",
        "transformer/config.json",
    ]

    MAX_SEQUENCE_LENGTH: ClassVar[int] = 512

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # De-distilled: real CFG. guidance_scale=2.0 validated on GPU as the sweet
        # spot — higher (the community's ~4.0) oversaturates colours on this stack.
        # negative_prompt is honored (see encode_prompts). scheduler is flow
        # matching (name ignored).
        return GenerationDefaults(guidance_scale=2.0, num_steps=28)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import ChromaPipeline

        if base_model:
            return self._load_with_base_model(local_path, base_model)

        logger.info(f"Loading Chroma pipeline from {local_path}")
        return ChromaPipeline.from_single_file(
            local_path,
            torch_dtype=torch.bfloat16,
        )

    def _load_with_base_model(self, local_path: str, base_repo: str) -> Any:
        """Transformer WEIGHTS from local_path + shared components (T5-XXL, VAE,
        tokenizer, scheduler) from the base repo, resolved into the flat offline
        tree (no HF hit once populated)."""
        import os

        import torch
        from diffusers import ChromaPipeline, ChromaTransformer2DModel

        from imference_engine.runtime.offline import local_repo_dir

        base_dir = local_repo_dir(
            base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
            cdn_base=self._cdn_base)
        logger.info(f"Loading Chroma shared components from {base_dir} (base={base_repo})")

        transformer = ChromaTransformer2DModel.from_single_file(
            local_path,
            config=os.path.join(base_dir, "transformer"),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        return ChromaPipeline.from_pretrained(
            base_dir,
            transformer=transformer,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        if not base_model:
            return
        from imference_engine.runtime.offline import local_repo_dir
        local_repo_dir(base_model, self.BASE_PATTERNS, self._cache_dir,
                       namespace="image", cdn_base=self._cdn_base)
        logger.info("Chroma base-components warmed (base=%s)", base_model)

    # ------------------------------------------------------------------
    # Img2img
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        from diffusers import ChromaImg2ImgPipeline
        return ChromaImg2ImgPipeline(
            scheduler=t2i_pipe.scheduler,
            vae=t2i_pipe.vae,
            text_encoder=t2i_pipe.text_encoder,
            tokenizer=t2i_pipe.tokenizer,
            transformer=t2i_pipe.transformer,
        )

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        # De-distilled → real CFG → negative prompt is honored. T5 has no
        # 77-token limit, so raw strings (no weighted/BREAK chunking).
        return {"prompt": prompt, "negative_prompt": negative_prompt or ""}

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any
    ) -> None:
        """Chroma uses FlowMatchEulerDiscreteScheduler (set at load). The
        `scheduler` name is ignored; an explicit `shift` via backend_options
        rebuilds it with a fixed shift (advanced)."""
        shift = kwargs.get("shift")
        if shift is None:
            return
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, shift=float(shift), use_dynamic_shifting=False
        )
        logger.info(f"Chroma scheduler: FlowMatchEuler fixed shift={shift}")

    def build_inference_kwargs(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002 — Chroma has no clip_skip
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
            "max_sequence_length": self.MAX_SEQUENCE_LENGTH,
        }
        if image is not None:
            kwargs["image"] = image
            kwargs["strength"] = strength
        else:
            kwargs["width"] = width
            kwargs["height"] = height
        return kwargs

    def make_generator(self, seed: int, device: str) -> Any:
        import torch
        try:
            return torch.Generator(device=device).manual_seed(seed)
        except (RuntimeError, ValueError) as e:
            logger.warning(
                f"Generator on device={device} failed ({e}); falling back to CPU"
            )
            return torch.Generator().manual_seed(seed)
