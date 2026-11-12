"""Ovis-Image backend — Alibaba AIDC's 7B text-to-image model, a text-rendering specialist.

Own sub-package parallel to ``imference_engine.longcat`` / ``.ernieimage``.
Ovis-Image-7B is a FLUX-style flow-matching transformer (7B) conditioned on
**Ovis2.5-2B** (a ``Qwen3Model`` language tower) with the FLUX VAE. Apache-2.0,
ungated. ``OvisImagePipeline`` is in diffusers 0.40.

Undistilled: 50 steps of true classifier-free guidance (``guidance_scale=5.0``),
negative prompt honored. Text-to-image only — diffusers has no img2img / edit
pipeline for it (``make_img2img`` raises) and the pipeline carries no LoRA
loader.

Weights (``weights_path``): ``"auto"`` (``AIDC-AI/Ovis-Image-7B``), a repo id /
directory in the diffusers layout, or a transformer ``.safetensors`` in the
diffusers key layout. diffusers 0.40 has no single-file / GGUF loader for the
Ovis transformer, so the community GGUF files (stable-diffusion.cpp builds)
cannot be used: the low-memory builds store the bf16 weights in fp8 instead
(layerwise casting), for the transformer and for the encoder.

Env: ``OVIS_IMAGE_TRANSFORMER`` / ``OVIS_IMAGE_ENCODER`` = auto | bf16 | fp8
(auto follows the VRAM table of ``builds.py``).
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.ovisimage.builds import AUTO, recommend_build
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)

DEFAULT_REPO = "AIDC-AI/Ovis-Image-7B"


def _vram_gb() -> Optional[float]:
    from imference_engine.managers.builds import cuda_vram_gb
    return cuda_vram_gb()


class OvisImageBackend(PipelineBackend):
    """Backend for Ovis-Image checkpoints."""

    engine: ClassVar[str] = "ovis-image"

    # The diffusers part of the repo. Its root also holds the original
    # single-file checkpoint (29 GB) and the full Ovis2.5-2B VLM, which the
    # pipeline does not use.
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*",
        "vae/*",
        "transformer/*",
        # Apache-2.0 text + attribution notice, mirrored with the weights.
        "LICENSE", "NOTICE",
    ]
    # A transformer single file needs everything but the transformer weights.
    CONFIG_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*",
        "vae/*",
        "transformer/config.json",
        "LICENSE", "NOTICE",
    ]

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # Upstream recipe (undistilled, true CFG).
        return GenerationDefaults(guidance_scale=5.0, num_steps=50)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def resolve_weights(self, weights_path: str) -> str:
        """``"auto"`` -> the official repo."""
        return DEFAULT_REPO if weights_path == AUTO else weights_path

    def _repo_dir(self, repo: str, patterns: list) -> str:
        from imference_engine.runtime.offline import local_repo_dir

        if os.path.isdir(repo):
            return repo
        return local_repo_dir(repo, patterns, self._cache_dir, namespace="image",
                              cdn_base=self._cdn_base)

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import OvisImagePipeline, OvisImageTransformer2DModel
        from transformers import Qwen3Model

        from imference_engine.core.fp8 import apply_fp8_storage, fp8_transformer, resolve_storage

        checkpoint = self.resolve_weights(local_path)
        vram = _vram_gb()
        build = recommend_build(vram)

        if checkpoint.lower().endswith(".safetensors") and not os.path.isdir(checkpoint):
            repo_dir = self._repo_dir(base_model or DEFAULT_REPO, self.CONFIG_PATTERNS)
            transformer = self._load_single_file(checkpoint, os.path.join(repo_dir, "transformer"))
        else:
            repo_dir = self._repo_dir(checkpoint, self.BASE_PATTERNS)
            logger.info(f"Loading Ovis-Image transformer from {repo_dir}")
            transformer = OvisImageTransformer2DModel.from_pretrained(
                repo_dir, subfolder="transformer", local_files_only=True,
                torch_dtype=torch.bfloat16)
        if resolve_storage("OVIS_IMAGE_TRANSFORMER", build.transformer, vram) == "fp8":
            fp8_transformer(transformer, "Ovis-Image", logger)

        components: dict = {"transformer": transformer}
        if resolve_storage("OVIS_IMAGE_ENCODER", build.encoder, vram) == "fp8":
            encoder = Qwen3Model.from_pretrained(
                os.path.join(repo_dir, "text_encoder"), local_files_only=True,
                dtype=torch.bfloat16)
            components["text_encoder"] = apply_fp8_storage(encoder)
            logger.info("Ovis-Image text encoder: fp8-resident storage enabled")
        return OvisImagePipeline.from_pretrained(
            repo_dir, local_files_only=True, dtype=torch.bfloat16, **components)

    def _load_single_file(self, path: str, cfg_dir: str) -> Any:
        import accelerate
        import torch
        from diffusers import OvisImageTransformer2DModel
        from safetensors.torch import load_file

        logger.info(f"Loading Ovis-Image transformer from {path}")
        sd = {k: v.to(torch.bfloat16) for k, v in load_file(path).items()}
        with accelerate.init_empty_weights():
            transformer = OvisImageTransformer2DModel.from_config(
                OvisImageTransformer2DModel.load_config(cfg_dir, local_files_only=True))
        missing, unexpected = transformer.load_state_dict(sd, assign=True, strict=False)
        del sd
        if missing or unexpected:
            raise ValueError(
                f"{os.path.basename(path)} is not an Ovis-Image transformer in the diffusers "
                f"key layout ({len(missing)} missing, {len(unexpected)} unexpected keys; "
                f"first missing: {list(missing)[:3]})")
        return transformer

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        self._repo_dir(base_model or DEFAULT_REPO, self.CONFIG_PATTERNS)
        logger.info("Ovis-Image base-components warmed (base=%s)", base_model or DEFAULT_REPO)

    # ------------------------------------------------------------------
    # Img2img — unsupported (diffusers ships a text-to-image pipeline only)
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        raise NotImplementedError(
            "Ovis-Image has no diffusers img2img / edit pipeline (text-to-image "
            "only). Call generate() without source_image."
        )

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        return {"prompt": prompt, "negative_prompt": negative_prompt or ""}

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Flow matching with the repo's own shift config; the ``scheduler``
        name is ignored."""
        if scheduler:
            logger.debug("Ovis-Image ignores scheduler=%r (flow matching only)", scheduler)

    def build_inference_kwargs(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002 — no clip_skip
        chunk_size: int,
        generator: Any,
        image: Any = None,  # noqa: ARG002 — img2img unsupported (make_img2img raises)
        strength: float = 0.75,  # noqa: ARG002
    ) -> dict:
        return {
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
        }

    def make_generator(self, seed: int, device: str) -> Any:
        import torch
        try:
            return torch.Generator(device=device).manual_seed(seed)
        except (RuntimeError, ValueError) as e:
            logger.warning(
                f"Generator on device={device} failed ({e}); falling back to CPU"
            )
            return torch.Generator().manual_seed(seed)
