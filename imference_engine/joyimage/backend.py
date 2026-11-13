"""JoyAI-Image-Edit-Plus backend — JD's 16B model for multi-image instruction editing.

Own sub-package parallel to ``imference_engine.longcat``. JoyAI-Image-Edit-Plus
is a 16B MMDiT conditioned on **Qwen3-VL-8B** (which also sees the reference
images) with the Wan VAE. Apache-2.0, ungated. ``JoyImageEditPlusPipeline`` is
in diffusers 0.40.

Engine surface: ``Engine.generate(model=..., prompt=<instruction>,
source_image=<image or list of 1-5 images>)``. Several references can be
combined ("the person from image 2 in the scene of image 1"). There is no
text-to-image mode and no ``strength``; the output size follows the LAST
reference (the pipeline's 1024-base buckets).

Undistilled: 30 steps of true classifier-free guidance (``guidance_scale=4.0``),
negative prompt honored — 60 passes of a 16B transformer per image.

The pipeline renders ONE image per call (it has no ``num_images_per_prompt``),
so ``max_batch_per_call = 1``: a batch is rendered image by image.

Weights (``weights_path``): ``"auto"`` (``jdopensource/JoyAI-Image-Edit-Plus-Diffusers``)
or a repo id / directory in the diffusers layout. diffusers 0.40 has no
single-file / GGUF loader for this transformer; the low-memory builds store the
bf16 weights in fp8 (layerwise casting), transformer and encoder.

Env: ``JOYIMAGE_TRANSFORMER`` / ``JOYIMAGE_ENCODER`` = auto | bf16 | fp8.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.joyimage.builds import AUTO, recommend_build
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)

DEFAULT_REPO = "jdopensource/JoyAI-Image-Edit-Plus-Diffusers"


def _vram_gb() -> Optional[float]:
    from imference_engine.managers.builds import cuda_vram_gb
    return cuda_vram_gb()


class JoyImageEditBackend(PipelineBackend):
    """Backend for JoyAI-Image-Edit-Plus checkpoints."""

    engine: ClassVar[str] = "joyimage-edit"

    # The whole diffusers repo (no shared base: one checkpoint).
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "processor/*", "text_encoder/*",
        "vae/*",
        "transformer/*",
        "LICENSE*", "NOTICE*",
    ]

    # Every request is an edit (needs a source image); tools key off this.
    instruction_edit: ClassVar[bool] = True

    # The pipeline has no num_images_per_prompt: one image per call.
    max_batch_per_call: ClassVar[int] = 1

    MAX_REFERENCES: ClassVar[int] = 5

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # Upstream recipe (undistilled, true CFG).
        return GenerationDefaults(guidance_scale=4.0, num_steps=30)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def resolve_weights(self, weights_path: str) -> str:
        """``"auto"`` -> the official diffusers repo."""
        return DEFAULT_REPO if weights_path == AUTO else weights_path

    def _repo_dir(self, repo: str) -> str:
        from imference_engine.runtime.offline import local_repo_dir

        if os.path.isdir(repo):
            return repo
        return local_repo_dir(repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
                              cdn_base=self._cdn_base)

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None  # noqa: ARG002 — one repo
    ) -> Any:
        import torch
        from diffusers import JoyImageEditPlusPipeline, JoyImageEditPlusTransformer3DModel
        from transformers import Qwen3VLForConditionalGeneration

        from imference_engine.core.fp8 import apply_fp8_storage, fp8_transformer, resolve_storage

        repo_dir = self._repo_dir(self.resolve_weights(local_path))
        vram = _vram_gb()
        build = recommend_build(vram)

        logger.info(f"Loading JoyAI-Image-Edit-Plus from {repo_dir}")
        transformer = JoyImageEditPlusTransformer3DModel.from_pretrained(
            repo_dir, subfolder="transformer", local_files_only=True, torch_dtype=torch.bfloat16)
        if resolve_storage("JOYIMAGE_TRANSFORMER", build.transformer, vram) == "fp8":
            fp8_transformer(transformer, "JoyAI-Image-Edit-Plus", logger)

        components: dict = {"transformer": transformer}
        if resolve_storage("JOYIMAGE_ENCODER", build.encoder, vram) == "fp8":
            encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                os.path.join(repo_dir, "text_encoder"), local_files_only=True,
                dtype=torch.bfloat16)
            components["text_encoder"] = apply_fp8_storage(encoder)
            logger.info("JoyAI-Image-Edit-Plus text encoder: fp8-resident storage enabled")
        return JoyImageEditPlusPipeline.from_pretrained(
            repo_dir, local_files_only=True, dtype=torch.bfloat16, **components)

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        self._repo_dir(base_model or DEFAULT_REPO)
        logger.info("JoyAI-Image-Edit-Plus warmed (repo=%s)", base_model or DEFAULT_REPO)

    def make_img2img(self, t2i_pipe: Any) -> Any:
        # The resident pipe already is the image-conditioned edit pipeline.
        return t2i_pipe

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        out = {"prompt": prompt}
        if negative_prompt:
            out["negative_prompt"] = negative_prompt
        return out

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Flow matching with the repo's fixed shift; the ``scheduler`` name is
        ignored."""
        if scheduler:
            logger.debug("JoyAI-Image ignores scheduler=%r (flow matching only)", scheduler)

    def build_inference_kwargs(
        self,
        *,
        width: int,  # noqa: ARG002 — the output follows the last reference image
        height: int,  # noqa: ARG002
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002
        chunk_size: int,  # noqa: ARG002 — always 1 (max_batch_per_call)
        generator: Any,
        image: Any = None,
        strength: float = 0.75,  # noqa: ARG002 — no strength: the images are conditions
    ) -> dict:
        if image is None:
            raise ValueError(
                "JoyAI-Image-Edit-Plus needs at least one input image (instruction "
                "editing has no text-to-image mode)")
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        if len(images) > self.MAX_REFERENCES:
            raise ValueError(
                f"JoyAI-Image-Edit-Plus takes at most {self.MAX_REFERENCES} reference "
                f"images ({len(images)} given)")
        return {
            "images": [im.convert("RGB") if hasattr(im, "convert") else im for im in images],
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
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
