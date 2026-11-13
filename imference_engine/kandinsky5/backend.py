"""Kandinsky 5.0 Image Lite backends — a 6B flow-matching DiT, text-to-image and editing.

Own sub-package parallel to ``imference_engine.longcat``. Kandinsky 5.0 Image
Lite pairs a 6B transformer with **Qwen2.5-VL-7B** and **CLIP ViT-L** text
encoders and the FLUX VAE. MIT, ungated. Both pipelines are in diffusers 0.40.
Two backends, one per pipeline:

- ``kandinsky5`` — ``Kandinsky5T2IPipeline``, text-to-image;
- ``kandinsky5-edit`` — ``Kandinsky5I2IPipeline``, instruction editing of
  ``source_image``.

Both are undistilled: 50 steps of true classifier-free guidance
(``guidance_scale=3.5``), negative prompt honored.

Weights (``weights_path``): ``"auto"`` (the backend's default ``*-sft-Diffusers``
repo), a repo id / directory in the diffusers layout, or a transformer
``.safetensors`` in the diffusers key layout. There is no single-file / GGUF
loader for this transformer in diffusers 0.40: the low-memory builds store the
bf16 weights in fp8 (layerwise casting), for the transformer and for the Qwen
encoder.

The T2I and I2I repos ship the same encoders and VAE. As for LongCat, a
checkpoint contributes only ``model_index.json``, ``scheduler/`` and
``transformer/``; the rest comes from ``base_model`` (default: the T2I repo).
The Qwen2.5-VL encoder is the same file as LongCat-Image's, too.

LoRAs: the pipelines carry a loader, not enabled until validated
(``supports_loras = False``).

Env: ``KANDINSKY5_TRANSFORMER`` / ``KANDINSKY5_ENCODER`` = auto | bf16 | fp8.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.kandinsky5.builds import AUTO, recommend_build
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"


def _vram_gb() -> Optional[float]:
    from imference_engine.managers.builds import cuda_vram_gb
    return cuda_vram_gb()


def _with_images_output(base_cls: Any) -> Any:
    """``base_cls`` whose call result exposes ``.images``.

    The Kandinsky 5 image pipelines return a ``KandinskyImagePipelineOutput``
    whose field is ``image`` (singular); the engine — like every other diffusers
    image pipeline — reads ``.images``."""
    import types

    class Kandinsky5EnginePipeline(base_cls):
        def __call__(self, *args, **kwargs):
            out = super().__call__(*args, **kwargs)
            if hasattr(out, "images"):
                return out
            images = out.image
            return types.SimpleNamespace(
                images=list(images) if isinstance(images, (list, tuple)) else [images])

    Kandinsky5EnginePipeline.__name__ = base_cls.__name__
    return Kandinsky5EnginePipeline


class Kandinsky5Backend(PipelineBackend):
    """Backend for Kandinsky 5.0 Image Lite text-to-image checkpoints."""

    engine: ClassVar[str] = "kandinsky5"

    DEFAULT_CHECKPOINT: ClassVar[str] = "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"

    # Shared components (base repo). text_encoder_2 is listed file by file: the
    # repo also carries the CLIP weights as flax / tf / pickle (5 GB of copies).
    SHARED_PATTERNS: ClassVar[list] = [
        "text_encoder/*", "tokenizer/*",
        "text_encoder_2/config.json", "text_encoder_2/model.safetensors",
        "tokenizer_2/*", "vae/*",
    ]
    # Per-checkpoint files.
    CHECKPOINT_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*", "transformer/*",
    ]
    CONFIG_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*", "transformer/config.json",
    ]
    # What validation/stage_r2.py mirrors for the base repo (it is also the
    # default text-to-image checkpoint).
    BASE_PATTERNS: ClassVar[list] = [*CHECKPOINT_PATTERNS, *SHARED_PATTERNS]

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # Upstream recipe (undistilled, true CFG).
        return GenerationDefaults(guidance_scale=3.5, num_steps=50)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _pipeline_class(self) -> Any:
        from diffusers import Kandinsky5T2IPipeline
        return _with_images_output(Kandinsky5T2IPipeline)

    def resolve_weights(self, weights_path: str) -> str:
        """``"auto"`` -> this backend's default checkpoint repo."""
        return self.DEFAULT_CHECKPOINT if weights_path == AUTO else weights_path

    def _repo_dir(self, repo: str, patterns: list, sentinel: str) -> str:
        from imference_engine.runtime.offline import local_repo_dir

        if os.path.isdir(repo):
            return repo
        return local_repo_dir(repo, patterns, self._cache_dir, namespace="image",
                              sentinel=sentinel, cdn_base=self._cdn_base)

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import (
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler,
            Kandinsky5Transformer3DModel,
        )
        from transformers import (
            AutoProcessor,
            CLIPTextModel,
            CLIPTokenizer,
            Qwen2_5_VLForConditionalGeneration,
        )

        from imference_engine.core.fp8 import apply_fp8_storage, fp8_transformer, resolve_storage

        checkpoint = self.resolve_weights(local_path)
        base_repo = base_model or DEFAULT_BASE
        vram = _vram_gb()
        build = recommend_build(vram)

        if checkpoint.lower().endswith(".safetensors") and not os.path.isdir(checkpoint):
            ckpt_dir = self._repo_dir(self.DEFAULT_CHECKPOINT, self.CONFIG_PATTERNS,
                                      "model_index.json")
            transformer = self._load_single_file(checkpoint, os.path.join(ckpt_dir, "transformer"))
        else:
            ckpt_dir = self._repo_dir(checkpoint, self.CHECKPOINT_PATTERNS, "model_index.json")
            logger.info(f"Loading Kandinsky 5 transformer from {ckpt_dir}")
            transformer = Kandinsky5Transformer3DModel.from_pretrained(
                ckpt_dir, subfolder="transformer", local_files_only=True,
                torch_dtype=torch.bfloat16)
        if resolve_storage("KANDINSKY5_TRANSFORMER", build.transformer, vram) == "fp8":
            fp8_transformer(transformer, "Kandinsky 5", logger)

        base_dir = self._repo_dir(base_repo, self.SHARED_PATTERNS, "vae/config.json")
        logger.info(f"Loading Kandinsky 5 shared components from {base_dir} (base={base_repo})")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(base_dir, "text_encoder"), local_files_only=True, dtype=torch.bfloat16)
        if resolve_storage("KANDINSKY5_ENCODER", build.encoder, vram) == "fp8":
            apply_fp8_storage(text_encoder)
            logger.info("Kandinsky 5 text encoder: fp8-resident storage enabled")

        return self._pipeline_class()(
            transformer=transformer,
            vae=AutoencoderKL.from_pretrained(
                os.path.join(base_dir, "vae"), local_files_only=True, torch_dtype=torch.bfloat16),
            text_encoder=text_encoder,
            tokenizer=AutoProcessor.from_pretrained(
                os.path.join(base_dir, "tokenizer"), local_files_only=True),
            text_encoder_2=CLIPTextModel.from_pretrained(
                os.path.join(base_dir, "text_encoder_2"), local_files_only=True,
                dtype=torch.bfloat16, use_safetensors=True),
            tokenizer_2=CLIPTokenizer.from_pretrained(
                os.path.join(base_dir, "tokenizer_2"), local_files_only=True),
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
                os.path.join(ckpt_dir, "scheduler"), local_files_only=True),
        )

    def _load_single_file(self, path: str, cfg_dir: str) -> Any:
        import accelerate
        import torch
        from diffusers import Kandinsky5Transformer3DModel
        from safetensors.torch import load_file

        logger.info(f"Loading Kandinsky 5 transformer from {path}")
        sd = {k: v.to(torch.bfloat16) for k, v in load_file(path).items()}
        with accelerate.init_empty_weights():
            transformer = Kandinsky5Transformer3DModel.from_config(
                Kandinsky5Transformer3DModel.load_config(cfg_dir, local_files_only=True))
        missing, unexpected = transformer.load_state_dict(sd, assign=True, strict=False)
        del sd
        if missing or unexpected:
            raise ValueError(
                f"{os.path.basename(path)} is not a Kandinsky 5 transformer in the diffusers "
                f"key layout ({len(missing)} missing, {len(unexpected)} unexpected keys; "
                f"first missing: {list(missing)[:3]})")
        return transformer

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        self._repo_dir(base_model or DEFAULT_BASE, self.SHARED_PATTERNS, "vae/config.json")
        logger.info("Kandinsky 5 base-components warmed (base=%s)", base_model or DEFAULT_BASE)

    # ------------------------------------------------------------------
    # Img2img — the text-to-image pipeline has none (see kandinsky5-edit)
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        raise NotImplementedError(
            "Kandinsky 5 text-to-image has no img2img pipeline. Use the "
            "'kandinsky5-edit' backend to edit an image from an instruction."
        )

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
        """Flow matching with the checkpoint's own config; the ``scheduler``
        name is ignored."""
        if scheduler:
            logger.debug("Kandinsky 5 ignores scheduler=%r (flow matching only)", scheduler)

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


class Kandinsky5EditBackend(Kandinsky5Backend):
    """Backend for Kandinsky 5.0 Image Lite I2I checkpoints (instruction editing)."""

    engine: ClassVar[str] = "kandinsky5-edit"

    DEFAULT_CHECKPOINT: ClassVar[str] = "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers"

    # Every request is an edit (needs a source image); tools key off this.
    instruction_edit: ClassVar[bool] = True

    def _pipeline_class(self) -> Any:
        from diffusers import Kandinsky5I2IPipeline
        return _with_images_output(Kandinsky5I2IPipeline)

    def make_img2img(self, t2i_pipe: Any) -> Any:
        # The resident pipe already is the image-conditioned edit pipeline.
        return t2i_pipe

    def build_inference_kwargs(
        self,
        *,
        width: int,  # noqa: ARG002 — the output follows the input image
        height: int,  # noqa: ARG002
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002
        chunk_size: int,
        generator: Any,
        image: Any = None,
        strength: float = 0.75,  # noqa: ARG002 — no strength: the image is a condition
    ) -> dict:
        if image is None:
            raise ValueError(
                "Kandinsky 5 I2I needs an input image (instruction editing has no "
                "text-to-image mode; use the 'kandinsky5' backend for that)")
        return {
            "image": image,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
        }
