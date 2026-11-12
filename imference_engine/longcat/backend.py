"""LongCat-Image backends — Meituan's 6B flow-matching DiT, text-to-image and editing.

Own sub-package parallel to ``imference_engine.lumina2`` / ``.ernieimage``.
LongCat-Image is a FLUX-style double/single-stream transformer (6B) conditioned
on **Qwen2.5-VL-7B** with the FLUX VAE. Apache-2.0, ungated. Both pipelines are
in diffusers 0.40. Two backends, one per pipeline:

- ``longcat`` — ``LongCatImagePipeline``, text-to-image. Undistilled: 50 steps,
  true CFG (guidance 4.0), negative prompt honored.
- ``longcat-edit`` — ``LongCatImageEditPipeline``, instruction editing of
  ``source_image`` (the image also goes through the VL encoder). The default
  checkpoint is **LongCat-Image-Edit-Turbo** (8 steps, guidance 1.0); the
  undistilled Edit checkpoint runs 50 steps at 4.5.

Weights (``weights_path``): ``"auto"`` (the backend's default checkpoint), a
LongCat repo id / directory (diffusers layout), or a transformer ``.safetensors``
in the diffusers key layout (finetunes). diffusers 0.40 has no single-file or
GGUF loader for the LongCat transformer, so the community GGUF files are not
usable here: the low-memory builds store the bf16 weights in fp8 instead
(layerwise casting), for the transformer and for the encoder.

Every LongCat repo ships the same 16.6 GB encoder. To mirror it once, a
checkpoint contributes only ``model_index.json``, ``scheduler/`` and
``transformer/``; the encoder, VAE, tokenizer and processor come from
``base_model`` (default ``meituan-longcat/LongCat-Image``).

``backend_options``: ``prompt_rewrite`` (text-to-image only, default false) —
the pipeline can have its own encoder rewrite the prompt before encoding
(upstream default). It costs an LLM generation per request and changes what was
asked, so it is opt-in here.

Env: ``LONGCAT_TRANSFORMER`` / ``LONGCAT_ENCODER`` = auto | bf16 | fp8 (auto
follows the VRAM table of ``builds.py``).
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.longcat.builds import AUTO, recommend_build
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE = "meituan-longcat/LongCat-Image"


def _storage_setting(var: str) -> str:
    v = os.environ.get(var, "auto").strip().lower()
    return v if v in ("bf16", "fp8") else "auto"


def _vram_gb() -> Optional[float]:
    from imference_engine.managers.builds import cuda_vram_gb
    return cuda_vram_gb()


class LongCatBackend(PipelineBackend):
    """Backend for LongCat-Image text-to-image checkpoints."""

    engine: ClassVar[str] = "longcat"

    DEFAULT_CHECKPOINT: ClassVar[str] = "meituan-longcat/LongCat-Image"

    # Shared components (base repo): the 16.6 GB encoder, VAE, tokenizer, processor.
    SHARED_PATTERNS: ClassVar[list] = [
        "text_encoder/*", "tokenizer/*", "text_processor/*", "vae/*",
    ]
    # Per-checkpoint files: pipeline index, scheduler recipe, transformer.
    CHECKPOINT_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*", "transformer/*",
    ]
    # A transformer single file needs the layout only.
    CONFIG_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*", "transformer/config.json",
    ]
    # What validation/stage_r2.py mirrors for the base repo (it is also the
    # default text-to-image checkpoint): everything above.
    BASE_PATTERNS: ClassVar[list] = [*CHECKPOINT_PATTERNS, *SHARED_PATTERNS]

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base
        # Per-request pipe(...) extras from backend_options: set in
        # apply_scheduler, read in build_inference_kwargs right after.
        self._call_options: dict = {}

    def engine_defaults(self) -> GenerationDefaults:
        # Upstream recipe for the undistilled text-to-image checkpoint.
        return GenerationDefaults(guidance_scale=4.0, num_steps=50)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _pipeline_class(self) -> Any:
        from diffusers import LongCatImagePipeline
        return LongCatImagePipeline

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
            LongCatImageTransformer2DModel,
        )
        from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        checkpoint = self.resolve_weights(local_path)
        base_repo = base_model or DEFAULT_BASE
        build = recommend_build(_vram_gb())

        if checkpoint.lower().endswith(".safetensors") and not os.path.isdir(checkpoint):
            # A finetune's transformer in the diffusers key layout; layout and
            # scheduler from the backend's default checkpoint repo.
            ckpt_dir = self._repo_dir(self.DEFAULT_CHECKPOINT, self.CONFIG_PATTERNS,
                                      "model_index.json")
            transformer = self._load_single_file(checkpoint, os.path.join(ckpt_dir, "transformer"))
        else:
            ckpt_dir = self._repo_dir(checkpoint, self.CHECKPOINT_PATTERNS, "model_index.json")
            logger.info(f"Loading LongCat transformer from {ckpt_dir}")
            transformer = LongCatImageTransformer2DModel.from_pretrained(
                ckpt_dir, subfolder="transformer", local_files_only=True,
                torch_dtype=torch.bfloat16)

        if self._resolve_storage("LONGCAT_TRANSFORMER", build.transformer) == "fp8":
            try:
                transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
                logger.info("LongCat transformer: fp8-resident storage enabled")
            except Exception as e:  # noqa: BLE001 — bf16 beats a hard fail
                logger.warning(f"LongCat fp8-resident storage unavailable ({e}); staying bf16")

        base_dir = self._repo_dir(base_repo, self.SHARED_PATTERNS, "vae/config.json")
        logger.info(f"Loading LongCat shared components from {base_dir} (base={base_repo})")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(base_dir, "text_encoder"), local_files_only=True, dtype=torch.bfloat16)
        if self._resolve_storage("LONGCAT_ENCODER", build.encoder) == "fp8":
            from imference_engine.core.fp8 import apply_fp8_storage
            apply_fp8_storage(text_encoder)
            logger.info("LongCat text encoder: fp8-resident storage enabled")

        return self._pipeline_class()(
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
                os.path.join(ckpt_dir, "scheduler"), local_files_only=True),
            vae=AutoencoderKL.from_pretrained(
                os.path.join(base_dir, "vae"), local_files_only=True, torch_dtype=torch.bfloat16),
            text_encoder=text_encoder,
            tokenizer=AutoTokenizer.from_pretrained(
                os.path.join(base_dir, "tokenizer"), local_files_only=True),
            text_processor=AutoProcessor.from_pretrained(
                os.path.join(base_dir, "text_processor"), local_files_only=True),
            transformer=transformer,
        )

    def _load_single_file(self, path: str, cfg_dir: str) -> Any:
        import accelerate
        import torch
        from diffusers import LongCatImageTransformer2DModel
        from safetensors.torch import load_file

        logger.info(f"Loading LongCat transformer from {path}")
        sd = {k: v.to(torch.bfloat16) for k, v in load_file(path).items()}
        with accelerate.init_empty_weights():
            transformer = LongCatImageTransformer2DModel.from_config(
                LongCatImageTransformer2DModel.load_config(cfg_dir, local_files_only=True))
        missing, unexpected = transformer.load_state_dict(sd, assign=True, strict=False)
        del sd
        if missing or unexpected:
            raise ValueError(
                f"{os.path.basename(path)} is not a LongCat transformer in the diffusers "
                f"key layout ({len(missing)} missing, {len(unexpected)} unexpected keys; "
                f"first missing: {list(missing)[:3]})")
        return transformer

    @staticmethod
    def _resolve_storage(env_var: str, recommended: str) -> str:
        """bf16 | fp8 from the env override, else the build table. fp8 storage
        needs CUDA (float8 kernels): anything else stays bf16."""
        if _vram_gb() is None:
            return "bf16"
        setting = _storage_setting(env_var)
        return setting if setting != "auto" else (recommended or "bf16")

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        self._repo_dir(base_model or DEFAULT_BASE, self.SHARED_PATTERNS, "vae/config.json")
        logger.info("LongCat base-components warmed (base=%s)", base_model or DEFAULT_BASE)

    # ------------------------------------------------------------------
    # Img2img — the text-to-image pipeline has none (see longcat-edit)
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        raise NotImplementedError(
            "LongCat-Image text-to-image has no img2img pipeline. Use the "
            "'longcat-edit' backend to edit an image from an instruction."
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
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any
    ) -> None:
        """Flow matching with the checkpoint's own dynamic-shift config; the
        ``scheduler`` name is ignored. Records the per-request options."""
        self._call_options = {}
        if kwargs.get("prompt_rewrite") is not None:
            self._call_options["enable_prompt_rewrite"] = bool(kwargs["prompt_rewrite"])

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
            "enable_cfg_renorm": True,
            # Upstream defaults to True: opt-in here (backend_options.prompt_rewrite).
            "enable_prompt_rewrite": self._call_options.get("enable_prompt_rewrite", False),
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


class LongCatEditBackend(LongCatBackend):
    """Backend for LongCat-Image-Edit checkpoints (instruction editing)."""

    engine: ClassVar[str] = "longcat-edit"

    DEFAULT_CHECKPOINT: ClassVar[str] = "meituan-longcat/LongCat-Image-Edit-Turbo"

    # Every request is an edit (needs a source image); tools key off this.
    instruction_edit: ClassVar[bool] = True

    def engine_defaults(self) -> GenerationDefaults:
        # Edit-Turbo, the default checkpoint (distilled).
        return GenerationDefaults(guidance_scale=1.0, num_steps=8)

    def weights_defaults(self, weights_path: str) -> Optional[GenerationDefaults]:
        """The undistilled Edit checkpoint wants 50 steps of true CFG; Turbo
        (and ``"auto"``, which is Turbo) keep the engine defaults."""
        if weights_path == AUTO or "turbo" in weights_path.lower():
            return None
        return GenerationDefaults(guidance_scale=4.5, num_steps=50)

    def _pipeline_class(self) -> Any:
        from diffusers import LongCatImageEditPipeline
        return LongCatImageEditPipeline

    def make_img2img(self, t2i_pipe: Any) -> Any:
        # The resident pipe already is the image-conditioned edit pipeline.
        return t2i_pipe

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any  # noqa: ARG002
    ) -> None:
        self._call_options = {}

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
                "LongCat-Image-Edit needs an input image (instruction editing has "
                "no text-to-image mode; use the 'longcat' backend for that)")
        return {
            "image": image,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
        }
