"""Qwen-Image backend — Alibaba Qwen-Image, a 20B MMDiT (flow matching).

Own sub-package parallel to ``imference_engine.flux`` / ``.chroma``. Qwen-Image is
a 20B multimodal DiT whose standout is native, reliable text rendering
(multi-line, paragraph, CJK). It rides the generic Engine / ModelManager /
RuntimeConfig machinery on the same diffusers 0.40 stack.

Two backend-relevant specifics:

- **Qwen2.5-VL text encoder** (tens of GB) — like Z-Image's Qwen encoder, it is
  a shared base component, so prefer CDN-on-demand for the base_model path.
  Single encoder (no CLIP), so no ``text_encoder_2`` / ``tokenizer_2``.
- **True CFG.** Qwen-Image's canonical call uses ``true_cfg_scale`` (real
  classifier-free guidance) with a negative prompt — NOT the distilled
  ``guidance_scale``. This backend maps the engine's ``guidance_scale`` onto
  ``true_cfg_scale`` and passes the negative prompt through (default " ",
  matching upstream).

VRAM: the 20B transformer is very heavy (41 GB bf16 + a 16.6 GB encoder). bf16
single files run with ``RuntimeConfig(enable_offload=True)``; the realistic
route below 40 GB is the **lite build** (``lite.py``): an unsloth ``.gguf``
transformer + lightx2v's Lightning LoRA (4 steps, CFG 1.0) + the NF4 encoder
loaded for the encode phase only. ``weights_path="auto"`` fetches the
Qwen-Image-2512 lite build (``builds.py``); a ``.gguf`` path selects it too.
Lightning rows carry ``num_steps=4`` / ``guidance_scale=1.0`` (``weights_defaults``).
Env: ``QWEN_IMAGE_LIGHTNING`` (4 | 8 | none) plus the encoder knobs shared with
the Edit backend (``QWEN_EDIT_ENCODER``, ``QWEN_EDIT_KEEP_ENCODER``).

Checkpoints: Qwen-Image and **Qwen-Image-2512** share everything but the
transformer weights (same encoder, VAE and configs — identical sha256), so
``base_model="Qwen/Qwen-Image"`` serves both.

Qwen-Image-Edit (instruction-based editing) is a DIFFERENT pipeline with an
image-condition call signature: see ``edit_backend.py`` (engine key
``qwenimage-edit``).
"""
from __future__ import annotations
import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE = "Qwen/Qwen-Image"


class QwenImageBackend(PipelineBackend):
    """Backend for Qwen-Image checkpoints (.safetensors single-file)."""

    engine: ClassVar[str] = "qwenimage"

    # Files needed from a Qwen-Image base repo (Qwen/Qwen-Image, diffusers format).
    # The Qwen2.5-VL text encoder + its tokenizer + the VAE + scheduler are FULL
    # weights/config. The transformer is CONFIG-ONLY: from_single_file reads its
    # layout here while the WEIGHTS come from the checkpoint. Single encoder — no
    # text_encoder_2/tokenizer_2.
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*",
        "vae/*",
        "transformer/config.json",
    ]

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # Qwen-Image family norm: real CFG ~4.0 (mapped to true_cfg_scale). Base
        # recommends ~40-50 steps; leave num_steps to GLOBAL_DEFAULTS (28) and let
        # a catalog row bump it (or a Lightning-distilled checkpoint lower it).
        return GenerationDefaults(guidance_scale=4.0)

    def weights_defaults(self, weights_path: str) -> Optional[GenerationDefaults]:
        """A GGUF (or ``"auto"``) is the lite build with the Lightning LoRA: its
        recipe (4 or 8 steps, CFG 1.0) unless a catalog row says otherwise."""
        from imference_engine.qwenimage.builds import AUTO
        from imference_engine.qwenimage.lite import lightning_steps

        steps = lightning_steps()
        if steps is not None and (weights_path == AUTO or weights_path.lower().endswith(".gguf")):
            return GenerationDefaults(num_steps=steps, guidance_scale=1.0)
        return None

    def resolve_weights(self, weights_path: str) -> str:
        """``"auto"`` -> the local path of the lite transformer (downloaded on
        first use); anything else is returned as is."""
        from imference_engine.qwenimage.builds import AUTO, recommend_build

        if weights_path != AUTO:
            return weights_path
        from imference_engine.managers.builds import cuda_vram_gb
        from imference_engine.runtime.offline import fetch

        build = recommend_build(cuda_vram_gb())
        logger.info("Qwen-Image auto build: %s (%s)", build.name, build.reason)
        if build.offload:
            logger.info(
                "Qwen-Image build %r expects RuntimeConfig(enable_offload=True, "
                "offload_mode=%r) on this card", build.name, build.offload)
        return fetch(build.repo, build.filename, self._cdn_base, self._cache_dir, namespace="image")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import QwenImagePipeline

        local_path = self.resolve_weights(local_path)
        if local_path.lower().endswith(".gguf") and not os.path.isdir(local_path):
            from imference_engine.qwenimage.lite import load_lite_pipeline
            return load_lite_pipeline(
                local_path, base_model or DEFAULT_BASE, pipeline_cls=QwenImagePipeline,
                cache_dir=self._cache_dir, cdn_base=self._cdn_base,
                full_base_patterns=self.BASE_PATTERNS)
        if base_model:
            return self._load_with_base_model(local_path, base_model)

        logger.info(f"Loading Qwen-Image pipeline from {local_path}")
        return QwenImagePipeline.from_single_file(
            local_path,
            dtype=torch.bfloat16,
        )

    def _load_with_base_model(self, local_path: str, base_repo: str) -> Any:
        """Transformer WEIGHTS from local_path + shared components (Qwen2.5-VL,
        VAE, tokenizer, scheduler) from the base repo, resolved into the flat
        offline tree (no HF hit once populated)."""
        import torch
        from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

        from imference_engine.runtime.offline import local_repo_dir

        base_dir = local_repo_dir(
            base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
            cdn_base=self._cdn_base)
        logger.info(f"Loading Qwen-Image shared components from {base_dir} (base={base_repo})")

        transformer = QwenImageTransformer2DModel.from_single_file(
            local_path,
            config=os.path.join(base_dir, "transformer"),
            local_files_only=True,
            dtype=torch.bfloat16,
        )
        return QwenImagePipeline.from_pretrained(
            base_dir,
            transformer=transformer,
            local_files_only=True,
            dtype=torch.bfloat16,
        )

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        if not base_model:
            return
        from imference_engine.runtime.offline import local_repo_dir
        local_repo_dir(base_model, self.BASE_PATTERNS, self._cache_dir,
                       namespace="image", cdn_base=self._cdn_base)
        logger.info("Qwen-Image base-components warmed (base=%s)", base_model)

    # ------------------------------------------------------------------
    # Img2img
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        from diffusers import QwenImageImg2ImgPipeline

        cls: Any = QwenImageImg2ImgPipeline
        loader = getattr(t2i_pipe, "_encoder_loader", None)
        if loader is not None:
            # Lite build: the encoder may not be resident. The wrapper loads it
            # for its own encode phase and always lets go of it afterwards; the
            # resident pipe owns the decision to keep an encoder loaded.
            from imference_engine.qwenimage.lite import lite_pipeline_class
            cls = lite_pipeline_class(QwenImageImg2ImgPipeline)
        pipe = cls(
            scheduler=t2i_pipe.scheduler,
            vae=t2i_pipe.vae,
            text_encoder=t2i_pipe.text_encoder,
            tokenizer=t2i_pipe.tokenizer,
            transformer=t2i_pipe.transformer,
        )
        if loader is not None:
            pipe._encoder_loader = loader
            pipe._always_release = True
        return pipe

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        # True CFG → negative prompt is honored. Upstream uses a single space as
        # the default negative (not ""), so mirror that. Qwen2.5-VL has no
        # 77-token limit → raw strings.
        return {"prompt": prompt, "negative_prompt": negative_prompt or " "}

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any
    ) -> None:
        """Qwen-Image uses FlowMatchEulerDiscreteScheduler (set at load). The
        `scheduler` name is ignored; an explicit `shift` via backend_options
        rebuilds it with a fixed shift (advanced)."""
        shift = kwargs.get("shift")
        if shift is None:
            return
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, shift=float(shift), use_dynamic_shifting=False
        )
        logger.info(f"Qwen-Image scheduler: FlowMatchEuler fixed shift={shift}")

    def build_inference_kwargs(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002 — Qwen-Image has no clip_skip
        chunk_size: int,
        generator: Any,
        image: Any = None,
        strength: float = 0.75,
    ) -> dict:
        # Map the engine's guidance_scale onto true_cfg_scale (real CFG). The
        # distilled `guidance_scale` pipeline arg is left at its default.
        kwargs = {
            "num_inference_steps": num_steps,
            "true_cfg_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
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
