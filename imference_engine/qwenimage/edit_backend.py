"""Qwen-Image-Edit backend — instruction-based image editing (2509 / 2511).

``QwenImageEditPlusPipeline`` (diffusers): the 20B Qwen-Image MMDiT conditioned
on the input image(s) through the Qwen2.5-VL encoder AND the VAE, steered by a
plain-language instruction ("make the sky stormy", "remove the man on the
left", "replace the text with 'OPEN'"). Unlike strength-based img2img there is
no ``strength``: the source image is a condition, the output is a new render.

Rides the engine's img2img path: ``Engine.generate(model=..., prompt=<the
instruction>, source_image=<image>)``. The resident pipe IS the edit pipe, so
``make_img2img`` returns it unchanged; a call without a source image is an
error (there is no text-to-image mode).

Weights, same two layouts as Qwen-Image:

- a diffusers repo id / dir (``Qwen/Qwen-Image-Edit-2511``) -> whole repo,
  resolved into the flat offline tree (CDN when ``IMAGE_MODEL_CDN`` is set);
- a transformer-only ``.safetensors`` (e.g. the Comfy-Org bf16 repackage) ->
  transformer weights from the file + shared components (Qwen2.5-VL encoder,
  processor, tokenizer, VAE, scheduler) from ``base_model`` (default
  ``Qwen/Qwen-Image-Edit-2511``).

- an unsloth ``.gguf`` (Q4_K_M: the Desktop AND cloud build) -> the
  low-memory recipe in ``edit_lite.py``: GGUF transformer + Lightning 4-step
  LoRA (adapter) + NF4 encoder loaded for the encode phase only. Catalog rows
  for it set ``num_steps=4`` / ``guidance_scale=1.0`` (the Lightning recipe).
  Env: ``QWEN_EDIT_LIGHTNING`` (4 | none), ``QWEN_EDIT_ENCODER`` (auto | nf4 | bf16).

VRAM: same 20B transformer as Qwen-Image — run with ``enable_offload=True``
(``offload_mode="group"``). bf16: ~5 GB of VRAM, PCIe-bound. GGUF Q4_K_M +
NF4 encoder: 6.5 GB peak, ~52 s per edit on an RTX 4070 Laptop (8 GB), with
~24 GB of host RAM while loading.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.qwenimage.backend import QwenImageBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE = "Qwen/Qwen-Image-Edit-2511"


class QwenImageEditBackend(QwenImageBackend):
    """Backend for Qwen-Image-Edit (Plus) checkpoints."""

    engine: ClassVar[str] = "qwenimage-edit"

    # The edit pipeline adds the Qwen2-VL image processor (vision tokens for the
    # condition image) to the Qwen-Image components. Transformer stays
    # CONFIG-ONLY: its weights come from the single-file checkpoint.
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*", "processor/*",
        "vae/*",
        "transformer/config.json",
    ]
    # Full repo (diffusers-format weights) when weights is a repo id / dir.
    FULL_PATTERNS: ClassVar[list] = ["*"]

    # Every request is an edit (needs a source image); tools key off this.
    instruction_edit: ClassVar[bool] = True

    def engine_defaults(self) -> GenerationDefaults:
        # Upstream 2511 recipe: 40 steps, true CFG 4.0. Lightning LoRA builds
        # override per catalog row (e.g. 8 steps / cfg 1.0).
        return GenerationDefaults(num_steps=40, guidance_scale=4.0)

    def weights_defaults(self, weights_path: str) -> Optional[GenerationDefaults]:
        """A GGUF is the lite build with the Lightning 4-step LoRA: its recipe
        (4 steps, CFG 1.0) unless a catalog row says otherwise."""
        from imference_engine.qwenimage.edit_lite import lightning_setting

        if weights_path.lower().endswith(".gguf") and lightning_setting() != "none":
            return GenerationDefaults(num_steps=4, guidance_scale=1.0)
        return None

    def load_pipeline(self, *, local_path: str, base_model: Optional[str] = None) -> Any:
        import torch
        from diffusers import QwenImageEditPlusPipeline

        if local_path.lower().endswith(".gguf") and not os.path.isdir(local_path):
            from imference_engine.qwenimage.edit_lite import load_lite_pipeline
            return load_lite_pipeline(local_path, base_model or DEFAULT_BASE, cache_dir=self._cache_dir,
                                      cdn_base=self._cdn_base, full_base_patterns=self.BASE_PATTERNS)
        if local_path.endswith(".safetensors") and not os.path.isdir(local_path):
            return self._load_single_file(local_path, base_model or DEFAULT_BASE)

        src = local_path
        if not os.path.isdir(local_path):
            from imference_engine.runtime.offline import local_repo_dir
            src = local_repo_dir(local_path, self.FULL_PATTERNS, self._cache_dir, namespace="image",
                                 cdn_base=self._cdn_base)
        logger.info(f"Loading Qwen-Image-Edit pipeline from {src}")
        return QwenImageEditPlusPipeline.from_pretrained(src, local_files_only=True, dtype=torch.bfloat16)

    def _load_single_file(self, local_path: str, base_repo: str) -> Any:
        import torch
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

        from imference_engine.runtime.offline import local_repo_dir

        from imference_engine.qwenimage.edit_lite import ensure_text_encoder_weights

        base_dir = local_repo_dir(base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
                                  cdn_base=self._cdn_base)
        # A lite (GGUF) pull of the same base skips the bf16 encoder shards.
        ensure_text_encoder_weights(base_repo, base_dir, cdn_base=self._cdn_base)
        logger.info(f"Loading Qwen-Image-Edit shared components from {base_dir} (base={base_repo})")
        transformer = QwenImageTransformer2DModel.from_single_file(
            local_path, config=os.path.join(base_dir, "transformer"),
            local_files_only=True, dtype=torch.bfloat16)
        return QwenImageEditPlusPipeline.from_pretrained(
            base_dir, transformer=transformer, local_files_only=True, dtype=torch.bfloat16)

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        from imference_engine.qwenimage.edit_lite import LITE_BASE_PATTERNS, encoder_mode
        from imference_engine.runtime.offline import local_repo_dir

        # With the NF4 encoder (the GGUF build) the 16.6 GB bf16 encoder shards
        # are not needed; a bf16 checkpoint fetches them on load.
        patterns = LITE_BASE_PATTERNS if encoder_mode() == "nf4" else self.BASE_PATTERNS
        local_repo_dir(base_model or DEFAULT_BASE, patterns, self._cache_dir,
                       namespace="image", cdn_base=self._cdn_base)
        logger.info("Qwen-Image-Edit base-components warmed (base=%s)", base_model or DEFAULT_BASE)

    def make_img2img(self, t2i_pipe: Any) -> Any:
        # The resident pipe already is the image-conditioned edit pipeline.
        return t2i_pipe

    def build_inference_kwargs(
        self,
        *,
        width: int,  # noqa: ARG002 — the output size follows the input image
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
            raise ValueError("Qwen-Image-Edit needs an input image (instruction editing has no text-to-image mode)")
        # A list = the main image followed by reference images (EditPlus takes
        # up to 3: "put the jacket from image 2 on the person in image 1").
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        return {
            "image": images,
            "num_inference_steps": num_steps,
            "true_cfg_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
        }
