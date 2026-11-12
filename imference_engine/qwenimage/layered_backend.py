"""Qwen-Image-Layered backend — decompose an image into RGBA layers.

``QwenImageLayeredPipeline`` (diffusers): the Qwen-Image MMDiT trained to split
an input image into ``layers`` transparent layers (background first), each a
full-size RGBA image. Stacked back in order they reproduce the input; moved,
recolored or removed individually they give an edit without touching the rest.
Apache-2.0.

Engine surface: ``Engine.generate(model=..., prompt="", source_image=<image>)``.

- ``source_image`` is required (there is no text-to-image mode).
- ``prompt`` describes the WHOLE image. Left empty, the pipeline captions the
  image itself with the Qwen2.5-VL encoder (a few seconds more).
- The result's ``media[i]`` is the recomposed image (all layers stacked) and
  ``MediaResult.layers[i]`` the list of RGBA layers, background first.
- ``backend_options``: ``layers`` (default 4), ``resolution`` (640 — the
  bucket the model was trained on — or 1024).

Weights: a ``.gguf`` (unsloth ``Qwen-Image-Layered-GGUF``) or ``"auto"`` for the
lite build (GGUF Q4_K_M + NF4 encoder, ``lite.py``), a transformer-only
``.safetensors``, or the diffusers repo. ``base_model`` defaults to
``Qwen/Qwen-Image-Layered``: the text encoder is the same file as every other
Qwen-Image repo, but the **VAE is different** (it encodes and decodes an alpha
channel), so the base cannot be shared with ``qwenimage``.

There is no Lightning LoRA for this checkpoint: it runs the upstream recipe
(50 steps, true CFG 4.0 — two transformer passes per step, over
``layers + 1`` frames of tokens). It is a tool, not an interactive generator.
Measured on an RTX 3090: 20 GB and ~8 s per step with the lite build; model
offload did not lower the peak, so treat it as a 24 GB tool for now.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.qwenimage.backend import QwenImageBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE = "Qwen/Qwen-Image-Layered"

GGUF_REPO = "unsloth/Qwen-Image-Layered-GGUF"
GGUF_FILE = "qwen-image-layered-Q4_K_M.gguf"


def compose_layers(layers: list) -> Any:
    """Stack RGBA layers bottom-up into one RGB image (white behind any
    remaining transparency)."""
    from PIL import Image

    base = Image.new("RGBA", layers[0].size, (255, 255, 255, 255))
    for layer in layers:
        base = Image.alpha_composite(base, layer.convert("RGBA"))
    return base.convert("RGB")


class QwenImageLayeredBackend(QwenImageBackend):
    """Backend for Qwen-Image-Layered checkpoints."""

    engine: ClassVar[str] = "qwenimage-layered"

    # The layered pipeline adds the Qwen2-VL processor (it captions the input
    # image when no prompt is given). Transformer CONFIG-ONLY.
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*", "processor/*",
        "vae/*",
        "transformer/config.json",
    ]
    LITE_BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "processor/*",
        "vae/*",
        "transformer/config.json",
        "text_encoder/config.json", "text_encoder/generation_config.json",
    ]
    FULL_PATTERNS: ClassVar[list] = ["*"]

    # Every request needs a source image; tools key off this.
    instruction_edit: ClassVar[bool] = True

    DEFAULT_LAYERS: ClassVar[int] = 4
    DEFAULT_RESOLUTION: ClassVar[int] = 640

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        super().__init__(cache_dir=cache_dir, cdn_base=cdn_base)
        # Per-request options (see Lumina2Backend): set in apply_scheduler, read
        # in build_inference_kwargs right after.
        self._call_options: dict = {}

    def engine_defaults(self) -> GenerationDefaults:
        # Upstream recipe; no distilled variant exists.
        return GenerationDefaults(num_steps=50, guidance_scale=4.0)

    def weights_defaults(self, weights_path: str) -> Optional[GenerationDefaults]:
        return None  # no Lightning for the layered checkpoint

    def resolve_weights(self, weights_path: str) -> str:
        from imference_engine.qwenimage.builds import AUTO

        if weights_path != AUTO:
            return weights_path
        from imference_engine.runtime.offline import fetch

        logger.info("Qwen-Image-Layered auto build: GGUF Q4_K_M + NF4 encoder")
        return fetch(GGUF_REPO, GGUF_FILE, self._cdn_base, self._cache_dir, namespace="image")

    def load_pipeline(self, *, local_path: str, base_model: Optional[str] = None) -> Any:
        import torch
        from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel

        from imference_engine.runtime.offline import local_repo_dir

        base_repo = base_model or DEFAULT_BASE
        local_path = self.resolve_weights(local_path)
        lower = local_path.lower()
        is_file = not os.path.isdir(local_path)

        if is_file and lower.endswith(".gguf"):
            from imference_engine.qwenimage.lite import load_lite_pipeline
            return load_lite_pipeline(
                local_path, base_repo, pipeline_cls=QwenImageLayeredPipeline,
                cache_dir=self._cache_dir, cdn_base=self._cdn_base,
                full_base_patterns=self.BASE_PATTERNS,
                lite_base_patterns=self.LITE_BASE_PATTERNS, lightning=False)
        if is_file and lower.endswith(".safetensors"):
            base_dir = base_repo if os.path.isdir(base_repo) else local_repo_dir(
                base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
                cdn_base=self._cdn_base)
            transformer = QwenImageTransformer2DModel.from_single_file(
                local_path, config=os.path.join(base_dir, "transformer"),
                local_files_only=True, dtype=torch.bfloat16)
            return QwenImageLayeredPipeline.from_pretrained(
                base_dir, transformer=transformer, local_files_only=True, dtype=torch.bfloat16)

        src = local_path
        if is_file:
            src = local_repo_dir(local_path, self.FULL_PATTERNS, self._cache_dir,
                                 namespace="image", cdn_base=self._cdn_base)
        logger.info(f"Loading Qwen-Image-Layered pipeline from {src}")
        return QwenImageLayeredPipeline.from_pretrained(
            src, local_files_only=True, dtype=torch.bfloat16)

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        from imference_engine.qwenimage.edit_lite import encoder_mode
        from imference_engine.runtime.offline import local_repo_dir

        patterns = self.LITE_BASE_PATTERNS if encoder_mode() == "nf4" else self.BASE_PATTERNS
        local_repo_dir(base_model or DEFAULT_BASE, patterns, self._cache_dir,
                       namespace="image", cdn_base=self._cdn_base)
        logger.info("Qwen-Image-Layered base-components warmed (base=%s)",
                    base_model or DEFAULT_BASE)

    def make_img2img(self, t2i_pipe: Any) -> Any:
        # The resident pipe already is the image-conditioned pipeline.
        return t2i_pipe

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        out: dict = {"negative_prompt": negative_prompt or " "}
        if prompt and prompt.strip():
            out["prompt"] = prompt
        # No prompt: the pipeline captions the image with the VL encoder.
        return out

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any
    ) -> None:
        self._call_options = {}
        if kwargs.get("layers") is not None:
            self._call_options["layers"] = int(kwargs["layers"])
        if kwargs.get("resolution") is not None:
            self._call_options["resolution"] = int(kwargs["resolution"])
        super().apply_scheduler(pipe, scheduler, **kwargs)

    def build_inference_kwargs(
        self,
        *,
        width: int,  # noqa: ARG002 — the output follows the input's aspect ratio
        height: int,  # noqa: ARG002   at the `resolution` bucket
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002
        chunk_size: int,
        generator: Any,
        image: Any = None,
        strength: float = 0.75,  # noqa: ARG002 — no strength: the image is decomposed
    ) -> dict:
        if image is None:
            raise ValueError(
                "Qwen-Image-Layered needs an input image (it decomposes an image "
                "into layers; there is no text-to-image mode)")
        return {
            "image": image.convert("RGBA") if hasattr(image, "convert") else image,
            "num_inference_steps": num_steps,
            "true_cfg_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
            "layers": self._call_options.get("layers", self.DEFAULT_LAYERS),
            "resolution": self._call_options.get("resolution", self.DEFAULT_RESOLUTION),
            "cfg_normalize": True,
            "use_en_prompt": True,
        }

    def split_output(self, produced: list) -> tuple:
        """The pipeline returns one LIST of RGBA layers per image: hand the
        engine one recomposed image per entry, and the layers on the side."""
        groups = [list(group) for group in produced]
        return [compose_layers(g) for g in groups], groups
