"""Anima backend — CircleStone Labs / Comfy Org Anima, a MODULAR-pipeline model.

Anima is unlike the other image backends: diffusers ships it ONLY as a Modular
Diffusers pipeline (``AnimaModularPipeline`` / loaded generically via
``ModularPipeline.from_pretrained``). There is NO standard ``AnimaPipeline``, no
``from_single_file`` and no documented img2img variant, so this backend adapts
the modular API onto ``PipelineBackend`` rather than mirroring the FLUX/Chroma
transformer-only load pattern.

Architecture (for context): a ``CosmosTransformer3DModel`` DiT + a Qwen3 text
encoder + an ``AnimaTextConditioner`` (learned T5 tokens cross-attending Qwen3
hidden states) + the ``AutoencoderKLQwenImage`` VAE.

VERIFIED against the diffusers docs/source (main + v0.39.0):
  - Loaded via ``ModularPipeline.from_pretrained(repo)`` then
    ``pipe.load_components(torch_dtype=torch.bfloat16)``.
  - ``pipe.to("cuda")`` IS supported → the ModelManager's ``.to(device)`` /
    ``.to("cpu")`` residency moves work.
  - ``pipe(prompt=...).images[0]`` — returns ``.images``.

UNVERIFIED — the GPU validation gate (the doc example only passes ``prompt``):
  - Which extra kwargs the modular ``__call__`` accepts (num_inference_steps,
    guidance_scale, height, width, generator, negative_prompt,
    num_images_per_prompt). This backend passes the standard set; if the modular
    pipeline rejects one, trim it in ``build_inference_kwargs`` / ``encode_prompts``
    — that is the single place to adjust.
  - CPU-offload: ``enable_model_cpu_offload`` may not exist on a ModularPipeline;
    the ModelManager already falls back to ``.to(device)`` if it raises.

NOTE: ``weights_path`` for Anima is a diffusers-format repo id or local directory
(e.g. ``circlestone-labs/Anima-Base-v1.0-Diffusers``), NOT a single .safetensors.
``base_model`` is unused (no transformer/base split).
"""
from __future__ import annotations
import logging
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)


class AnimaBackend(PipelineBackend):
    """Backend for Anima (Modular Diffusers pipeline)."""

    engine: ClassVar[str] = "anima"

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        # cache_dir/cdn_base are accepted for signature parity with the other
        # backends but the modular loader manages its own component fetch; the
        # offline flat-tree wiring is not plumbed through ModularPipeline here.
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # No opinion: Anima's recommended steps/guidance aren't documented, so let
        # GLOBAL_DEFAULTS (28 / 6.0 / 1024) apply and let a catalog row refine
        # once real numbers are known. bfloat16 is set at load, not here.
        return GenerationDefaults()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None  # noqa: ARG002
    ) -> Any:
        import torch
        from diffusers import ModularPipeline

        # local_path is a diffusers-format repo id or directory (NOT a .safetensors).
        logger.info(f"Loading Anima modular pipeline from {local_path}")
        pipe = ModularPipeline.from_pretrained(local_path)
        pipe.load_components(torch_dtype=torch.bfloat16)
        return pipe

    # ------------------------------------------------------------------
    # Img2img (not supported)
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        raise NotImplementedError(
            "Anima has no documented img2img pipeline (Modular Diffusers, "
            "text-to-image only). Call generate() without source_image."
        )

    def get_compute_module(self, pipe: Any) -> Any:
        # Best-effort: the modular pipeline may expose the DiT as .transformer or
        # under .components. Not used by the ModelManager's device moves, so a
        # None here is harmless.
        return getattr(pipe, "transformer", None)

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        # Pass negative only when provided (Cosmos-based CFG likely supports it,
        # but the doc example omits it — if the modular __call__ rejects
        # negative_prompt, drop this branch). Qwen3 has no 77-token limit.
        kwargs: dict = {"prompt": prompt}
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        return kwargs

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any  # noqa: ARG002
    ) -> None:
        # Scheduler is block-defined in the modular pipeline; no standard
        # from_config swap. Leave the pipeline's configured scheduler.
        return None

    def build_inference_kwargs(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002 — Anima has no clip_skip
        chunk_size: int,
        generator: Any,
        image: Any = None,  # noqa: ARG002 — img2img unsupported (make_img2img raises)
        strength: float = 0.75,  # noqa: ARG002
    ) -> dict:
        # Standard diffusion kwargs. If the modular __call__ rejects any of these
        # on the first GPU run, remove it here (see the module docstring). t2i
        # only — `image`/`strength` are ignored (make_img2img raises before here).
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
