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

VALIDATED end-to-end (text-to-image) on diffusers 0.39 — RTX PRO 5000 Blackwell,
torch 2.12 — against ``circlestone-labs/Anima-Base-v1.0-Diffusers``:
  - Loaded via ``ModularPipeline.from_pretrained(repo)`` then
    ``pipe.load_components(torch_dtype=torch.bfloat16)``; ``pipe.to(device)`` /
    ``.to("cpu")`` residency moves work; ``pipe(...).images`` returns images.
  - The modular ``__call__`` accepts num_inference_steps, height, width,
    generator, num_images_per_prompt, and negative_prompt (when set). It does
    NOT take ``guidance_scale`` — guidance is a separate Guider block, so passing
    it warns "Unexpected input ... will be ignored" (confirmed on GPU); this
    backend therefore omits it and the request's guidance_scale is a no-op for
    Anima. If a future diffusers changes the modular signature,
    ``build_inference_kwargs`` / ``encode_prompts`` are the single place to adjust.
  - CPU-offload: ``enable_model_cpu_offload`` may not exist on a ModularPipeline;
    the ModelManager already falls back to ``.to(device)`` if it raises. img2img
    is unsupported (``make_img2img`` raises — no documented modular variant).

``weights_path`` may be EITHER a diffusers-format repo id / local directory (the
whole modular repo, e.g. ``circlestone-labs/Anima-Base-v1.0-Diffusers``) OR a
single ``.safetensors`` holding just the CosmosTransformer3D DiT (the community /
Civitai form). For the single-file form the DiT is loaded via
``CosmosTransformer3DModel.from_single_file`` and injected into a base modular
pipeline (``base_model`` or ``DEFAULT_BASE``) that supplies the Qwen3 encoder,
text conditioner, VAE and modular config — mirroring the FLUX/Chroma/Qwen
transformer-only load.

Offline / CDN: a repo-id ``weights_path`` is resolved into the flat offline tree
via ``local_repo_dir`` (the whole modular repo — DiT + Qwen3 encoder + text
conditioner + VAE + index) BEFORE ``from_pretrained``, so with ``IMAGE_MODEL_CDN``
set the model loads from the R2 mirror, never HuggingFace — same contract as the
other image backends. A ``weights_path`` that is already a local directory is used
as-is (no resolution).
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

    # Anima is a single self-contained modular repo (no transformer/base split):
    # every component — the CosmosTransformer3D DiT, the Qwen3 encoder, the
    # AnimaTextConditioner and the VAE — ships in the one repo, so we mirror the
    # WHOLE repo (``["*"]`` matches every path). SENTINEL is the modular index file
    # (not the classic ``model_index.json``) — the "already populated" marker for
    # the strict-offline fast path.
    BASE_PATTERNS: ClassVar[list] = ["*"]
    SENTINEL: ClassVar[str] = "modular_model_index.json"

    # Default base modular repo for the single-file DiT path (community / Civitai
    # Anima checkpoints ship the CosmosTransformer3D as one .safetensors; the
    # Qwen3 encoder, text conditioner, VAE and modular config come from here).
    # Overridable per model via ``base_model`` at register_model.
    DEFAULT_BASE: ClassVar[str] = "circlestone-labs/Anima-Base-v1.0-Diffusers"

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
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
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import os

        import torch
        from diffusers import ModularPipeline

        from imference_engine.runtime.offline import local_repo_dir

        # Single-file DiT checkpoint (community / Civitai Anima): the .safetensors
        # holds only the CosmosTransformer3D weights; the Qwen3 encoder, text
        # conditioner, VAE and modular config come from the base repo. Mirrors the
        # FLUX/Chroma/Qwen transformer-only load. A repo id / local dir keeps the
        # whole-repo path below.
        if local_path.endswith(".safetensors") and not os.path.isdir(local_path):
            return self._load_single_file(local_path, base_model)

        # local_path is a diffusers-format repo id or directory. A repo id is
        # resolved into the flat offline tree first (CDN when cdn_base is set, else
        # HuggingFace); an existing local dir is used verbatim.
        src = local_path
        if not os.path.isdir(local_path):
            src = local_repo_dir(
                local_path, self.BASE_PATTERNS, self._cache_dir,
                namespace="image", sentinel=self.SENTINEL, cdn_base=self._cdn_base)

        logger.info(f"Loading Anima modular pipeline from {src}")
        pipe = ModularPipeline.from_pretrained(src)
        pipe.load_components(torch_dtype=torch.bfloat16)
        return pipe

    def _load_single_file(self, local_path: str, base_model: Optional[str]) -> Any:
        """Load the DiT from a single .safetensors + the rest of the modular
        components from the base repo. The base repo (``base_model`` or
        ``DEFAULT_BASE``) is resolved into the flat offline tree (CDN when set)."""
        import os

        import torch
        from diffusers import CosmosTransformer3DModel, ModularPipeline
        from safetensors.torch import load_file

        from imference_engine.runtime.offline import local_repo_dir

        base_repo = base_model or self.DEFAULT_BASE
        base_dir = local_repo_dir(
            base_repo, self.BASE_PATTERNS, self._cache_dir,
            namespace="image", sentinel=self.SENTINEL, cdn_base=self._cdn_base)
        logger.info(
            "Loading Anima DiT from single file %s + base components from %s (base=%s)",
            local_path, base_dir, base_repo)

        # Build the DiT from the checkpoint, using the base repo's transformer
        # config for the layout. The diffusers Cosmos converter strips a "net."
        # prefix but NOT a ComfyUI-style "diffusion_model." one — normalize that
        # first, then feed the state_dict to from_single_file (a supported input)
        # so the official Cosmos key-rename still applies.
        sd = load_file(local_path)
        if any(k.startswith("diffusion_model.") for k in sd):
            sd = {k.removeprefix("diffusion_model."): v for k, v in sd.items()}
        transformer = CosmosTransformer3DModel.from_single_file(
            sd,
            config=os.path.join(base_dir, "transformer"),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )

        # Base modular pipeline; inject our DiT, then load the remaining components
        # (load_components skips the already-set transformer, so the base DiT is
        # never instantiated).
        pipe = ModularPipeline.from_pretrained(base_dir)
        pipe.update_components(transformer=transformer)
        pipe.load_components(torch_dtype=torch.bfloat16)
        return pipe

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        """Warm the base modular repo (Qwen3 encoder + VAE + text conditioner +
        config) that the single-file DiT path needs. Best-effort; no-op behaviour
        for the whole-repo path (which resolves its repo lazily at load)."""
        from imference_engine.runtime.offline import local_repo_dir
        local_repo_dir(base_model or self.DEFAULT_BASE, self.BASE_PATTERNS,
                       self._cache_dir, namespace="image", sentinel=self.SENTINEL,
                       cdn_base=self._cdn_base)

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
        guidance_scale: float,  # noqa: ARG002 — ignored by Anima's modular __call__
        clip_skip: Optional[int],  # noqa: ARG002 — Anima has no clip_skip
        chunk_size: int,
        generator: Any,
        image: Any = None,  # noqa: ARG002 — img2img unsupported (make_img2img raises)
        strength: float = 0.75,  # noqa: ARG002
    ) -> dict:
        # Anima's modular pipeline configures guidance via a separate Guider block,
        # NOT a `guidance_scale` __call__ kwarg — passing it triggers a diffusers
        # "Unexpected input 'guidance_scale' ... will be ignored" warning on every
        # render (confirmed on GPU). So it's deliberately NOT forwarded here; the
        # request's guidance_scale has no effect on Anima. t2i only — `image` /
        # `strength` are ignored (make_img2img raises before this is reached).
        return {
            "num_inference_steps": num_steps,
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
