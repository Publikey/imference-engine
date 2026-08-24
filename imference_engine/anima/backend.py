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
    ``pipe.load_components(dtype=torch.bfloat16)``; ``pipe.to(device)`` /
    ``.to("cpu")`` residency moves work; ``pipe(...).images`` returns images.
  - The modular ``__call__`` accepts num_inference_steps, height, width,
    generator, num_images_per_prompt, and negative_prompt (when set). It does
    NOT take ``guidance_scale`` — guidance is a separate ``ClassifierFreeGuidance``
    *guider* component, so passing it as a kwarg warns "Unexpected input ... will
    be ignored". The request's guidance_scale is instead applied to the guider by
    ``apply_guidance`` (``pipe.guider.guidance_scale = ...``) before the call; the
    scheduler is flow-matching (``FlowMatchEulerDiscreteScheduler``) and honors a
    ``shift`` via ``backend_options``; clip_skip is a genuine no-op (Qwen3 + T5
    text stack, no CLIP). If a future diffusers changes the modular signature,
    ``apply_guidance`` / ``apply_scheduler`` / ``build_inference_kwargs`` /
    ``encode_prompts`` are the single place to adjust.
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
as-is (no resolution). Resolving the tree is NOT sufficient on its own: the modular
index names each component by hub repo id, so ``_load_components`` repoints the
component specs at the local tree before loading — see its docstring.
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
        self._load_components(pipe, src)
        return pipe

    def _load_components(self, pipe: Any, local_dir: str) -> None:
        """Load the modular components off the mirrored tree, strictly.

        Two things ``pipe.load_components(dtype=...)`` won't do on its own:

        1. ``modular_model_index.json`` records each component's source as the HUB
           REPO ID, so a pipe built from a local dir still holds specs pointing at
           huggingface.co — under ``HF_HUB_OFFLINE=1`` every component load then
           fails. Repoint each spec at ``local_dir`` (only where the component's
           subfolder really exists there, so a component sourced from some other
           repo isn't silently mis-pointed).
        2. ``load_components`` catches per-component failures and only *warns*, so
           a failed load leaves the attribute None and the crash surfaces much
           later as ``'NoneType' object has no attribute 'dtype'`` inside a
           denoise block. Raise here instead, while the cause is still in hand.
        """
        import os

        import torch

        names, paths = [], {}
        for name in pipe.null_component_names:
            spec = pipe.get_component_spec(name)
            src = getattr(spec, "pretrained_model_name_or_path", None)
            if not src or spec.default_creation_method != "from_pretrained":
                continue  # nothing to load — load_components skips these too
            names.append(name)
            if os.path.isdir(src):
                continue  # already a local path
            subfolder = getattr(spec, "subfolder", "") or ""
            if os.path.isdir(os.path.join(local_dir, subfolder)):
                paths[name] = local_dir
            else:
                logger.warning(
                    "Anima component %r is sourced from %r, which is not mirrored "
                    "under %s; leaving its spec untouched (needs network)",
                    name, src, local_dir)

        pipe.load_components(names=names, pretrained_model_name_or_path=paths,
                             dtype=torch.bfloat16)

        missing = [n for n in names if getattr(pipe, n, None) is None]
        if missing:
            raise RuntimeError(
                f"Anima components failed to load from {local_dir}: "
                f"{', '.join(missing)}. See the warnings above for each component's "
                f"traceback — offline, this usually means the base repo tree is "
                f"incomplete (expected subfolders: {', '.join(missing)})."
            )

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
        # config for the layout. Community / ComfyUI Anima checkpoints wrap the
        # (already diffusers-format) DiT keys under a "model.diffusion_model." or
        # "diffusion_model." prefix; the diffusers Cosmos converter only strips
        # "net.", so a wrapped checkpoint matches nothing and every param stays on
        # the meta device -> "Cannot copy out of meta tensor". Strip the wrapper
        # prefix ourselves; the inner keys are already diffusers-format, so no
        # further rename is needed. Then feed the dict to from_single_file (a
        # supported input) — it no-ops the converter on an exact key match.
        sd = load_file(local_path)
        for _prefix in ("model.diffusion_model.", "diffusion_model."):
            if any(k.startswith(_prefix) for k in sd):
                sd = {k.removeprefix(_prefix): v for k, v in sd.items()}
                break
        transformer = CosmosTransformer3DModel.from_single_file(
            sd,
            config=os.path.join(base_dir, "transformer"),
            local_files_only=True,
            dtype=torch.bfloat16,
        )

        # Base modular pipeline; inject our DiT, then load the remaining components
        # (load_components skips the already-set transformer, so the base DiT is
        # never instantiated).
        pipe = ModularPipeline.from_pretrained(base_dir)
        pipe.update_components(transformer=transformer)
        self._load_components(pipe, base_dir)
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

    def apply_guidance(self, pipe: Any, guidance_scale: float) -> None:
        # Anima's modular pipeline holds a ``ClassifierFreeGuidance`` component
        # named ``guider`` (code default guidance_scale=4.0). Guidance is NOT a
        # ``pipe(...)`` kwarg — the scale is read off the guider object at denoise
        # time (``pred_uncond + scale*(pred_cond - pred_uncond)``), and the guider
        # also gates whether the unconditional/negative branch is encoded at all
        # (``num_conditions > 1``). So we set it on the guider before the call.
        # ``guidance_scale`` is a plain register_to_config attr (not a property), so
        # direct assignment is runtime-effective immediately. A value of ~1.0
        # disables CFG and skips the negative-prompt encode — a valid, faster path.
        # Verified against diffusers 0.39 ``modular_pipelines/anima`` blocks +
        # ``guiders/classifier_free_guidance.py``.
        guider = getattr(pipe, "guider", None)
        if guider is None:
            logger.warning(
                "Anima pipe exposes no guider; guidance_scale=%.2f not applied",
                guidance_scale)
            return
        try:
            guider.guidance_scale = float(guidance_scale)
            logger.info(
                "Anima guidance: ClassifierFreeGuidance guidance_scale=%.2f",
                guidance_scale)
        except Exception as e:  # extremely defensive — attr set shouldn't raise
            logger.warning("Anima: failed to set guider.guidance_scale (%s)", e)

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Anima is flow-matching (``FlowMatchEulerDiscreteScheduler``, block-defined).

        The ``scheduler`` NAME arg is ignored — DPM/Euler/Karras samplers don't map
        onto flow matching (same stance as FLUX / Z-Image). The one meaningful knob
        is ``shift``: pass it via ``backend_options`` to rebuild the flow-match
        scheduler with a fixed shift::

            engine.generate(..., backend_options={"shift": 3.0})

        No ``shift`` → leave the pipeline's configured scheduler untouched (the
        validated default path). Swap uses the documented modular API
        (``update_components``); verified the component is
        ``FlowMatchEulerDiscreteScheduler`` against diffusers 0.39
        ``modular_pipelines/anima``.
        """
        shift = kwargs.get("shift")
        if shift is None:
            return  # leave the pipe's default flow-match scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.update_components(
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(
                pipe.scheduler.config, shift=float(shift),
                use_dynamic_shifting=False,
            )
        )
        logger.info("Anima scheduler: FlowMatchEuler fixed shift=%s", shift)

    def build_inference_kwargs(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,  # noqa: ARG002 — applied via apply_guidance (guider), not here
        clip_skip: Optional[int],  # noqa: ARG002 — Anima has no CLIP (Qwen3+T5) → no-op
        chunk_size: int,
        generator: Any,
        image: Any = None,  # noqa: ARG002 — img2img unsupported (make_img2img raises)
        strength: float = 0.75,  # noqa: ARG002
    ) -> dict:
        # Guidance is NOT a `guidance_scale` __call__ kwarg for Anima (modular
        # pipeline warns "Unexpected input ... will be ignored"). It is set on the
        # ClassifierFreeGuidance *guider* component instead — see `apply_guidance`,
        # which the Engine calls per request before pipe(...). clip_skip is a no-op
        # (Anima uses a Qwen3 + T5 text stack, no CLIP layer to skip). t2i only —
        # `image` / `strength` are ignored (make_img2img raises before this).
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
