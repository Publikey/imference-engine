"""FLUX backend — Black Forest Labs FLUX.1 pipelines (flow-matching DiT).

Its own sub-package parallel to ``imference_engine.zimage`` / ``.wan``: FLUX is a
12B rectified-flow transformer with the largest community-finetune ecosystem
after SDXL, so it earns a self-contained unit. It still rides the generic
``Engine`` / ``ModelManager`` / ``RuntimeConfig`` machinery and the same diffusers
0.39 stack as SDXL / Z-Image — the split is a packaging boundary, not an engine
fork.

FLUX checkpoints come in two flavors, mirroring Z-Image:
- *Full* base: a self-contained safetensors with all components → from_single_file.
- *Community finetunes* (the common case, e.g. Civitai): a transformer-only
  safetensors that needs the two text encoders (CLIP-L + T5-XXL), their
  tokenizers, and the VAE from a base repo (``black-forest-labs/FLUX.1-dev``).

VRAM: the 12B transformer is heavy (~24 GB in bf16). On consumer GPUs run with
``RuntimeConfig(enable_offload=True)`` — the generic ModelManager then lets
accelerate shuttle submodels, dropping peak VRAM to roughly the transformer
alone. No FLUX-specific code is needed for that; it is the shared offload path.

Guidance: FLUX.1-dev is GUIDANCE-DISTILLED — ``guidance_scale`` (default 3.5) is
an embedded conditioning signal, NOT classifier-free guidance, so there is no
true negative prompt without doubling compute (true-CFG). FLUX.1-schnell is
timestep-distilled (~4 steps) and ignores guidance entirely. dev vs schnell is a
per-CHECKPOINT distinction → expressed as model catalog defaults (num_steps), not
in the engine layer here.

Offline: the base components resolve into the flat, symlink-free tree
(``IMAGE_MODEL_CACHE``, ``namespace="image"``) and load with
``local_files_only=True``, so a cold load never contacts HF once the tree is
populated (prefetch / CDN mirror) under ``HF_HUB_OFFLINE=1``. NOTE:
``black-forest-labs/FLUX.1-dev`` is a GATED HF repo — mirror its base components
to ``IMAGE_MODEL_CDN`` (or pre-populate the tree) for offline / worker use.
"""
from __future__ import annotations
import logging
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)


class FluxBackend(PipelineBackend):
    """Backend for FLUX.1 checkpoints (.safetensors single-file)."""

    engine: ClassVar[str] = "flux"

    # Files needed from a FLUX base repo for the base_model path. The two text
    # encoders (CLIP-L + T5-XXL), their tokenizers, the VAE and the scheduler are
    # FULL weights/config. The transformer is CONFIG-ONLY: from_single_file reads
    # its layout from here (config=base_dir/transformer) while the transformer
    # WEIGHTS come from the checkpoint. Mirroring transformer/* would drag the ~24
    # GB base transformer weights the checkpoint replaces.
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "tokenizer_2/*",
        "text_encoder/*", "text_encoder_2/*",
        "vae/*",
        "transformer/config.json",
    ]

    # FLUX.1-dev supports up to 512 T5 tokens; schnell caps at 256. 512 is the
    # right default for the dev-based community checkpoints that dominate.
    MAX_SEQUENCE_LENGTH: ClassVar[int] = 512

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        # No scheduler cache: FLUX only uses FlowMatchEulerDiscreteScheduler,
        # configured (dynamic shifting) at load. Flat offline tree root + optional
        # CDN mirror, threaded from RuntimeConfig by the Engine.
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # FLUX.1-dev family norm: guidance-distilled scale ~3.5, ~28 steps. schnell
        # (4 steps) overrides num_steps via its catalog model defaults. scheduler is
        # ignored (flow matching). No negative prompt (guidance-distilled).
        return GenerationDefaults(guidance_scale=3.5, num_steps=28)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import FluxPipeline

        if base_model:
            return self._load_with_base_model(local_path, base_model)

        # Self-contained checkpoint. NOTE: this path still infers its config from
        # HF; every offline catalog row sets base_model (the path below), which is
        # the offline-correct one. If a full self-contained FLUX is served offline,
        # mirror its config repo and pass config=<dir> here too.
        logger.info(f"Loading FLUX pipeline from {local_path}")
        return FluxPipeline.from_single_file(
            local_path,
            torch_dtype=torch.bfloat16,
        )

    def _load_with_base_model(self, local_path: str, base_repo: str) -> Any:
        """Load the transformer from local_path + shared components (CLIP-L,
        T5-XXL, VAE, tokenizers, scheduler) from the base repo, resolved into the
        flat offline tree (no HF hit once populated)."""
        import os

        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel

        from imference_engine.runtime.offline import local_repo_dir

        base_dir = local_repo_dir(
            base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
            cdn_base=self._cdn_base)
        logger.info(f"Loading FLUX shared components from {base_dir} (base={base_repo})")

        # Transformer WEIGHTS from the checkpoint; layout (config.json) from the
        # mirrored base tree. local_files_only pins it offline.
        transformer = FluxTransformer2DModel.from_single_file(
            local_path,
            config=os.path.join(base_dir, "transformer"),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )

        # Everything else (CLIP-L, T5-XXL, VAE, both tokenizers, scheduler) from the
        # base tree; the passed transformer overrides the base one.
        return FluxPipeline.from_pretrained(
            base_dir,
            transformer=transformer,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        """Pull a base_model's shared components into the offline tree (no load) —
        same dir _load_with_base_model resolves. No-op without a base_model (a
        self-contained checkpoint has none)."""
        if not base_model:
            return
        from imference_engine.runtime.offline import local_repo_dir
        local_repo_dir(base_model, self.BASE_PATTERNS, self._cache_dir,
                       namespace="image", cdn_base=self._cdn_base)
        logger.info("FLUX base-components warmed (base=%s)", base_model)

    # ------------------------------------------------------------------
    # Img2img
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        from diffusers import FluxImg2ImgPipeline
        return FluxImg2ImgPipeline(
            scheduler=t2i_pipe.scheduler,
            vae=t2i_pipe.vae,
            text_encoder=t2i_pipe.text_encoder,
            tokenizer=t2i_pipe.tokenizer,
            text_encoder_2=t2i_pipe.text_encoder_2,
            tokenizer_2=t2i_pipe.tokenizer_2,
            transformer=t2i_pipe.transformer,
        )

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        # FLUX is guidance-distilled: no classifier-free negative without true-CFG
        # (doubles compute). Pass the positive prompt only; the T5 encoder has no
        # 77-token limit, so no weighted/BREAK chunking either.
        if negative_prompt:
            logger.info(
                "FLUX is guidance-distilled — ignoring negative_prompt %r "
                "(true-CFG negatives are not wired in V1)", negative_prompt)
        return {"prompt": prompt}

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any
    ) -> None:
        """FLUX uses FlowMatchEulerDiscreteScheduler with dynamic shifting set at
        load. The `scheduler` name arg is ignored. An explicit `shift` via
        backend_options rebuilds the scheduler with a fixed shift (advanced;
        overrides dynamic shifting)."""
        shift = kwargs.get("shift")
        if shift is None:
            return  # leave the pipe's default flow-match scheduler
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            pipe.scheduler.config, shift=float(shift), use_dynamic_shifting=False
        )
        logger.info(f"FLUX scheduler: FlowMatchEuler fixed shift={shift}")

    def build_inference_kwargs(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002 — FLUX has no clip_skip
        chunk_size: int,
        generator: Any,
        image: Any = None,
        strength: float = 0.75,
    ) -> dict:
        kwargs = {
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
            "max_sequence_length": self.MAX_SEQUENCE_LENGTH,
        }
        if image is not None:
            # img2img: FluxImg2ImgPipeline takes the size from the source image.
            kwargs["image"] = image
            kwargs["strength"] = strength
        else:
            kwargs["width"] = width
            kwargs["height"] = height
        return kwargs

    def make_generator(self, seed: int, device: str) -> Any:
        import torch
        # Flow matching prefers the generator on the transformer's device, like
        # Z-Image. Fall back to CPU if the device generator is unavailable/flaky.
        try:
            return torch.Generator(device=device).manual_seed(seed)
        except (RuntimeError, ValueError) as e:
            logger.warning(
                f"Generator on device={device} failed ({e}); falling back to CPU"
            )
            return torch.Generator().manual_seed(seed)
