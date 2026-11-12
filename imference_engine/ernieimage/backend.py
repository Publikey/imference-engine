"""ERNIE-Image backend — Baidu's 8B single-stream flow-matching DiT.

Own sub-package parallel to ``imference_engine.flux2klein`` / ``.krea2``.
ERNIE-Image pairs an 8B transformer (36 layers, hidden 4096) with a
**Ministral-3-3B** text encoder (``Mistral3Model``, second-to-last hidden
state) and the **FLUX.2 VAE** (``AutoencoderKLFlux2``). Everything is
Apache-2.0 and ungated. ``ErnieImagePipeline`` is in diffusers 0.40, so this
rides the pinned stack and the generic Engine / ModelManager machinery.

Built for **ERNIE-Image-Turbo** (DMD-distilled): 8 steps, ``guidance_scale=1.0``
— no classifier-free guidance, so the negative prompt is inert. The undistilled
**ERNIE-Image** (50 steps, CFG ~4.0, negative prompt honored) uses the same
backend: register it with ``base_model="baidu/ERNIE-Image"`` and per-model
defaults.

Text-to-image only: diffusers has no ERNIE img2img / edit pipeline
(``make_img2img`` raises, like Krea 2), and the pipeline carries no LoRA loader.

Weights (``weights_path``):

- ``"auto"`` — the build picked for this machine by ``builds.recommend_build``
  (GGUF level + encoder storage per VRAM), fetched into the flat offline tree
  from ``IMAGE_MODEL_CDN`` (or HuggingFace);
- a ``.gguf`` (unsloth ``ERNIE-Image-Turbo-GGUF``);
- a transformer-only ``.safetensors`` (the Comfy-Org repackage / civitai
  finetunes, bf16);
- a diffusers repo id / dir — transformer from ``transformer/``.

The transformer config always comes from the base tree.

GGUF caveat handled here: diffusers' GGUF loader only wraps ``nn.Linear``. The
unsloth files also quantize the patch-embedding ``Conv2d``, which diffusers
leaves as packed uint8 and then crashes on at the first forward —
``dequantize_non_linear`` restores it to bf16 at load (one small layer).

The **prompt enhancer** (``pe``, a second 3B Ministral that rewrites the prompt)
is not loaded: it doubles the language-model footprint (7.7 GB) and makes the
output depend on a sampled rewrite. The pipeline's ``use_pe`` defaults to True
upstream, so every call passes ``use_pe=False`` explicitly.

Text encoder residency (``ERNIE_IMAGE_ENCODER`` = auto | bf16 | fp8): ``fp8``
stores the encoder in float8_e4m3fn with bf16 compute (~3.9 GB instead of
7.7 GB); ``auto`` follows the VRAM table of ``builds.py``.

``shift`` (``backend_options``) overrides the scheduler's fixed flow shift
(4.0 for Turbo); the scheduler name is ignored.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.core.gguf import dequantize_non_linear
from imference_engine.ernieimage.builds import AUTO, recommend_build
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE = "baidu/ERNIE-Image-Turbo"


def encoder_setting() -> str:
    """``ERNIE_IMAGE_ENCODER``: auto (default) | bf16 | fp8."""
    v = os.environ.get("ERNIE_IMAGE_ENCODER", "auto").strip().lower()
    return v if v in ("bf16", "fp8") else "auto"


def _vram_gb() -> Optional[float]:
    """Total VRAM of the current CUDA device in GiB, None without CUDA."""
    from imference_engine.managers.builds import cuda_vram_gb
    return cuda_vram_gb()


class ErnieImageBackend(PipelineBackend):
    """Backend for ERNIE-Image (Turbo) checkpoints."""

    engine: ClassVar[str] = "ernie-image"

    # Shared components from the base repo. The transformer is CONFIG-ONLY (its
    # weights come from the checkpoint) and the prompt enhancer (pe/,
    # pe_tokenizer/ — 7.7 GB) is never pulled.
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*",
        "vae/*",
        "transformer/config.json",
        # Apache-2.0 text, mirrored with the weights it covers.
        "LICENSE",
    ]
    # weights_path = a diffusers repo: the same plus the transformer weights.
    REPO_PATTERNS: ClassVar[list] = [*BASE_PATTERNS, "transformer/*"]

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # Turbo family norm (DMD-distilled). An undistilled ERNIE-Image row
        # overrides both (num_steps ~50, guidance_scale ~4.0).
        return GenerationDefaults(guidance_scale=1.0, num_steps=8)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def resolve_weights(self, weights_path: str) -> str:
        """``"auto"`` -> the local path of the build recommended for this
        machine (downloaded on first use); anything else is returned as is."""
        if weights_path != AUTO:
            return weights_path
        from imference_engine.runtime.offline import fetch

        build = recommend_build(_vram_gb())
        logger.info("ERNIE-Image auto build: %s (%s)", build.name, build.reason)
        if build.offload:
            # Offload is the Engine's RuntimeConfig, not the backend's: say what
            # this build was measured with so a resident run that OOMs is obvious.
            logger.info(
                "ERNIE-Image build %r expects RuntimeConfig(enable_offload=True, "
                "offload_mode=%r) on this card", build.name, build.offload)
        return fetch(build.repo, build.filename, self._cdn_base, self._cache_dir, namespace="image")

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import ErnieImagePipeline, ErnieImageTransformer2DModel

        from imference_engine.runtime.offline import local_repo_dir

        base_repo = base_model or DEFAULT_BASE
        local_path = self.resolve_weights(local_path)
        lower = local_path.lower()
        is_file = not os.path.isdir(local_path)

        if is_file and lower.endswith((".gguf", ".safetensors")):
            base_dir = base_repo if os.path.isdir(base_repo) else local_repo_dir(
                base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
                cdn_base=self._cdn_base)
            logger.info(f"Loading ERNIE-Image shared components from {base_dir} (base={base_repo})")
            cfg_dir = os.path.join(base_dir, "transformer")
            if lower.endswith(".gguf"):
                transformer = self._load_gguf(local_path, cfg_dir)
            else:
                logger.info(f"Loading ERNIE-Image transformer from {local_path}")
                transformer = ErnieImageTransformer2DModel.from_single_file(
                    local_path, config=cfg_dir, local_files_only=True,
                    torch_dtype=torch.bfloat16)
        else:
            base_dir = local_path if not is_file else local_repo_dir(
                local_path, self.REPO_PATTERNS, self._cache_dir, namespace="image",
                cdn_base=self._cdn_base)
            logger.info(f"Loading ERNIE-Image pipeline from {base_dir}")
            transformer = ErnieImageTransformer2DModel.from_pretrained(
                base_dir, subfolder="transformer", local_files_only=True,
                torch_dtype=torch.bfloat16)

        # pe / pe_tokenizer: listed in model_index.json, deliberately not loaded.
        components: dict = {"transformer": transformer, "pe": None, "pe_tokenizer": None}
        if self._resolve_encoder_mode() == "fp8":
            components["text_encoder"] = self._load_fp8_encoder(base_dir)
        return ErnieImagePipeline.from_pretrained(
            base_dir, local_files_only=True, dtype=torch.bfloat16, **components)

    def _load_gguf(self, path: str, cfg_dir: str) -> Any:
        import torch
        from diffusers import ErnieImageTransformer2DModel, GGUFQuantizationConfig

        logger.info("Loading ERNIE-Image GGUF transformer %s", os.path.basename(path))
        transformer = ErnieImageTransformer2DModel.from_single_file(
            path, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=cfg_dir, local_files_only=True, torch_dtype=torch.bfloat16)
        converted = dequantize_non_linear(transformer, torch.bfloat16)
        if converted:
            logger.info("ERNIE-Image GGUF: dequantized non-linear weights %s", converted)
        return transformer

    def _load_fp8_encoder(self, base_dir: str) -> Any:
        """The Ministral-3-3B encoder stored in float8_e4m3fn, computing in
        bf16 (~3.9 GB instead of 7.7 GB)."""
        import torch
        from diffusers.hooks import apply_layerwise_casting
        from transformers import Mistral3Model

        te = Mistral3Model.from_pretrained(
            os.path.join(base_dir, "text_encoder"), local_files_only=True, dtype=torch.bfloat16)
        apply_layerwise_casting(te, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        logger.info("ERNIE-Image text encoder: fp8-resident storage enabled")
        return te

    @staticmethod
    def _resolve_encoder_mode() -> str:
        """bf16 | fp8. auto asks the build table for this machine's VRAM; fp8
        storage needs CUDA (float8 kernels), so anything else stays bf16."""
        vram = _vram_gb()
        if vram is None:
            return "bf16"
        setting = encoder_setting()
        return setting if setting != "auto" else recommend_build(vram).encoder

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        from imference_engine.runtime.offline import local_repo_dir

        base_repo = base_model or DEFAULT_BASE
        if os.path.isdir(base_repo):
            return
        local_repo_dir(base_repo, self.BASE_PATTERNS, self._cache_dir,
                       namespace="image", cdn_base=self._cdn_base)
        logger.info("ERNIE-Image base-components warmed (base=%s)", base_repo)

    # ------------------------------------------------------------------
    # Img2img — unsupported (diffusers ships a text-to-image pipeline only)
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        raise NotImplementedError(
            "ERNIE-Image has no diffusers img2img / edit pipeline (text-to-image "
            "only). Call generate() without source_image."
        )

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        # Raw strings: the pipeline tokenizes without a length cap and takes
        # the encoder's second-to-last hidden state. The negative only acts
        # when guidance_scale > 1 (never on Turbo's 1.0).
        out = {"prompt": prompt}
        if negative_prompt is not None:
            out["negative_prompt"] = negative_prompt
        return out

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any
    ) -> None:
        """FlowMatchEulerDiscreteScheduler with a FIXED shift from the base repo
        (4.0 for Turbo). The ``scheduler`` name is ignored; a ``shift``
        backend_option overrides the shift for this request, and the base value
        is restored on the next request that does not set one."""
        default = getattr(pipe, "_imference_default_scheduler", None)
        if default is None:
            default = pipe.scheduler
            pipe._imference_default_scheduler = default
        shift = kwargs.get("shift")
        if shift is None:
            pipe.scheduler = default
            return
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            default.config, shift=float(shift))
        logger.info(f"ERNIE-Image scheduler: FlowMatchEuler shift={shift}")

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
            # Upstream defaults to True and would call the unloaded enhancer.
            "use_pe": False,
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
