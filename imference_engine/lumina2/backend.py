"""Lumina-Image 2.0 backend — Alpha-VLLM's 2.6B flow-matching DiT.

Own sub-package parallel to ``imference_engine.ernieimage`` / ``.flux2klein``.
Lumina-Image 2.0 is a small single-stream transformer conditioned on
**Gemma-2-2b** (``Gemma2Model``, a system prompt is prepended by the pipeline)
with the FLUX.1 VAE. The model is Apache-2.0; the Gemma encoder comes under
Google's Gemma Terms of Use (redistribution allowed with the terms attached —
they are mirrored next to the weights). ``Lumina2Pipeline`` is in diffusers
0.40, so this rides the pinned stack and the generic Engine machinery.

Its ecosystem is the anime finetunes — **Neta Lumina** and **NetaYume Lumina**
(both Apache-2.0) — which ship as ComfyUI single files. Those load as-is:

- transformer-only ``.safetensors`` (``Unet/…``, Comfy-Org
  ``lumina_2_model_bf16``) — diffusers' ``from_single_file``;
- **all-in-one** ``.safetensors`` (transformer + Gemma + VAE in one file, the
  common civitai form) — diffusers cannot load it, so the backend reads only the
  ``model.diffusion_model.*`` tensors and converts them; the encoder and VAE
  inside the file are identical to the base ones and are not read;
- a ``.gguf`` (calcuis ``lumina-gguf``, neta-art ``neta-lumina-gguf``);
- ``"auto"`` — the build picked for this machine by ``builds.recommend_build``;
- a diffusers repo id / dir.

Undistilled: true classifier-free guidance on every request (engine defaults
30 steps, ``guidance_scale=4.0``), so the negative prompt is honored and a step
costs two transformer passes. Text-to-image only (no diffusers img2img
pipeline; ``make_img2img`` raises).

``backend_options``:

- ``shift`` — overrides the scheduler's fixed flow shift (6.0);
- ``cfg_trunc_ratio`` — fraction of the schedule after which guidance is
  dropped (1.0 = guidance on every step, the default; lower is faster);
- ``system_prompt`` — replaces the pipeline's built-in system prompt (the
  finetunes document their own).

Base components: upstream stores the Gemma encoder in fp32 (10.4 GB). The CDN
mirror carries a derived base, ``imference/Lumina-Image-2.0-bf16``, with the
same files and the encoder re-saved in bf16 (5.2 GB) — what the pipeline casts
it to anyway. It is used when ``IMAGE_MODEL_CDN`` serves it; otherwise the
upstream repo is pulled.

Text encoder residency (``LUMINA2_ENCODER`` = auto | bf16 | fp8): ``fp8``
stores the encoder in float8_e4m3fn with bf16 compute (~2.6 GB); ``auto``
follows the VRAM table of ``builds.py``.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.core.gguf import dequantize_non_linear
from imference_engine.lumina2.builds import AUTO, recommend_build
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE = "Alpha-VLLM/Lumina-Image-2.0"
# Derived artifact (the base with its encoder re-saved in bf16): our mirror
# only, never on HuggingFace. Absent -> the upstream base (fp32 encoder).
SLIM_BASE = "imference/Lumina-Image-2.0-bf16"

AIO_PREFIX = "model.diffusion_model."


def encoder_setting() -> str:
    """``LUMINA2_ENCODER``: auto (default) | bf16 | fp8."""
    v = os.environ.get("LUMINA2_ENCODER", "auto").strip().lower()
    return v if v in ("bf16", "fp8") else "auto"


def _vram_gb() -> Optional[float]:
    """Total VRAM of the current CUDA device in GiB, None without CUDA."""
    from imference_engine.managers.builds import cuda_vram_gb
    return cuda_vram_gb()


def is_all_in_one(header: dict) -> bool:
    """True for a ComfyUI all-in-one checkpoint: the transformer sits under
    ``model.diffusion_model.`` next to other components (text encoder, VAE)."""
    names = [k for k in header if k != "__metadata__"]
    has_model = any(k.startswith(AIO_PREFIX) for k in names)
    return has_model and any(not k.startswith(AIO_PREFIX) for k in names)


class Lumina2Backend(PipelineBackend):
    """Backend for Lumina-Image 2.0 checkpoints and their finetunes."""

    engine: ClassVar[str] = "lumina2"

    # Shared components from the base repo. The transformer is CONFIG-ONLY: its
    # weights come from the checkpoint.
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*",
        "vae/*",
        "transformer/config.json",
        # Licence texts mirrored with the weights they cover (the derived base
        # ships the Apache-2.0 notice and the Gemma Terms of Use).
        "LICENSE*", "NOTICE*", "GEMMA_TERMS*",
    ]
    # weights_path = a diffusers repo: the same plus the transformer weights.
    REPO_PATTERNS: ClassVar[list] = [*BASE_PATTERNS, "transformer/*"]

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base
        # Per-request pipe(...) extras from backend_options. Set in
        # apply_scheduler (the hook that receives backend_options) and read in
        # build_inference_kwargs right after — the Engine is single-threaded.
        self._call_options: dict = {}

    def engine_defaults(self) -> GenerationDefaults:
        # Undistilled: true CFG. 30 steps is the floor the finetunes document
        # (Neta 30, NetaYume 40-50, base 50) — catalog rows raise it per model.
        return GenerationDefaults(guidance_scale=4.0, num_steps=30)

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
        logger.info("Lumina-Image 2.0 auto build: %s (%s)", build.name, build.reason)
        if build.offload:
            logger.info(
                "Lumina-Image 2.0 build %r expects RuntimeConfig(enable_offload=True, "
                "offload_mode=%r) on this card", build.name, build.offload)
        return fetch(build.repo, build.filename, self._cdn_base, self._cache_dir, namespace="image")

    def _base_dir(self, base_repo: str) -> str:
        """Local dir of the shared components. For the default base, the slim
        mirror (bf16 encoder) when the CDN serves it, else the upstream repo."""
        from imference_engine.runtime import offline

        if os.path.isdir(base_repo):
            return base_repo
        if self._cdn_base and base_repo == DEFAULT_BASE:
            d = os.path.join(offline.flat_root(self._cache_dir, namespace="image"), SLIM_BASE)
            try:
                if not offline._cdn_complete(d, None):
                    offline._cdn_snapshot(self._cdn_base, SLIM_BASE, d)
                return d
            except Exception as e:  # noqa: BLE001 — not mirrored: upstream base
                logger.info("Slim Lumina base not on the CDN (%s); using %s", e, base_repo)
        return offline.local_repo_dir(
            base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
            cdn_base=self._cdn_base)

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import Lumina2Pipeline, Lumina2Transformer2DModel

        from imference_engine.runtime.offline import local_repo_dir

        base_repo = base_model or DEFAULT_BASE
        local_path = self.resolve_weights(local_path)
        lower = local_path.lower()
        is_file = not os.path.isdir(local_path)

        if is_file and lower.endswith((".gguf", ".safetensors")):
            base_dir = self._base_dir(base_repo)
            logger.info(f"Loading Lumina-Image 2.0 shared components from {base_dir} (base={base_repo})")
            cfg_dir = os.path.join(base_dir, "transformer")
            if lower.endswith(".gguf"):
                transformer = self._load_gguf(local_path, cfg_dir)
            else:
                transformer = self._load_single_file(local_path, cfg_dir)
        else:
            base_dir = local_path if not is_file else local_repo_dir(
                local_path, self.REPO_PATTERNS, self._cache_dir, namespace="image",
                cdn_base=self._cdn_base)
            logger.info(f"Loading Lumina-Image 2.0 pipeline from {base_dir}")
            transformer = Lumina2Transformer2DModel.from_pretrained(
                base_dir, subfolder="transformer", local_files_only=True,
                torch_dtype=torch.bfloat16)

        components: dict = {"transformer": transformer}
        if self._resolve_encoder_mode() == "fp8":
            components["text_encoder"] = self._load_fp8_encoder(base_dir)
        return Lumina2Pipeline.from_pretrained(
            base_dir, local_files_only=True, dtype=torch.bfloat16, **components)

    def _load_gguf(self, path: str, cfg_dir: str) -> Any:
        import torch
        from diffusers import GGUFQuantizationConfig, Lumina2Transformer2DModel

        logger.info("Loading Lumina-Image 2.0 GGUF transformer %s", os.path.basename(path))
        transformer = Lumina2Transformer2DModel.from_single_file(
            path, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=cfg_dir, local_files_only=True, torch_dtype=torch.bfloat16)
        converted = dequantize_non_linear(transformer, torch.bfloat16)
        if converted:
            logger.info("Lumina-Image 2.0 GGUF: dequantized non-linear weights %s", converted)
        return transformer

    def _load_single_file(self, path: str, cfg_dir: str) -> Any:
        """Transformer-only files go through diffusers' ``from_single_file``. An
        all-in-one file is filtered to its transformer tensors and converted
        here (diffusers trips over the bundled encoder / VAE keys)."""
        import torch
        from diffusers import Lumina2Transformer2DModel

        from imference_engine.flux2klein.convert import read_safetensors_header

        header = read_safetensors_header(path)
        if not is_all_in_one(header):
            logger.info(f"Loading Lumina-Image 2.0 transformer from {path}")
            return Lumina2Transformer2DModel.from_single_file(
                path, config=cfg_dir, local_files_only=True, torch_dtype=torch.bfloat16)

        import accelerate
        from diffusers.loaders.single_file_utils import convert_lumina2_to_diffusers
        from safetensors import safe_open

        logger.info(f"Loading Lumina-Image 2.0 transformer from all-in-one {path}")
        sd = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(AIO_PREFIX):
                    sd[key[len(AIO_PREFIX):]] = f.get_tensor(key).to(torch.bfloat16)
        sd = convert_lumina2_to_diffusers(sd)
        with accelerate.init_empty_weights():
            transformer = Lumina2Transformer2DModel.from_config(
                Lumina2Transformer2DModel.load_config(cfg_dir, local_files_only=True))
        missing, unexpected = transformer.load_state_dict(sd, assign=True, strict=False)
        del sd
        if missing or unexpected:
            raise ValueError(
                f"{os.path.basename(path)} does not match the Lumina-Image 2.0 transformer "
                f"layout ({len(missing)} missing, {len(unexpected)} unexpected keys; "
                f"first missing: {list(missing)[:3]})")
        return transformer

    def _load_fp8_encoder(self, base_dir: str) -> Any:
        """The Gemma-2-2b encoder stored in float8_e4m3fn, computing in bf16
        (~3.2 GB instead of 5.2 GB).

        The token embedding is left in bf16 on purpose: it is the encoder's
        first parameter, so ``text_encoder.dtype`` keeps reporting bf16. The
        pipeline casts the prompt embeddings to that dtype and creates the
        latents in it — a fully-fp8 encoder would make both float8 (lossy
        embeds, and torch cannot sample float8 noise)."""
        import torch
        from diffusers.hooks import apply_layerwise_casting
        from transformers import Gemma2Model

        te = Gemma2Model.from_pretrained(
            os.path.join(base_dir, "text_encoder"), local_files_only=True, dtype=torch.bfloat16)
        apply_layerwise_casting(
            te, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
            skip_modules_pattern=("embed_tokens", "norm"))
        logger.info("Lumina-Image 2.0 text encoder: fp8-resident storage enabled")
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
        self._base_dir(base_model or DEFAULT_BASE)
        logger.info("Lumina-Image 2.0 base-components warmed (base=%s)", base_model or DEFAULT_BASE)

    # ------------------------------------------------------------------
    # Img2img — unsupported (diffusers ships a text-to-image pipeline only)
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        raise NotImplementedError(
            "Lumina-Image 2.0 has no diffusers img2img pipeline (text-to-image "
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
        # Raw strings: the pipeline prepends its system prompt and tokenizes
        # (256 tokens). True CFG: the negative prompt is always used.
        out = {"prompt": prompt}
        if negative_prompt is not None:
            out["negative_prompt"] = negative_prompt
        return out

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any
    ) -> None:
        """FlowMatchEulerDiscreteScheduler with the base repo's fixed shift
        (6.0). The ``scheduler`` name is ignored; a ``shift`` backend_option
        overrides it for this request and the base scheduler is restored on the
        next one. Also records the per-request ``cfg_trunc_ratio`` /
        ``system_prompt`` options for ``build_inference_kwargs``."""
        self._call_options = {}
        if kwargs.get("cfg_trunc_ratio") is not None:
            self._call_options["cfg_trunc_ratio"] = float(kwargs["cfg_trunc_ratio"])
        if kwargs.get("system_prompt") is not None:
            self._call_options["system_prompt"] = str(kwargs["system_prompt"])

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
        logger.info(f"Lumina-Image 2.0 scheduler: FlowMatchEuler shift={shift}")

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
            **self._call_options,
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
