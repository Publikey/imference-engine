"""FLUX.2 [klein] backend — Black Forest Labs' 4B rectified-flow DiT.

Own sub-package parallel to ``imference_engine.flux`` / ``.krea2``. klein is
the small, Apache-2.0 member of the FLUX.2 family (the 9B and [dev] are
non-commercial): a 3.9B double/single-stream transformer conditioned on a
**Qwen3-4B** text encoder (three hidden layers stacked) with the **FLUX.2 VAE**
(``AutoencoderKLFlux2``). ``Flux2KleinPipeline`` has been in diffusers since
0.37, so this rides the pinned 0.40 stack and the generic Engine / ModelManager
machinery.

One pipeline, two uses:

- **text-to-image** — ``generate(prompt=...)``;
- **reference editing** — ``generate(prompt=<instruction>, source_image=<img or
  [imgs]>)``. The image is a CONDITION (its latents are appended to the token
  sequence), not a noised starting point: there is no ``strength``, and the
  output takes the reference's size. ``make_img2img`` returns the same pipe.

Two checkpoint families, told apart by ``is_distilled`` in the base repo's
``model_index.json``:

- **klein 4B** (``black-forest-labs/FLUX.2-klein-4B``, the default base) —
  step-distilled: 4 steps, guidance fixed at 1.0, no negative prompt;
- **klein Base 4B** (``…-klein-base-4B``) — undistilled: ~50 steps, true CFG
  (~4.0), negative prompt honored. Register it with ``base_model=`` that repo
  and per-model defaults.

Weights (``weights_path``):

- ``"auto"`` — the build picked for this machine by ``builds.recommend_build``
  (free VRAM → bf16 / fp8 / GGUF), fetched into the flat offline tree from
  ``IMAGE_MODEL_CDN`` (or HuggingFace);
- a transformer-only ``.safetensors`` in BFL's native layout — bf16, or the
  official **scaled fp8** half-size file, dequantized in memory by
  ``convert.py`` and kept fp8-resident (layerwise casting, bf16 compute);
- a ``.gguf`` (unsloth Q8_0 / Q5_K_M / Q4_K_M) — ``GGUFQuantizationConfig``;
- a diffusers repo id / dir — transformer from ``transformer/``.

The single files carry no config: diffusers 0.40's single-file detection only
knows FLUX.2 [dev] (it would build the 32B layout), so the transformer config
always comes from the base tree (``config=<base>/transformer``).

Text encoder residency (``FLUX2_KLEIN_ENCODER`` = auto | bf16 | fp8): the
encoder is as big as the transformer (8 GB bf16). ``fp8`` stores it in
float8_e4m3fn with bf16 compute (~4 GB); ``auto`` follows the same VRAM table
as the build choice.

The scheduler name and ``shift`` are ignored: the pipeline computes the
flow-match ``mu`` from the token count itself.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.flux2klein.builds import AUTO, recommend_build
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE = "black-forest-labs/FLUX.2-klein-4B"


def encoder_setting() -> str:
    """``FLUX2_KLEIN_ENCODER``: auto (default) | bf16 | fp8."""
    v = os.environ.get("FLUX2_KLEIN_ENCODER", "auto").strip().lower()
    return v if v in ("bf16", "fp8") else "auto"


def fp8_storage_setting() -> str:
    """``FLUX2_KLEIN_FP8_STORAGE``: auto (default) | 1 | 0 — fp8-resident
    storage for the transformer. auto = on when the checkpoint is the fp8 file
    (the caller already chose the small footprint)."""
    v = os.environ.get("FLUX2_KLEIN_FP8_STORAGE", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return "1"
    if v in ("0", "false", "no", "off"):
        return "0"
    return "auto"


def _vram_gb() -> Optional[float]:
    """Total VRAM of the current CUDA device in GiB, None without CUDA."""
    from imference_engine.managers.builds import cuda_vram_gb
    return cuda_vram_gb()


def _pipeline_class():
    import torch
    from diffusers import Flux2KleinPipeline

    class Flux2KleinEnginePipeline(Flux2KleinPipeline):
        """``Flux2KleinPipeline`` that keeps prompt embeddings in the compute
        dtype. Upstream takes ``dtype = text_encoder.dtype``; an fp8-storage
        encoder reports float8 there, and the latents (created in the embeds'
        dtype) would then be sampled in float8 — which torch cannot do."""

        @staticmethod
        def _get_qwen3_prompt_embeds(text_encoder, tokenizer, prompt, dtype=None, device=None, **kwargs):
            if dtype is None and text_encoder.dtype == torch.float8_e4m3fn:
                dtype = torch.bfloat16
            return Flux2KleinPipeline._get_qwen3_prompt_embeds(
                text_encoder, tokenizer, prompt, dtype=dtype, device=device, **kwargs)

    return Flux2KleinEnginePipeline


class Flux2KleinBackend(PipelineBackend):
    """Backend for FLUX.2 [klein] 4B checkpoints."""

    engine: ClassVar[str] = "flux2-klein"

    # Shared components from the base repo. The transformer is CONFIG-ONLY: its
    # weights come from the checkpoint (bf16 / fp8 / GGUF).
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*",
        "vae/*",
        "transformer/config.json",
        # Apache-2.0 text, mirrored with the weights it covers.
        "LICENSE.md",
    ]
    # weights_path = a diffusers repo: the same plus the transformer weights
    # (never the root single file, which duplicates them).
    REPO_PATTERNS: ClassVar[list] = [*BASE_PATTERNS, "transformer/*"]

    # Qwen3 chat-template tokens; the pipeline default.
    MAX_SEQUENCE_LENGTH: ClassVar[int] = 512

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # klein 4B (distilled) family norm. A Base checkpoint's catalog row
        # overrides both (num_steps ~50, guidance_scale ~4.0).
        return GenerationDefaults(guidance_scale=1.0, num_steps=4)

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
        logger.info("FLUX.2 klein auto build: %s (%s)", build.name, build.reason)
        if build.offload:
            # Offload is the Engine's RuntimeConfig, not the backend's: say what
            # this build was measured with so a resident run that OOMs is obvious.
            logger.info(
                "FLUX.2 klein build %r expects RuntimeConfig(enable_offload=True, "
                "offload_mode=%r) on this card", build.name, build.offload)
        return fetch(build.repo, build.filename, self._cdn_base, self._cache_dir, namespace="image")

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch

        from imference_engine.runtime.offline import local_repo_dir

        base_repo = base_model or DEFAULT_BASE
        local_path = self.resolve_weights(local_path)
        lower = local_path.lower()
        is_file = not os.path.isdir(local_path)

        if is_file and lower.endswith((".gguf", ".safetensors")):
            base_dir = base_repo if os.path.isdir(base_repo) else local_repo_dir(
                base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
                cdn_base=self._cdn_base)
            logger.info(f"Loading FLUX.2 klein shared components from {base_dir} (base={base_repo})")
            if lower.endswith(".gguf"):
                transformer = self._load_gguf(local_path, base_dir)
            else:
                transformer = self._load_single_file(local_path, base_dir)
        else:
            base_dir = local_path if not is_file else local_repo_dir(
                local_path, self.REPO_PATTERNS, self._cache_dir, namespace="image",
                cdn_base=self._cdn_base)
            from diffusers import Flux2Transformer2DModel
            logger.info(f"Loading FLUX.2 klein pipeline from {base_dir}")
            transformer = Flux2Transformer2DModel.from_pretrained(
                base_dir, subfolder="transformer", local_files_only=True,
                torch_dtype=torch.bfloat16)

        components: dict = {"transformer": transformer}
        if self._resolve_encoder_mode() == "fp8":
            components["text_encoder"] = self._load_fp8_encoder(base_dir)
        return _pipeline_class().from_pretrained(
            base_dir, local_files_only=True, dtype=torch.bfloat16, **components)

    def _load_gguf(self, path: str, base_dir: str) -> Any:
        import torch
        from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig

        logger.info("Loading FLUX.2 klein GGUF transformer %s", os.path.basename(path))
        return Flux2Transformer2DModel.from_single_file(
            path, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=os.path.join(base_dir, "transformer"), local_files_only=True,
            torch_dtype=torch.bfloat16)

    def _load_single_file(self, path: str, base_dir: str) -> Any:
        """Native-layout ``.safetensors``: bf16 goes through diffusers'
        ``from_single_file``; BFL's scaled fp8 is dequantized in memory first
        (diffusers has no path for it) and then kept fp8-resident."""
        import torch
        from diffusers import Flux2Transformer2DModel

        from imference_engine.flux2klein.convert import (
            dequantize_scaled_fp8,
            detect_format,
            read_safetensors_header,
        )

        cfg_dir = os.path.join(base_dir, "transformer")
        fmt = detect_format(read_safetensors_header(path))
        logger.info(f"Loading FLUX.2 klein transformer from {path} (format={fmt})")
        if fmt == "nvfp4":
            raise ValueError(
                f"{os.path.basename(path)} is an NVFP4 build (NVIDIA 4-bit float): "
                "not supported — use the bf16, fp8 or GGUF build of FLUX.2 klein.")
        if fmt == "bf16":
            transformer = Flux2Transformer2DModel.from_single_file(
                path, config=cfg_dir, local_files_only=True, torch_dtype=torch.bfloat16)
        else:
            import accelerate
            from diffusers.loaders.single_file_utils import (
                convert_flux2_transformer_checkpoint_to_diffusers,
            )
            from safetensors.torch import load_file

            sd = dequantize_scaled_fp8(load_file(path), torch.bfloat16)
            sd = convert_flux2_transformer_checkpoint_to_diffusers(sd)
            with accelerate.init_empty_weights():
                transformer = Flux2Transformer2DModel.from_config(
                    Flux2Transformer2DModel.load_config(cfg_dir, local_files_only=True))
            missing, unexpected = transformer.load_state_dict(sd, assign=True, strict=False)
            del sd
            if missing or unexpected:
                raise ValueError(
                    f"{os.path.basename(path)} does not match the FLUX.2 klein transformer "
                    f"layout of {base_dir} ({len(missing)} missing, {len(unexpected)} "
                    f"unexpected keys; first missing: {list(missing)[:3]})")

        if self._resolve_fp8_storage(fmt != "bf16"):
            try:
                transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
                logger.info("FLUX.2 klein transformer: fp8-resident storage enabled")
            except Exception as e:  # noqa: BLE001 — bf16 beats a hard fail
                logger.warning(f"FLUX.2 klein fp8-resident storage unavailable ({e}); staying bf16")
        return transformer

    def _load_fp8_encoder(self, base_dir: str) -> Any:
        """The Qwen3-4B encoder stored in float8_e4m3fn, computing in bf16
        (~4 GB instead of 8 GB)."""
        import torch
        from diffusers.hooks import apply_layerwise_casting
        from transformers import Qwen3ForCausalLM

        te = Qwen3ForCausalLM.from_pretrained(
            os.path.join(base_dir, "text_encoder"), local_files_only=True, dtype=torch.bfloat16)
        apply_layerwise_casting(te, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        logger.info("FLUX.2 klein text encoder: fp8-resident storage enabled")
        return te

    @staticmethod
    def _resolve_fp8_storage(source_is_fp8: bool) -> bool:
        setting = fp8_storage_setting()
        if setting != "auto":
            return setting == "1"
        try:
            import torch
            return source_is_fp8 and torch.cuda.is_available()
        except ImportError:
            return False

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
        logger.info("FLUX.2 klein base-components warmed (base=%s)", base_repo)

    # ------------------------------------------------------------------
    # Reference editing — the same pipe
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        # The image is passed to the same pipeline as a reference condition.
        return t2i_pipe

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        out: dict = {"prompt": prompt}
        if not negative_prompt or not negative_prompt.strip():
            return out
        distilled = bool(getattr(getattr(pipe, "config", None), "is_distilled", True))
        if distilled:
            logger.info(
                "FLUX.2 klein (distilled) has no classifier-free guidance — "
                "ignoring negative_prompt %r", negative_prompt)
            return out
        # Base (undistilled): the pipeline takes no negative_prompt string (it
        # encodes "" itself), only pre-computed embeddings.
        embeds, _ = pipe.encode_prompt(
            prompt=negative_prompt, max_sequence_length=self.MAX_SEQUENCE_LENGTH)
        out["negative_prompt_embeds"] = embeds
        return out

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any  # noqa: ARG002
    ) -> None:
        """No-op: FlowMatchEulerDiscreteScheduler with a ``mu`` the pipeline
        derives from the token count and step count on every call."""
        if scheduler:
            logger.debug("FLUX.2 klein ignores scheduler=%r (flow matching only)", scheduler)

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
        image: Any = None,
        strength: float = 0.75,  # noqa: ARG002 — the image is a condition, not a start
    ) -> dict:
        kwargs = {
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
            "max_sequence_length": self.MAX_SEQUENCE_LENGTH,
        }
        if image is not None:
            # One reference or several ("the jacket from image 2 on the person
            # in image 1"). The output takes the first reference's size.
            kwargs["image"] = list(image) if isinstance(image, (list, tuple)) else [image]
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
