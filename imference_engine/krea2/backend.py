"""Krea 2 (Turbo) backend — Krea AI's 12.9B single-stream flow-matching DiT.

Own sub-package parallel to ``imference_engine.flux`` / ``.qwenimage``. Krea 2
rides the generic Engine / ModelManager / RuntimeConfig machinery on the same
diffusers 0.40 stack (``Krea2Pipeline`` shipped in 0.39; modular in 0.40).

Architecture (for context): single-stream DiT (GQA, gated sigmoid attention,
3-axis RoPE) + a **Qwen3-VL-4B** text encoder whose hidden states are tapped at
12 layers and fused inside the transformer + the **Qwen-Image VAE**
(``AutoencoderKLQwenImage`` — the same VAE family the qwenimage backend uses).

This backend is built for **Krea 2 Turbo** (the TDM few-step distillate) and
the civitai/ComfyUI finetune ecosystem around it:

- **Checkpoint form**: transformer-only single-file ``.safetensors`` in the
  NATIVE key layout, predominantly ComfyUI "scaled fp8" (float8_e4m3fn +
  per-tensor ``weight_scale``); bf16 and all-in-one prefixed variants too.
  diffusers has no ``from_single_file`` for Krea 2 (issue #14122) and no
  scaled-fp8 path at all, so ``krea2/convert.py`` normalizes the state dict IN
  MEMORY at load (prefix strip → exact fp8 dequant → native→diffusers key
  remap) and the result is injected into ``Krea2Pipeline`` from the base repo's
  components — the same transformer-only composition as FLUX/Chroma/Qwen-Image.
- **base_model is REQUIRED** (e.g. ``krea/Krea-2-Turbo`` — gated, accept the
  Krea 2 Community License; mirror it on the CDN for offline workers). The
  single files carry no text encoder / VAE / scheduler / tokenizer.
- **fp8-resident storage**: when the checkpoint was fp8 on disk, the
  dequantized transformer is re-cast to float8_e4m3fn storage with bf16 compute
  via diffusers layerwise casting (same approach as InvokeAI) — ~13 GB resident
  instead of ~26 GB, which is the whole point of the fp8 ecosystem. Env
  override: ``KREA2_FP8_STORAGE=1|0`` (unset = auto: fp8 source + CUDA).
- **Guidance**: Krea 2 Turbo is TDM-distilled WITHOUT CFG — the family norm is
  ``guidance_scale=0.0`` + ``num_steps=8``. The pipeline's ``guidance_scale``
  follows the KREA convention (velocity ``cond + g*(cond-uncond)``; g>0 enables
  guidance; conventional CFG scale ≈ 1+g), and ``negative_prompt`` is ignored
  whenever g<=0 — like FLUX in the behavior matrix. The scheduler name is
  ignored (FlowMatchEulerDiscreteScheduler with resolution-aware dynamic
  shifting; Turbo checkpoints carry ``is_distilled=true`` → fixed mu=1.15).
- **t2i only**: diffusers has no Krea 2 img2img yet (PR #14290 open) —
  ``make_img2img`` raises, like Anima.

Known upstream quirks handled here (both observed by InvokeAI):
- the base repo's ``model_index.json`` declares the slow ``Qwen2Tokenizer`` but
  ships only ``tokenizer.json`` → the tokenizer is loaded via ``AutoTokenizer``
  (resolving to the fast one) and passed into ``from_pretrained`` explicitly;
- older transformers read Qwen3-VL rope settings from ``rope_scaling`` while
  the repo stores ``rope_parameters`` — NOT patched here (transformers >= 5.3,
  the repo floor, reads ``rope_parameters`` natively); revisit if a validation
  run crashes in the rotary embedding.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)


class Krea2Backend(PipelineBackend):
    """Backend for Krea 2 (Turbo) checkpoints (.safetensors single-file)."""

    engine: ClassVar[str] = "krea2"

    # Files needed from a Krea 2 base repo (krea/Krea-2-Turbo, diffusers format).
    # Tokenizer + Qwen3-VL text encoder + VAE + scheduler are FULL weights/config;
    # the transformer is CONFIG-ONLY (the weights come from the checkpoint).
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
        # Turbo family norm: TDM-distilled, guidance OFF, 8 steps. A Raw
        # (undistilled) checkpoint's catalog row overrides these
        # (num_steps ~28-52, guidance_scale ~3.5-4.5 in the Krea convention).
        return GenerationDefaults(guidance_scale=0.0, num_steps=8)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        if not base_model:
            raise ValueError(
                "Krea 2 checkpoints are transformer-only single files: register "
                "with base_model=<diffusers base repo> (e.g. 'krea/Krea-2-Turbo') "
                "supplying the Qwen3-VL text encoder, VAE, tokenizer and scheduler."
            )
        return self._load_with_base_model(local_path, base_model)

    def _load_with_base_model(self, local_path: str, base_repo: str) -> Any:
        """Transformer WEIGHTS from local_path (native/ComfyUI layout, fp8 or
        bf16 — normalized in memory by krea2/convert.py) + shared components
        from the base repo, resolved into the flat offline tree."""
        import accelerate
        import torch
        from diffusers import Krea2Pipeline, Krea2Transformer2DModel
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        from imference_engine.krea2.convert import (
            KREA2_TRANSFORMER_CONFIG,
            prepare_krea2_state_dict,
            reject_incomplete_load,
        )
        from imference_engine.runtime.offline import local_repo_dir

        base_dir = local_repo_dir(
            base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
            cdn_base=self._cdn_base)
        logger.info(f"Loading Krea 2 shared components from {base_dir} (base={base_repo})")

        # --- transformer: normalize the single file in memory, then assign.
        logger.info(f"Loading Krea 2 transformer from {local_path}")
        sd = load_file(local_path)
        sd, source_was_fp8 = prepare_krea2_state_dict(sd, torch.bfloat16)

        transformer_cfg_dir = os.path.join(base_dir, "transformer")
        if os.path.isfile(os.path.join(transformer_cfg_dir, "config.json")):
            cfg = Krea2Transformer2DModel.load_config(
                transformer_cfg_dir, local_files_only=True)
        else:  # bare fallback — BASE_PATTERNS normally guarantees the config
            cfg = dict(KREA2_TRANSFORMER_CONFIG)
        with accelerate.init_empty_weights():
            transformer = Krea2Transformer2DModel.from_config(cfg)
        transformer.load_state_dict(sd, assign=True, strict=False)
        reject_incomplete_load(transformer, what="Krea 2 single-file checkpoint")
        del sd

        # --- fp8-resident storage (auto for fp8 sources on CUDA; the point of
        # the civitai fp8 ecosystem: ~13 GB resident instead of ~26 GB).
        # KREA2_FP8_STORAGE=1|0 forces it on/off. Compute stays bf16; norm-like
        # modules are excluded by diffusers' default skip patterns.
        if self._resolve_fp8_storage(source_was_fp8):
            try:
                transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn,
                    compute_dtype=torch.bfloat16,
                )
                logger.info(
                    "Krea 2 transformer: fp8-resident storage enabled "
                    "(storage=float8_e4m3fn, compute=bfloat16)")
            except Exception as e:  # noqa: BLE001 — bf16 fallback beats a hard fail
                logger.warning(f"Krea 2 fp8-resident storage unavailable ({e}); staying bf16")

        # --- tokenizer: model_index.json declares the slow Qwen2Tokenizer but
        # the repo ships only tokenizer.json, so resolve via AutoTokenizer (fast)
        # and inject it. extra_special_tokens={} works around the list-format
        # tokenizer_config the repo ships (tokens are baked into tokenizer.json).
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(base_dir, "tokenizer"), local_files_only=True,
            extra_special_tokens={},
        )

        return Krea2Pipeline.from_pretrained(
            base_dir,
            transformer=transformer,
            tokenizer=tokenizer,
            local_files_only=True,
            dtype=torch.bfloat16,
        )

    @staticmethod
    def _resolve_fp8_storage(source_was_fp8: bool) -> bool:
        """KREA2_FP8_STORAGE=1 forces on, =0 forces off; unset = auto (fp8
        source + CUDA available)."""
        import torch

        env = os.environ.get("KREA2_FP8_STORAGE", "").strip()
        if env in ("1", "true", "yes"):
            return True
        if env in ("0", "false", "no"):
            return False
        return source_was_fp8 and torch.cuda.is_available()

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        if not base_model:
            return
        from imference_engine.runtime.offline import local_repo_dir
        local_repo_dir(base_model, self.BASE_PATTERNS, self._cache_dir,
                       namespace="image", cdn_base=self._cdn_base)
        logger.info("Krea 2 base-components warmed (base=%s)", base_model)

    # ------------------------------------------------------------------
    # Img2img — unsupported (no diffusers Krea 2 img2img yet; PR #14290 open)
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        raise NotImplementedError(
            "Krea 2 has no diffusers img2img pipeline yet (upstream PR #14290 "
            "is open; text-to-image only). Call generate() without source_image."
        )

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        # Qwen3-VL chat-template tokenization inside the pipeline (fixed 512
        # tokens) — raw strings, no 77-token limit, no weighted embeddings.
        # negative_prompt passes through but the pipeline ignores it whenever
        # guidance_scale <= 0 (the Turbo norm).
        out = {"prompt": prompt}
        if negative_prompt is not None:
            out["negative_prompt"] = negative_prompt
        return out

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Krea 2 uses FlowMatchEulerDiscreteScheduler with resolution-aware
        dynamic shifting; a Turbo checkpoint's ``is_distilled`` config pins a
        fixed mu=1.15 inside the pipeline. Neither the ``scheduler`` name nor a
        ``shift`` backend_option applies — no-op."""
        if scheduler:
            logger.debug("Krea 2 ignores scheduler=%r (flow matching only)", scheduler)

    def build_inference_kwargs(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002 — Krea 2 has no clip_skip
        chunk_size: int,
        generator: Any,
        image: Any = None,  # noqa: ARG002 — img2img unsupported (make_img2img raises)
        strength: float = 0.75,  # noqa: ARG002
    ) -> dict:
        # guidance_scale follows the KREA convention (0.0 = off, the Turbo
        # norm; velocity cond + g*(cond-uncond), conventional CFG ≈ 1+g).
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
