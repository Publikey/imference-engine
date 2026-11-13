"""Low-memory Qwen-Image (text-to-image): GGUF transformer + Lightning LoRA + NF4 encoder.

The same recipe ``edit_lite.py`` validated for Qwen-Image-Edit, applied to the
text-to-image pipeline — and what makes **Qwen-Image-2512** usable below 40 GB
of VRAM (bf16 is a 41 GB transformer plus a 16.6 GB encoder):

- transformer: an unsloth ``qwen-image-2512-Q4_K_M.gguf`` (13.3 GB), loaded with
  ``GGUFQuantizationConfig``;
- Lightning: lightx2v's distillation LoRA kept as an adapter (GGUF weights
  cannot absorb a fused delta) with lightx2v's scheduler — 4 steps (or 8),
  CFG 1.0, so one transformer pass per step instead of two;
- encoder: the 7B Qwen2.5-VL in NF4, loaded for the encode phase and released
  before denoising when VRAM is short. It is byte-identical across Qwen-Image,
  Qwen-Image-2512, Qwen-Image-Edit-2511 and Qwen-Image-Layered (same sha256 on
  HuggingFace), so the pre-quantized copy mirrored for the Edit backend
  (``imference/Qwen-Image-Edit-2511-text-encoder-nf4``) is reused as is, and so
  are the Edit backend's knobs: ``QWEN_EDIT_ENCODER``, ``QWEN_EDIT_KEEP_ENCODER``,
  ``QWEN_EDIT_DENOISE_RESERVE_GB``.

``lite_pipeline_class(base)`` builds the two-phase variant of any Qwen-Image
pipeline class (text-to-image, img2img, layered): they share
``_get_qwen_prompt_embeds`` and differ only in ``prepare_latents``' signature.

Without CUDA + bitsandbytes the encoder stays bf16 in the pipeline and the
regular offload handles it.
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Any, Callable, Optional

from imference_engine.qwenimage import edit_lite

logger = logging.getLogger(__name__)

# Lightning LoRA per transformer family: (repo, {steps: filename}).
LIGHTNING = {
    "2512": ("lightx2v/Qwen-Image-2512-Lightning", {
        4: "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
        8: "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors",
    }),
    "base": ("lightx2v/Qwen-Image-Lightning", {
        4: "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors",
        8: "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors",
    }),
}

# Base files for the lite build: the encoder's CONFIG only (its weights come
# from the NF4 cache), the transformer's config only (weights from the GGUF).
LITE_BASE_PATTERNS = [
    "model_index.json", "scheduler/*",
    "tokenizer/*",
    "vae/*",
    "transformer/config.json",
    "text_encoder/config.json", "text_encoder/generation_config.json",
]


def lightning_steps() -> Optional[int]:
    """``QWEN_IMAGE_LIGHTNING``: "4" (default) | "8" | "none"."""
    v = os.environ.get("QWEN_IMAGE_LIGHTNING", "4").strip().lower()
    if v in ("none", "0", "off", "false"):
        return None
    return 8 if v == "8" else 4


def lightning_family(weights_path: str) -> str:
    """Which Lightning LoRA matches a transformer file: the 2512 one for a 2512
    checkpoint, the original Qwen-Image one otherwise."""
    return "2512" if "2512" in os.path.basename(weights_path) else "base"


def lite_pipeline_class(base_cls: Any) -> Any:
    """``base_cls`` whose text encoder is loaded on demand for the encode phase
    and released before denoising when the free VRAM is short (see
    ``edit_lite``: same policy, same environment knobs)."""

    class QwenImageLitePipeline(base_cls):
        _encoder_loader: Optional[Callable[[], Any]] = None
        # A throwaway wrapper (img2img) must not keep an encoder the resident
        # pipe does not know about.
        _always_release: bool = False

        def _get_qwen_prompt_embeds(self, prompt=None, device=None, dtype=None):
            if self.text_encoder is None and self._encoder_loader is not None:
                self.register_modules(text_encoder=self._encoder_loader())
            target = device or self._execution_device
            enc_device = getattr(self.text_encoder, "device", None) or target
            embeds, mask = super()._get_qwen_prompt_embeds(prompt, enc_device, dtype)
            return embeds.to(target), mask.to(target)

        def get_image_caption(self, *args, **kwargs):
            # Layered pipeline only: captioning calls the encoder directly.
            if self.text_encoder is None and self._encoder_loader is not None:
                self.register_modules(text_encoder=self._encoder_loader())
            return super().get_image_caption(*args, **kwargs)

        def prepare_latents(self, *args, **kwargs):
            # First step after encoding: free the encoder unless there is room
            # to keep it next to the denoise.
            if not self._keep_encoder():
                self.release_encoder()
            return super().prepare_latents(*args, **kwargs)

        def _keep_encoder(self) -> bool:
            if self._encoder_loader is None or self.text_encoder is None:
                return False
            if self._always_release:
                return False
            setting = edit_lite.keep_encoder_setting()
            if setting != "auto":
                return setting == "1"
            try:
                import torch
                if not torch.cuda.is_available():
                    return False
                free, _ = torch.cuda.mem_get_info()
                free += max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
            except Exception:  # noqa: BLE001 — unknown: be safe, free it
                return False
            need = (edit_lite.denoise_reserve_bytes(None, None)
                    + edit_lite.incoming_weights_bytes(self))
            keep = free >= need
            logger.info("Qwen-Image: %.1f GiB free, denoise needs ~%.1f GiB -> encoder %s",
                        free / (1 << 30), need / (1 << 30), "kept" if keep else "released")
            return keep

        def __call__(self, *args, **kwargs):
            ok = False
            # The layered pipeline reads text_encoder.dtype before it encodes
            # anything: have the encoder in place from the start.
            if (hasattr(self, "get_image_caption") and self.text_encoder is None
                    and self._encoder_loader is not None):
                self.register_modules(text_encoder=self._encoder_loader())
            try:
                out = super().__call__(*args, **kwargs)
                ok = True
                return out
            finally:
                # A failed call (OOM...) never leaves the encoder behind.
                if not ok or self._always_release or edit_lite.keep_encoder_setting() == "0":
                    self.release_encoder()

        def release_encoder(self) -> None:
            if self._encoder_loader is None or self.text_encoder is None:
                return
            self.register_modules(text_encoder=None)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    QwenImageLitePipeline.__name__ = base_cls.__name__.replace("Pipeline", "LitePipeline")
    return QwenImageLitePipeline


def apply_lightning(pipe: Any, family: str, steps: int, *,
                    cache_dir: Optional[str], cdn_base: Optional[str]) -> None:
    """Load the Lightning LoRA as an adapter + lightx2v's scheduler."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    from imference_engine.runtime.offline import fetch

    repo, files = LIGHTNING[family]
    path = fetch(repo, files[steps], cdn_base, cache_dir, namespace="image")
    pipe.load_lora_weights(path, adapter_name="lightning")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(edit_lite.LIGHTNING_SCHEDULER)
    logger.info("Qwen-Image: Lightning %d-step LoRA active (%s, adapter, not fused)", steps, family)


def load_lite_pipeline(gguf_path: str, base_repo: str, *, pipeline_cls: Any,
                       cache_dir: Optional[str], cdn_base: Optional[str],
                       full_base_patterns: list, lite_base_patterns: Optional[list] = None,
                       lightning: bool = True) -> Any:
    """GGUF transformer + (NF4 two-phase | bf16) encoder, optionally Lightning.

    ``pipeline_cls`` is the diffusers pipeline to build (``QwenImagePipeline``,
    ``QwenImageLayeredPipeline``…). ``lightning=False`` for checkpoints that
    have no matching distillation LoRA.
    """
    import torch

    from imference_engine.runtime.offline import local_repo_dir

    mode = edit_lite.encoder_mode()
    if os.path.isdir(base_repo):
        base_dir = base_repo
    else:
        patterns = (lite_base_patterns or LITE_BASE_PATTERNS) if mode == "nf4" else full_base_patterns
        base_dir = local_repo_dir(base_repo, patterns, cache_dir, namespace="image", cdn_base=cdn_base)
    transformer = edit_lite.load_gguf_transformer(gguf_path, base_dir)
    # diffusers' GGUF loader only runs nn.Linear from a GGUF: any other layer it
    # finds there stays raw bytes (the layered checkpoint's extra embedding
    # table, stored bf16, comes out as uint8 with twice the width).
    from imference_engine.core.gguf import dequantize_non_linear
    converted = dequantize_non_linear(transformer, torch.bfloat16)
    if converted:
        logger.info("Qwen-Image GGUF: restored non-linear weights %s", converted)
    if mode == "nf4":
        pipe = lite_pipeline_class(pipeline_cls).from_pretrained(
            base_dir, transformer=transformer, text_encoder=None, local_files_only=True,
            dtype=torch.bfloat16)
        pipe._encoder_loader = edit_lite.nf4_encoder_loader(
            base_repo, base_dir, cache_dir=cache_dir, cdn_base=cdn_base)
    else:
        edit_lite.ensure_text_encoder_weights(base_repo, base_dir, cdn_base=cdn_base)
        pipe = pipeline_cls.from_pretrained(
            base_dir, transformer=transformer, local_files_only=True, dtype=torch.bfloat16)
    steps = lightning_steps() if lightning else None
    if steps is not None:
        apply_lightning(pipe, lightning_family(gguf_path), steps,
                        cache_dir=cache_dir, cdn_base=cdn_base)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.enable_tiling()
    logger.info("Qwen-Image lite pipeline ready (%s, encoder=%s, lightning=%s)",
                pipeline_cls.__name__, mode, steps)
    return pipe
