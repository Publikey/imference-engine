"""Z-Image backend — Tongyi Z-Image pipelines (text-to-image, flow-matching).

Z-Image checkpoints come in two flavors:
- *Full* base/turbo: a self-contained safetensors with all components → just
  call from_single_file.
- *Finetunes / ComfyUI-style*: a transformer-only safetensors that needs
  tokenizer + text_encoder + VAE loaded from a base HF repo. Some of these
  also use ComfyUI's `model.diffusion_model.` key prefix and need stripping
  before diffusers can ingest them.
"""
from __future__ import annotations
import logging
from typing import Any, ClassVar, Optional

from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)


class ZImageBackend(PipelineBackend):
    """Backend for Tongyi Z-Image checkpoints."""

    engine: ClassVar[str] = "zimage"

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        # No scheduler cache: Z-Image only uses FlowMatchEulerDiscreteScheduler,
        # which we re-instantiate cheaply with the requested `shift` per request.
        # Flat offline tree root + optional CDN mirror, threaded from RuntimeConfig.
        # Consumed by the offline base-component loading (the base_model path).
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import ZImagePipeline

        if base_model:
            return self._load_with_base_model(local_path, base_model)

        logger.info(f"Loading Z-Image pipeline from {local_path}")
        return ZImagePipeline.from_single_file(
            local_path,
            torch_dtype=torch.bfloat16,
        )

    def _load_with_base_model(self, local_path: str, base_repo: str) -> Any:
        """Load transformer from local_path + shared components from base HF repo."""
        import torch
        from diffusers import AutoencoderKL, ZImagePipeline
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading Z-Image with shared components from base: {base_repo}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_repo, subfolder="tokenizer", torch_dtype=torch.bfloat16
        )
        text_encoder = AutoModel.from_pretrained(
            base_repo, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKL.from_pretrained(
            base_repo, subfolder="vae", torch_dtype=torch.bfloat16
        )

        local_path = self._strip_comfyui_prefix(local_path)

        return ZImagePipeline.from_single_file(
            local_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )

    @staticmethod
    def _strip_comfyui_prefix(filepath: str) -> str:
        """Strip the `model.diffusion_model.` key prefix from a safetensors file.

        WARNING: rewrites the file in place. Acceptable for worker pods that own
        their model cache, but problematic for desktop users — a future refactor
        should strip in memory and pass the stripped state_dict directly to
        diffusers (need to bypass from_single_file).
        """
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file

        with safe_open(filepath, framework="pt") as f:
            first_key = next(iter(f.keys()))
        if not first_key.startswith("model.diffusion_model."):
            return filepath

        logger.info(f"Stripping ComfyUI 'model.diffusion_model.' prefix in {filepath}")
        state_dict = load_file(filepath)
        new_dict = {
            (k.replace("model.diffusion_model.", "", 1)
             if k.startswith("model.diffusion_model.") else k): v
            for k, v in state_dict.items()
        }
        save_file(new_dict, filepath)
        logger.info(f"Stripped {len(new_dict)} keys")
        return filepath

    # ------------------------------------------------------------------
    # Img2img
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        from diffusers import ZImageImg2ImgPipeline
        return ZImageImg2ImgPipeline(
            transformer=t2i_pipe.transformer,
            text_encoder=t2i_pipe.text_encoder,
            tokenizer=t2i_pipe.tokenizer,
            vae=t2i_pipe.vae,
            scheduler=t2i_pipe.scheduler,
        )

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        # Z-Image's tokenizer doesn't have the 77-token limit problem that
        # SDXL has, so we pass raw strings. No weighted embeddings / BREAK.
        return {"prompt": prompt, "negative_prompt": negative_prompt or ""}

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any
    ) -> None:
        """Configure FlowMatchEulerDiscreteScheduler with the given shift.

        The `scheduler` name arg is ignored — Z-Image only uses flow matching.
        Pass `shift` via backend_options to control sampler behavior:
            engine.generate(..., backend_options={"shift": 3.0})

        Turbo checkpoints (4-8 step) typically want shift=3.0. The worker
        auto-detects this from the base_model name; the engine does not yet
        — callers pass shift explicitly. Auto-detection lands with the
        catalog YAML loader so model presets can carry sampler defaults.
        """
        shift = kwargs.get("shift")
        if shift is None:
            return  # leave the pipe's default scheduler

        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=float(shift),
        )
        logger.info(f"Z-Image scheduler: FlowMatchEuler shift={shift}")

    def build_inference_kwargs(
        self,
        *,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],  # noqa: ARG002 — Z-Image has no clip_skip
        chunk_size: int,
        generator: Any,
    ) -> dict:
        return {
            "width": width,
            "height": height,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
        }

    def make_generator(self, seed: int, device: str) -> Any:
        import torch
        # Z-Image flow matching prefers the generator on the same device as
        # the transformer. On CPU this is a CPU generator; on CUDA, a CUDA one.
        # MPS Generator works on PyTorch >= 2.1 but is occasionally flaky;
        # callers on MPS may need to swap to "cpu" if they hit issues.
        try:
            return torch.Generator(device=device).manual_seed(seed)
        except (RuntimeError, ValueError) as e:
            logger.warning(
                f"Generator on device={device} failed ({e}); falling back to CPU"
            )
            return torch.Generator().manual_seed(seed)
