"""Z-Image backend — Tongyi Z-Image pipelines (text-to-image, flow-matching).

Split out of ``pipelines/`` into its own ``imference_engine.zimage`` sub-package
(parallel to ``imference_engine.wan``) so Z-Image is a self-contained unit. It
still rides the generic ``Engine``/``ModelManager``/``RuntimeConfig`` machinery
(same diffusers 0.38 stack as SDXL), so the split is a packaging boundary, not a
fork of the engine core. ``pipelines/zimage.py`` re-exports this class for
backward compatibility.

Z-Image checkpoints come in two flavors:
- *Full* base/turbo: a self-contained safetensors with all components → just
  call from_single_file.
- *Finetunes / ComfyUI-style*: a transformer-only safetensors that needs
  tokenizer + text_encoder + VAE loaded from a base repo. Some of these also
  use ComfyUI's `model.diffusion_model.` key prefix and need stripping before
  diffusers can ingest them.

HuggingFace-independence: the base components are resolved into the flat,
symlink-free offline tree (``IMAGE_MODEL_CACHE``, ``namespace="image"``) and
loaded with ``local_files_only=True``, so a cold load never contacts HF once the
tree is populated (prefetch / base tarball) under ``HF_HUB_OFFLINE=1``.
"""
from __future__ import annotations
import logging
from typing import Any, ClassVar, Optional

from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)


class ZImageBackend(PipelineBackend):
    """Backend for Tongyi Z-Image checkpoints."""

    engine: ClassVar[str] = "zimage"

    # Files needed from a Z-Image base repo (Tongyi-MAI/Z-Image[-Turbo]) for the
    # base_model path. The shared tokenizer + (Qwen-family) text_encoder + VAE are
    # FULL weights. The transformer + scheduler are CONFIG-ONLY: from_single_file
    # reads their layout from here (config=base_dir) while the transformer WEIGHTS
    # come from the checkpoint. Omitting transformer/config.json made
    # from_single_file fall back to a HF snapshot_download (crashes offline).
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "text_encoder/*", "vae/*",
        "transformer/config.json",
    ]

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        # No scheduler cache: Z-Image only uses FlowMatchEulerDiscreteScheduler,
        # which we re-instantiate cheaply with the requested `shift` per request.
        # Flat offline tree root + optional CDN mirror, threaded from RuntimeConfig.
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

        # Self-contained checkpoint. NOTE: this path still infers its config from
        # HF; every live catalog row sets base_model (the path above), which is the
        # offline-correct one. If a full self-contained Z-Image is ever served
        # offline, mirror its config repo and pass config=<dir> here too.
        logger.info(f"Loading Z-Image pipeline from {local_path}")
        return ZImagePipeline.from_single_file(
            local_path,
            torch_dtype=torch.bfloat16,
        )

    def _load_with_base_model(self, local_path: str, base_repo: str) -> Any:
        """Load transformer from local_path + shared components from the base repo,
        resolved into the flat offline tree (no HF hit once populated)."""
        import torch
        from diffusers import AutoencoderKL, ZImagePipeline
        from transformers import AutoModel, AutoTokenizer

        from imference_engine.runtime.offline import local_repo_dir

        base_dir = local_repo_dir(
            base_repo, self.BASE_PATTERNS, self._cache_dir, namespace="image",
            cdn_base=self._cdn_base)
        logger.info(f"Loading Z-Image shared components from {base_dir} (base={base_repo})")
        tokenizer = AutoTokenizer.from_pretrained(
            base_dir, subfolder="tokenizer", local_files_only=True
        )
        text_encoder = AutoModel.from_pretrained(
            base_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        vae = AutoencoderKL.from_pretrained(
            base_dir, subfolder="vae", torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        local_path = self._strip_comfyui_prefix(local_path)

        # config=base_dir + local_files_only pins the pipeline layout (model_index
        # + transformer/scheduler configs) to the mirrored tree. Without it,
        # from_single_file downloads the reference config from HF -> offline crash.
        # The passed text_encoder/tokenizer/vae override those components.
        return ZImagePipeline.from_single_file(
            local_path,
            config=base_dir,
            local_files_only=True,
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

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        """Pull a base_model's shared components (tokenizer + text_encoder + vae)
        into the offline tree (no load) — same dir _load_with_base_model resolves.
        No-op without a base_model (a self-contained checkpoint has none)."""
        if not base_model:
            return
        from imference_engine.runtime.offline import local_repo_dir
        local_repo_dir(base_model, self.BASE_PATTERNS, self._cache_dir,
                       namespace="image", cdn_base=self._cdn_base)
        logger.info("Z-Image base-components warmed (base=%s)", base_model)

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
        image: Any = None,
        strength: float = 0.75,
    ) -> dict:
        kwargs = {
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_images_per_prompt": chunk_size,
        }
        if image is not None:
            # img2img: ZImageImg2ImgPipeline takes the size from the source image.
            kwargs["image"] = image
            kwargs["strength"] = strength
        else:
            kwargs["width"] = width
            kwargs["height"] = height
        return kwargs

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
