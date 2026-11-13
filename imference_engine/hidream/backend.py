"""HiDream-I1 backend — a 17B flow-matching DiT with four text encoders.

Own sub-package parallel to ``imference_engine.ernieimage`` / ``.longcat``.
HiDream-I1 is a sparse (mixture-of-experts) DiT conditioned on **CLIP-L**,
**CLIP-G**, **T5-XXL** and **Llama-3.1-8B-Instruct** at once, with the FLUX
VAE. The model is MIT; the Llama encoder is under the Llama 3.1 Community
License ("Built with Llama") and T5 under Apache-2.0. ``HiDreamImagePipeline``
is in diffusers 0.40.

Three checkpoints, told apart by their base repo's scheduler and recipe:

- **Fast** (``HiDream-ai/HiDream-I1-Fast``, the default base) — 16 steps,
  guidance 0.0;
- **Dev** (``…-I1-Dev``) — 28 steps, guidance 0.0;
- **Full** (``…-I1-Full``) — 50 steps, true CFG 5.0, negative prompt honored.

Register a Dev / Full transformer with ``base_model=`` its own repo (the
scheduler differs); the step count and guidance follow the file name
(``weights_defaults``). Only Fast has been run on GPU.

Text-to-image only (HiDream-E1 editing is another pipeline, not wired). LoRAs:
the pipeline carries a loader, not enabled until validated.

Weights (``weights_path``): ``"auto"`` (the GGUF picked for this machine by
``builds.recommend_build``), a ``.gguf`` (city96 ``HiDream-I1-*-gguf``), a
transformer-only ``.safetensors`` (Comfy-Org repackage), or a diffusers repo.
The bf16 transformer is 34 GB, so GGUF is the practical form.

**The Llama encoder is not in the HiDream repos.** It is loaded from
``HIDREAM_LLAMA_REPO`` (default ``unsloth/Meta-Llama-3.1-8B-Instruct``, an
ungated copy; the CDN mirror serves it with the licence texts attached).

Memory: the four encoders (16 GB + 9.5 GB + 3.3 GB in bf16) never fit next to
the transformer, so every build runs with offload; the encoders are used once
per request. ``HIDREAM_ENCODER`` = auto | bf16 | fp8 stores Llama and T5 in fp8
(float8 storage, bf16 compute: 8.5 GB + 4.8 GB).
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.hidream.builds import AUTO, recommend_build
from imference_engine.pipelines.base import PipelineBackend

logger = logging.getLogger(__name__)

DEFAULT_BASE = "HiDream-ai/HiDream-I1-Fast"
DEFAULT_LLAMA = "unsloth/Meta-Llama-3.1-8B-Instruct"


def llama_repo() -> str:
    """``HIDREAM_LLAMA_REPO``: repo id or directory of the Llama-3.1-8B-Instruct
    encoder (default: the ungated unsloth copy)."""
    return os.environ.get("HIDREAM_LLAMA_REPO", "").strip() or DEFAULT_LLAMA


def _vram_gb() -> Optional[float]:
    from imference_engine.managers.builds import cuda_vram_gb
    return cuda_vram_gb()


class HiDreamBackend(PipelineBackend):
    """Backend for HiDream-I1 checkpoints."""

    engine: ClassVar[str] = "hidream"

    # Shared components from the base repo: CLIP-L, CLIP-G, T5-XXL, their
    # tokenizers, VAE and scheduler. The transformer is CONFIG-ONLY; the Llama
    # encoder lives in its own repo (LLAMA_PATTERNS).
    BASE_PATTERNS: ClassVar[list] = [
        "model_index.json", "scheduler/*",
        "tokenizer/*", "tokenizer_2/*", "tokenizer_3/*",
        "text_encoder/*", "text_encoder_2/*", "text_encoder_3/*",
        "vae/*",
        "transformer/config.json",
    ]
    REPO_PATTERNS: ClassVar[list] = [*BASE_PATTERNS, "transformer/*"]
    LLAMA_PATTERNS: ClassVar[list] = [
        "config.json", "generation_config.json", "model-*.safetensors",
        "model.safetensors.index.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json",
        # Llama 3.1 Community License + Acceptable Use Policy (mirror only).
        "LICENSE*", "USE_POLICY*", "NOTICE*",
    ]

    def __init__(
        self, *, cache_dir: Optional[str] = None, cdn_base: Optional[str] = None
    ) -> None:
        self._cache_dir = cache_dir
        self._cdn_base = cdn_base

    def engine_defaults(self) -> GenerationDefaults:
        # Fast, the default checkpoint (distilled, no CFG).
        return GenerationDefaults(guidance_scale=0.0, num_steps=16)

    def weights_defaults(self, weights_path: str) -> Optional[GenerationDefaults]:
        """Dev and Full files carry their own recipe; Fast (and ``"auto"``)
        keep the engine defaults."""
        name = os.path.basename(weights_path).lower()
        if "full" in name:
            return GenerationDefaults(guidance_scale=5.0, num_steps=50)
        if "dev" in name:
            return GenerationDefaults(guidance_scale=0.0, num_steps=28)
        return None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def resolve_weights(self, weights_path: str) -> str:
        """``"auto"`` -> the local path of the GGUF recommended for this
        machine (downloaded on first use); anything else is returned as is."""
        if weights_path != AUTO:
            return weights_path
        from imference_engine.runtime.offline import fetch

        build = recommend_build(_vram_gb())
        logger.info("HiDream-I1 auto build: %s (%s)", build.name, build.reason)
        return fetch(build.repo, build.filename, self._cdn_base, self._cache_dir, namespace="image")

    def _repo_dir(self, repo: str, patterns: list, sentinel: str = "model_index.json") -> str:
        from imference_engine.runtime.offline import local_repo_dir

        if os.path.isdir(repo):
            return repo
        return local_repo_dir(repo, patterns, self._cache_dir, namespace="image",
                              sentinel=sentinel, cdn_base=self._cdn_base)

    def load_pipeline(
        self, *, local_path: str, base_model: Optional[str] = None
    ) -> Any:
        import torch
        from diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
        from transformers import AutoTokenizer, LlamaForCausalLM, T5EncoderModel

        from imference_engine.core.fp8 import apply_fp8_storage, resolve_storage

        base_repo = base_model or DEFAULT_BASE
        local_path = self.resolve_weights(local_path)
        lower = local_path.lower()
        is_file = not os.path.isdir(local_path)
        vram = _vram_gb()
        build = recommend_build(vram)

        if is_file and lower.endswith((".gguf", ".safetensors")):
            base_dir = self._repo_dir(base_repo, self.BASE_PATTERNS)
            cfg_dir = os.path.join(base_dir, "transformer")
            if lower.endswith(".gguf"):
                transformer = self._load_gguf(local_path, cfg_dir)
            else:
                logger.info(f"Loading HiDream-I1 transformer from {local_path}")
                transformer = HiDreamImageTransformer2DModel.from_single_file(
                    local_path, config=cfg_dir, local_files_only=True,
                    torch_dtype=torch.bfloat16)
        else:
            base_dir = local_path if not is_file else self._repo_dir(local_path, self.REPO_PATTERNS)
            logger.info(f"Loading HiDream-I1 pipeline from {base_dir}")
            transformer = HiDreamImageTransformer2DModel.from_pretrained(
                base_dir, subfolder="transformer", local_files_only=True,
                torch_dtype=torch.bfloat16)
        logger.info(f"Loading HiDream-I1 shared components from {base_dir} (base={base_repo})")

        llama_dir = self._repo_dir(llama_repo(), self.LLAMA_PATTERNS, sentinel="config.json")
        logger.info(f"Loading the Llama-3.1-8B encoder from {llama_dir}")
        # The pipeline reads the hidden states of every Llama layer.
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            llama_dir, local_files_only=True, dtype=torch.bfloat16,
            output_hidden_states=True, output_attentions=True)
        components: dict = {
            "transformer": transformer,
            "text_encoder_4": text_encoder_4,
            "tokenizer_4": AutoTokenizer.from_pretrained(llama_dir, local_files_only=True),
        }
        if resolve_storage("HIDREAM_ENCODER", build.encoder, vram) == "fp8":
            apply_fp8_storage(text_encoder_4)
            t5 = T5EncoderModel.from_pretrained(
                os.path.join(base_dir, "text_encoder_3"), local_files_only=True,
                dtype=torch.bfloat16)
            components["text_encoder_3"] = apply_fp8_storage(t5, keep=("shared", "norm"))
            logger.info("HiDream-I1 Llama and T5 encoders: fp8-resident storage enabled")
        return HiDreamImagePipeline.from_pretrained(
            base_dir, local_files_only=True, dtype=torch.bfloat16, **components)

    def _load_gguf(self, path: str, cfg_dir: str) -> Any:
        import torch
        from diffusers import GGUFQuantizationConfig, HiDreamImageTransformer2DModel

        from imference_engine.core.gguf import dequantize_non_linear

        logger.info("Loading HiDream-I1 GGUF transformer %s", os.path.basename(path))
        transformer = HiDreamImageTransformer2DModel.from_single_file(
            path, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=cfg_dir, local_files_only=True, torch_dtype=torch.bfloat16)
        converted = dequantize_non_linear(transformer, torch.bfloat16)
        if converted:
            logger.info("HiDream-I1 GGUF: restored non-linear weights %s", converted)
        return transformer

    def prefetch_base(self, base_model: Optional[str] = None) -> None:
        self._repo_dir(base_model or DEFAULT_BASE, self.BASE_PATTERNS)
        self._repo_dir(llama_repo(), self.LLAMA_PATTERNS, sentinel="config.json")
        logger.info("HiDream-I1 base-components warmed (base=%s, llama=%s)",
                    base_model or DEFAULT_BASE, llama_repo())

    # ------------------------------------------------------------------
    # Img2img — unsupported (the editing model, HiDream-E1, is another pipeline)
    # ------------------------------------------------------------------

    def make_img2img(self, t2i_pipe: Any) -> Any:
        raise NotImplementedError(
            "HiDream-I1 has no img2img pipeline (text-to-image only; the HiDream-E1 "
            "editing model is not wired). Call generate() without source_image."
        )

    def get_compute_module(self, pipe: Any) -> Any:
        return pipe.transformer

    # ------------------------------------------------------------------
    # Prompts / scheduler / inference kwargs
    # ------------------------------------------------------------------

    def encode_prompts(
        self, pipe: Any, prompt: str, negative_prompt: Optional[str]
    ) -> dict:
        # One prompt for the four encoders (the pipeline fans it out). The
        # negative only acts when guidance_scale > 1 (Full).
        out = {"prompt": prompt}
        if negative_prompt:
            out["negative_prompt"] = negative_prompt
        return out

    def apply_scheduler(
        self, pipe: Any, scheduler: Optional[str], **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Each checkpoint's base repo carries its own scheduler (Fast:
        FlowMatchLCM, shift 3); the ``scheduler`` name is ignored."""
        if scheduler:
            logger.debug("HiDream-I1 ignores scheduler=%r (set by the base repo)", scheduler)

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
