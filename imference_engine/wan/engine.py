"""WanEngine — public entry point for Wan2.2 video generation.

Transport-agnostic, like the image-side ``Engine``: returns frames + metadata,
the caller does ``export_to_video`` + upload. Single-threaded and stateful;
concurrent callers must serialize.

    engine = WanEngine(runtime=WanRuntimeConfig(...)).load()
    engine.register_variant(WanVariant(...))            # or use built-ins
    res = engine.generate_video(variant="wan22-t2v-lightning", prompt="...")
    # res.frames -> list[PIL.Image]
"""
from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Optional

from imference_engine.runtime.device import resolve_device
from imference_engine.wan.config import MemoryProfile, WanRuntimeConfig
from imference_engine.wan.manager import ResidencyManager
from imference_engine.wan.presets import BUILTIN_VARIANTS, WanVariant
from imference_engine.wan.result import WanGenerationError, WanVideoResult

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

MAX_SEED = 2 ** 31 - 1


class WanEngine:
    def __init__(self, *, runtime: Optional[WanRuntimeConfig] = None) -> None:
        self._runtime = runtime or WanRuntimeConfig()
        self._device: Optional[str] = None
        self._manager: Optional[ResidencyManager] = None
        self._variants: dict[str, WanVariant] = dict(BUILTIN_VARIANTS)
        self._loaded = False

    # ------------------------------------------------------------------
    def load(self) -> "WanEngine":
        if self._loaded:
            return self
        dev = resolve_device(self._runtime.device)
        self._device = dev.torch_str
        logger.info("WanEngine device: %s", self._device)
        if dev.kind != "cuda":
            logger.warning(
                "WanEngine on %s — Wan is validated on CUDA only; expect slowness/failure.",
                self._device)

        profile = self._runtime.memory_profile
        if isinstance(profile, str):
            if profile.lower() != "auto":
                raise ValueError(f"memory_profile string must be 'auto', got {profile!r}")
            profile = self._auto_profile(dev.kind == "cuda")
        if not profile.is_gguf:
            raise NotImplementedError(
                f"memory_profile={profile} not wired yet; only GGUF profiles are "
                "implemented (bf16 path is a follow-up).")

        cache_dir = (str(self._runtime.model_cache_dir)
                     if self._runtime.model_cache_dir else None)
        self._manager = ResidencyManager(
            quant=profile.gguf_quant,
            device=self._device,
            enable_offload=self._runtime.enable_offload,
            vae_tiling=self._runtime.vae_tiling,
            max_resident=self._runtime.max_resident_variants or 1,
            cache_dir=cache_dir,
        )
        self._loaded = True
        return self

    @staticmethod
    def _auto_profile(is_cuda: bool) -> MemoryProfile:
        if not is_cuda:
            return MemoryProfile.GGUF_Q4
        import torch
        gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        profile = MemoryProfile.for_vram(gb)
        logger.info("auto profile: %.0f GB VRAM -> %s", gb, profile.value)
        return profile

    def register_variant(self, variant: WanVariant) -> None:
        """Add or override a variant. Call before generate_video(variant=name)."""
        self._variants[variant.name] = variant

    def list_variants(self) -> list[str]:
        return sorted(self._variants)

    # ------------------------------------------------------------------
    def generate_video(
        self,
        *,
        variant: str,
        prompt: str,
        image: Optional["Image"] = None,
        negative_prompt: Optional[str] = None,
        width: int = 832,
        height: int = 480,
        num_frames: int = 81,
        num_steps: int = 4,
        guidance_scale: float = 1.0,
        guidance_scale_2: Optional[float] = None,
        fps: int = 16,
        seed: Optional[int] = None,
    ) -> WanVideoResult:
        if not self._loaded:
            raise RuntimeError("Call WanEngine.load() before generate_video")
        if variant not in self._variants:
            raise KeyError(f"unknown variant {variant!r}; known: {self.list_variants()}")
        var = self._variants[variant]
        if var.mode == "i2v" and image is None:
            raise ValueError(f"variant {variant!r} is i2v — an input image is required")

        seed = seed if seed is not None else random.randint(0, MAX_SEED)

        try:
            import torch
            pipe = self._manager.get_or_load(var)

            call = dict(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                guidance_scale_2=(guidance_scale_2 if guidance_scale_2 is not None
                                  else guidance_scale),
                generator=torch.Generator("cpu").manual_seed(seed),
                output_type="pil",
            )
            if var.mode == "i2v":
                call["image"] = image.convert("RGB").resize((width, height))

            logger.info("generate_video variant=%s mode=%s %dx%d frames=%d steps=%d seed=%d",
                        variant, var.mode, width, height, num_frames, num_steps, seed)
            frames = pipe(**call).frames[0]
            frames = _ensure_pil(frames)
            return WanVideoResult(
                frames=frames, fps=fps, seed=seed, variant=variant,
                width=width, height=height, num_frames=num_frames)
        except Exception as e:  # noqa: BLE001
            logger.error("generate_video failed: %s", e, exc_info=True)
            return WanVideoResult(
                frames=None, fps=fps, seed=seed, variant=variant,
                width=width, height=height, num_frames=num_frames,
                error=WanGenerationError(error=str(e), seed=seed))

    @property
    def resident(self) -> list[str]:
        return self._manager.resident if self._manager else []


def _ensure_pil(frames) -> list["Image"]:
    """Wan may return numpy arrays even with output_type='pil' on some versions."""
    from PIL import Image
    out = []
    for fr in frames:
        if isinstance(fr, Image.Image):
            out.append(fr)
            continue
        import numpy as np
        arr = np.asarray(fr)
        if arr.dtype != np.uint8:
            arr = (arr.clip(0, 1) * 255).round().astype(np.uint8)
        out.append(Image.fromarray(arr))
    return out
