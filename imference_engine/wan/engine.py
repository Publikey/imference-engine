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
import os
from typing import TYPE_CHECKING, Optional

from imference_engine.core.engine_base import BaseEngine
from imference_engine.core.result import GenerationError, MediaResult
from imference_engine.wan.config import MemoryProfile, WanRuntimeConfig
from imference_engine.wan.manager import ResidencyManager
from imference_engine.wan.presets import BUILTIN_VARIANTS, WanVariant

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

# Profiles ordered by RAM/VRAM footprint (higher = heavier) for clamping.
_RANK = {MemoryProfile.GGUF_Q4: 1, MemoryProfile.GGUF_Q5: 2,
         MemoryProfile.GGUF_Q6: 3, MemoryProfile.GGUF_Q8: 4,
         MemoryProfile.BF16: 5}


def _clamp_profile(by_vram: MemoryProfile, by_ram: MemoryProfile) -> MemoryProfile:
    """Pick the lighter-footprint of the VRAM- and RAM-based choices."""
    return by_vram if _RANK[by_vram] <= _RANK[by_ram] else by_ram


def _total_ram_gb() -> float:
    """Best-effort total CPU RAM in GiB — cgroup limit (container) else physical.

    Mirrors the worker's cgroup-aware sizing so the auto profile sees the *real*
    memory the process is allowed, not the host's. Unknown -> inf (don't clamp)."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw == "max":
            break
        try:
            val = int(raw) / 1024 ** 3
        except ValueError:
            break
        # cgroup v1 "no limit" is a huge sentinel; ignore implausible values.
        if 0 < val < 1024 * 1024:
            return val
        break
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError):
        return float("inf")


class WanEngine(BaseEngine):
    def __init__(self, *, runtime: Optional[WanRuntimeConfig] = None) -> None:
        super().__init__(runtime=runtime or WanRuntimeConfig())
        self._manager: Optional[ResidencyManager] = None
        self._variants: dict[str, WanVariant] = dict(BUILTIN_VARIANTS)

    # ------------------------------------------------------------------
    def _setup(self) -> None:
        """Resolve the GGUF profile + build the ResidencyManager. Device
        resolution / CUDA tuning already ran in ``BaseEngine.load()``."""
        if self._device.kind != "cuda":
            logger.warning(
                "WanEngine on %s — Wan is validated on CUDA only; expect slowness/failure.",
                self._device.torch_str)

        profile = self._runtime.memory_profile
        # MemoryProfile subclasses str, so a concrete profile (e.g. the default
        # GGUF_Q8) IS a str — match it FIRST and use it as-is. Only a PLAIN string
        # reaches the "auto" branch; anything else there is an error.
        if isinstance(profile, MemoryProfile):
            pass
        elif isinstance(profile, str):
            if profile.lower() != "auto":
                raise ValueError(f"memory_profile string must be 'auto', got {profile!r}")
            profile = self._auto_profile(self._device.kind == "cuda")
        if not profile.is_gguf:
            raise NotImplementedError(
                f"memory_profile={profile} not wired yet; only GGUF profiles are "
                "implemented (bf16 path is a follow-up).")

        cache_dir = (str(self._runtime.model_cache_dir)
                     if self._runtime.model_cache_dir else None)
        self._manager = ResidencyManager(
            quant=profile.gguf_quant,
            device=self._device.torch_str,
            enable_offload=self._runtime.enable_offload,
            vae_tiling=self._runtime.vae_tiling,
            max_resident=self._runtime.max_resident_variants or 1,
            cache_dir=cache_dir,
            cdn_base=self._runtime.model_cdn,
            text_encoder_quant=self._runtime.text_encoder_quant,
        )

    @staticmethod
    def _auto_profile(is_cuda: bool) -> MemoryProfile:
        if not is_cuda:
            return MemoryProfile.GGUF_Q4
        import torch
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        ram = _total_ram_gb()
        # Clamp to the lighter of the VRAM- and RAM-based picks: a big-VRAM/
        # small-RAM box would OOM in CPU RAM on the VRAM-only pick (e.g. 24 GB
        # GPU / 50 GB RAM -> Q8 OOM).
        profile = _clamp_profile(MemoryProfile.for_vram(vram),
                                 MemoryProfile.for_ram(ram))
        logger.info("auto profile: %.0f GB VRAM / %.0f GB RAM -> %s",
                    vram, ram, profile.value)
        return profile

    def register_variant(self, variant: WanVariant) -> None:
        """Add or override a variant. Call before generate_video(variant=name)."""
        self._variants[variant.name] = variant

    def list_variants(self) -> list[str]:
        return sorted(self._variants)

    def warm(self) -> "WanEngine":
        """Pre-download the shared base-components (UMT5 / VAE, ~11.5 GB) + the
        registered variants' base configs into the offline tree WITHOUT loading
        anything — so a fresh pod is 'ready' with the base on disk and the first
        request only builds the variant. GGUF experts + LoRAs stay lazy.
        Best-effort: a failure logs a warning and is skipped (lazy fallback);
        warm() never raises, so it's safe to call in a worker's setup()."""
        if not self._loaded:
            self.load()
        from imference_engine.wan.loader import (_BASE_CFG_PATTERNS,
                                                 _SHARED_PATTERNS, _local_repo_dir)
        cache_dir = (str(self._runtime.model_cache_dir)
                     if self._runtime.model_cache_dir else None)
        cdn = self._runtime.model_cdn
        # The shared UMT5/VAE base (heavy) + each distinct variant base_repo
        # (small configs for the GGUF transformer). Dedup so the shared base
        # isn't pulled twice.
        targets = [(self._manager._shared_base_repo, _SHARED_PATTERNS)]
        seen = {self._manager._shared_base_repo}
        for v in self._variants.values():
            if v.base_repo not in seen:
                seen.add(v.base_repo)
                targets.append((v.base_repo, _BASE_CFG_PATTERNS))
        for repo, patterns in targets:
            try:
                _local_repo_dir(repo, patterns, cache_dir, cdn)
                logger.info("Wan base-components warmed (%s)", repo)
            except Exception as e:  # noqa: BLE001
                logger.warning("warm: %s failed (%s); will fetch lazily", repo, e)
        return self

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
    ) -> MediaResult:
        if not self._loaded:
            raise RuntimeError("Call WanEngine.load() before generate_video")
        if variant not in self._variants:
            raise KeyError(f"unknown variant {variant!r}; known: {self.list_variants()}")
        var = self._variants[variant]
        if var.mode == "i2v" and image is None:
            raise ValueError(f"variant {variant!r} is i2v — an input image is required")

        seed = self._make_seed(seed)

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
            return MediaResult(
                kind="video", media=frames, seeds=[seed],
                fps=fps, num_frames=num_frames, width=width, height=height,
                variant=variant)
        except Exception as e:  # noqa: BLE001
            logger.error("generate_video failed: %s", e, exc_info=True)
            return MediaResult(
                kind="video", media=[], seeds=[seed],
                errors=[GenerationError(error=str(e), seed=seed)],
                fps=fps, num_frames=num_frames, width=width, height=height,
                variant=variant)

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
