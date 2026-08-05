"""MiniMaxH3Engine — public entry point for MiniMax-H3 joint video+audio.

Transport-agnostic, like ``WanEngine``: returns PIL frames + a raw stereo
waveform + metadata; the caller muxes (e.g. diffusers' ``encode_video(frames,
fps=24, audio=torch.from_numpy(res.audio), audio_sample_rate=res.sample_rate)``
— it clips the waveform with torch ops, so tensor in) and uploads. Single-threaded and stateful; concurrent callers must serialize.

    engine = MiniMaxH3Engine(runtime=MiniMaxH3RuntimeConfig(...)).load()
    res = engine.generate_video(prompt="a red fox trotting through snow")
    # res.frames -> list[PIL.Image]; res.audio -> (2, n) float numpy; res.sample_rate

One variant serves t2v and i2v: pass ``image`` (start keyframe) and/or
``last_image`` (end keyframe) to condition, nothing to go text-only. There is no
negative prompt and no guidance scale — the checkpoint is guidance-distilled.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from imference_engine.core.engine_base import BaseEngine
from imference_engine.core.result import GenerationError, MediaResult
from imference_engine.minimax_h3.config import (H3MemoryProfile,
                                                MiniMaxH3RuntimeConfig,
                                                OFFLOAD_MODES)
from imference_engine.minimax_h3.constraints import (MINIMAX_H3_FPS,
                                                     check_canvas,
                                                     check_num_frames)
from imference_engine.minimax_h3.presets import BUILTIN_VARIANTS, H3Variant
from imference_engine.video.backend import VideoBackend, VideoBuildContext
from imference_engine.video.backends.minimax_h3 import MiniMaxH3Backend
from imference_engine.video.residency import ResidencyManager

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

# Host RAM the int8 weights want under "block"/"leaf" offload (upstream figure).
_INT8_HOST_RAM_GB = 75


class MiniMaxH3Engine(BaseEngine):
    def __init__(
        self,
        *,
        catalog_path: Optional[Union[str, Path]] = None,
        runtime: Optional[MiniMaxH3RuntimeConfig] = None,
    ) -> None:
        super().__init__(runtime=runtime or MiniMaxH3RuntimeConfig())
        self._catalog_path = Path(catalog_path) if catalog_path else None
        self._variants: dict[str, H3Variant] = dict(BUILTIN_VARIANTS)
        self._backends: dict[str, VideoBackend] = {}
        self._managers: dict[str, ResidencyManager] = {}
        self._ctx: Optional[VideoBuildContext] = None

    # ------------------------------------------------------------------
    def _setup(self) -> None:
        if self._device.kind != "cuda":
            logger.warning(
                "MiniMaxH3Engine on %s — H3 is CUDA-only in practice (33B DiT + "
                "32B conditioner); expect failure.", self._device.torch_str)

        profile = self._resolve_profile(self._runtime.memory_profile)
        offload_mode = self._resolve_offload(self._runtime.offload_mode)
        self._runtime.enable_offload = offload_mode != "none"  # base-class coherence
        self._warn_host_ram(profile, offload_mode)

        cache_dir = (str(self._runtime.model_cache_dir)
                     if self._runtime.model_cache_dir else None)
        self._ctx = VideoBuildContext(
            quant=profile.value,
            device=self._device.torch_str,
            enable_offload=offload_mode != "none",
            vae_tiling=self._runtime.vae_tiling,
            cache_dir=cache_dir,
            cdn_base=self._runtime.model_cdn,
            offload_mode=offload_mode,
            attention_backend=self._runtime.attention_backend,
        )
        self._backends = {MiniMaxH3Backend.engine: MiniMaxH3Backend()}
        if self._catalog_path is not None:
            self._register_catalog(self._catalog_path)

    @staticmethod
    def _resolve_profile(profile) -> H3MemoryProfile:
        if isinstance(profile, H3MemoryProfile):
            return profile  # MemoryProfile subclasses str — match it first
        if isinstance(profile, str):
            if profile.lower() == "auto":
                return H3MemoryProfile.for_hardware(0.0)
            return H3MemoryProfile(profile.lower())  # raises on unknown strings
        raise ValueError(f"memory_profile must be H3MemoryProfile|str, got {profile!r}")

    def _resolve_offload(self, mode: str) -> str:
        if mode in OFFLOAD_MODES:
            return mode
        if mode != "auto":
            raise ValueError(
                f"offload_mode must be one of {OFFLOAD_MODES} or 'auto', got {mode!r}")
        if self._device.kind != "cuda":
            return "block"
        import torch
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if vram >= 80:
            resolved = "none"
        elif vram >= 20:
            resolved = "block"
        else:
            resolved = "leaf"
        logger.info("auto offload: %.0f GB VRAM -> %s", vram, resolved)
        return resolved

    @staticmethod
    def _warn_host_ram(profile: H3MemoryProfile, offload_mode: str) -> None:
        """Offloaded weights live in host RAM (~75 GB at int8) — warn early on a
        small-RAM box instead of OOMing 40 minutes into a cold load."""
        if offload_mode == "none":
            return
        try:
            import os
            ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
        except (ValueError, OSError, AttributeError):
            return
        if ram_gb < _INT8_HOST_RAM_GB - 15:  # margin: swap/compression may save a box
            logger.warning(
                "MiniMax-H3 %s weights under %s offload want ~%d GB host RAM; this "
                "box has ~%.0f GB — expect OOM or heavy swapping.",
                profile.value, offload_mode, _INT8_HOST_RAM_GB, ram_gb)

    # ------------------------------------------------------------------
    def load_catalog(self, path: Optional[Union[str, Path]] = None) -> "MiniMaxH3Engine":
        """Load ``kind: video`` rows with ``engine: minimax_h3`` from a (possibly
        shared) ``models.yml`` and register them as variants (add/override by
        name). ``path`` defaults to the constructor's ``catalog_path``."""
        if not self._loaded:
            raise RuntimeError("Call MiniMaxH3Engine.load() before load_catalog")
        target = path if path is not None else self._catalog_path
        if target is None:
            raise ValueError("no catalog path given and MiniMaxH3Engine has no catalog_path")
        self._register_catalog(target)
        return self

    def _register_catalog(self, path) -> None:
        from imference_engine.catalog.loader import load_video
        from imference_engine.minimax_h3.presets import variant_from_catalog
        from imference_engine.video import KNOWN_VIDEO_ARCHS
        # Validate rows against every known video arch (typos fail loudly), keep
        # only ours — a shared models.yml may also carry wan rows.
        configs = [c for c in load_video(path, known_archs=KNOWN_VIDEO_ARCHS)
                   if c.arch in self._backends]
        for cfg in configs:
            self.register_variant(variant_from_catalog(cfg))
        logger.info("Loaded video catalog %s (%d minimax_h3 variants)", path, len(configs))

    def _manager_for(self, arch: str) -> ResidencyManager:
        if arch not in self._managers:
            backend = self._backends.get(arch)
            if backend is None:
                raise KeyError(
                    f"no video backend for arch {arch!r}; have {list(self._backends)}")
            self._managers[arch] = ResidencyManager(
                backend=backend, ctx=self._ctx,
                max_resident=self._runtime.max_resident_variants or 1)
        return self._managers[arch]

    def register_variant(self, variant: H3Variant) -> None:
        """Add or override a variant. Call before generate_video(variant=name)."""
        self._variants[variant.name] = variant

    def list_variants(self) -> list[str]:
        return sorted(self._variants)

    def warm(self) -> "MiniMaxH3Engine":
        """Pre-download the registered variants' repositories into the offline
        tree WITHOUT loading anything. Best-effort, never raises — safe in a
        worker's setup(). Heavy: ~50-90 GB per distinct repo."""
        if not self._loaded:
            self.load()
        from imference_engine.minimax_h3.loader import _H3_PATTERNS, _SENTINEL
        from imference_engine.runtime.offline import local_repo_dir
        import os
        seen = set()
        cache_dir = (str(self._runtime.model_cache_dir)
                     if self._runtime.model_cache_dir else None)
        for v in self._variants.values():
            if v.repo in seen or os.path.isdir(v.repo):
                continue
            seen.add(v.repo)
            try:
                local_repo_dir(v.repo, _H3_PATTERNS, cache_dir, namespace="video",
                               sentinel=_SENTINEL, cdn_base=self._runtime.model_cdn)
                logger.info("MiniMax-H3 repo warmed (%s)", v.repo)
            except Exception as e:  # noqa: BLE001
                logger.warning("warm: %s failed (%s); will fetch lazily", v.repo, e)
        return self

    # ------------------------------------------------------------------
    def generate_video(
        self,
        *,
        prompt: str,
        variant: str = "minimax-h3",
        image: Optional["Image"] = None,
        last_image: Optional["Image"] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: int = 124,
        num_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> MediaResult:
        """Generate one clip + its soundtrack. ``image``/``last_image`` switch
        t2v -> i2v (start and/or end keyframe). ``width``/``height`` omitted ->
        the model's native 768-short-edge canvas for the keyframe's (or 16:9)
        aspect; a smaller multiple-of-32 canvas (960x544) is ~2x+ faster.
        ``num_frames`` snaps up to the next ``17n+5`` (5-15 s at the fixed
        24 fps). Never raises for generation failures — see ``errors``."""
        if not self._loaded:
            raise RuntimeError("Call MiniMaxH3Engine.load() before generate_video")
        if variant not in self._variants:
            raise KeyError(f"unknown variant {variant!r}; known: {self.list_variants()}")
        var = self._variants[variant]

        seed = self._make_seed(seed)
        steps = num_steps if num_steps is not None else var.num_steps

        try:
            # Fail fast on constraint violations (the pipeline would too, but
            # only after ~75 GB of weights are resident) + get the frame count
            # the model will actually generate for the result metadata.
            check_canvas(width, height)
            aligned_frames = check_num_frames(num_frames)
            if aligned_frames != num_frames:
                logger.info("num_frames %d snapped up to %d (17n+5 grid)",
                            num_frames, aligned_frames)

            import torch
            backend = self._backends[var.arch]
            pipe = self._manager_for(var.arch).get_or_load(var)

            call = backend.build_call(
                prompt=prompt,
                negative_prompt=None,
                width=width,
                height=height,
                num_frames=aligned_frames,
                num_steps=steps,
                guidance_scale=None,
                guidance_scale_2=None,
                image=image,
                generator=torch.Generator("cpu").manual_seed(seed),
                last_image=last_image,
            )

            logger.info(
                "generate_video variant=%s mode=%s canvas=%s frames=%d steps=%d seed=%d",
                variant, "i2v" if (image or last_image) else "t2v",
                f"{width}x{height}" if width else "auto", aligned_frames, steps, seed)
            state = pipe(**call)

            frames = state.get("videos")[0]
            audio, sample_rate = _extract_audio(state)
            # The pipeline resolved the canvas when we passed None.
            out_w = state.get("width") or width
            out_h = state.get("height") or height
            return MediaResult(
                kind="video", media=list(frames), seeds=[seed],
                fps=MINIMAX_H3_FPS, num_frames=aligned_frames,
                width=out_w, height=out_h, variant=variant,
                audio=audio, sample_rate=sample_rate)
        except Exception as e:  # noqa: BLE001
            logger.error("generate_video failed: %s", e, exc_info=True)
            return MediaResult(
                kind="video", media=[], seeds=[seed],
                errors=[GenerationError(error=str(e), seed=seed)],
                fps=MINIMAX_H3_FPS, num_frames=num_frames,
                width=width, height=height, variant=variant)

    @property
    def resident(self) -> list[str]:
        return [name for m in self._managers.values() for name in m.resident]


def _extract_audio(state):
    """Pull the soundtrack off the pipeline state as a ``(2, n)`` float32 numpy
    waveform + its sample rate — the exact pair diffusers' ``encode_video``
    takes. Missing audio (shouldn't happen) degrades to (None, None)."""
    audio = state.get("audio")
    sample_rate = state.get("sampling_rate")
    if audio is None:
        return None, None
    waveform = audio[0]  # (1, 2, n) -> (2, n), the doc'd encode_video input
    if hasattr(waveform, "detach"):  # torch tensor -> portable numpy
        import torch
        waveform = waveform.detach().to(torch.float32).cpu().numpy()
    return waveform, sample_rate
