"""Dead-simple Stable Diffusion 1.5 engine.

A direct transcription of the working local snippet into a reusable class. The
whole point is to stay tiny: load the pipeline once in ``load()``, then call
``generate()`` per request. No catalog, no scheduler swapping, no VAE tricks,
no offline-tree plumbing. If you need any of that, use the real ``Engine``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from imference_engine.runtime.device import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

# Default checkpoint — the canonical SD 1.5 repo on the Hub. Override via the
# SD15Engine(model_id=...) constructor arg if you want a 1.5 fine-tune.
DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# Generation defaults. Only `prompt` is mandatory; everything else falls back
# here. 512x512 is SD 1.5's native resolution — go much bigger and anatomy
# drifts. 25 steps / cfg 7.5 is the classic sweet spot.
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_NUM_INFERENCE_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 7.5


class SD15Engine:
    """One SD 1.5 pipeline, loaded once, rendered many times.

    Single-threaded and stateful (the diffusers pipeline is not re-entrant);
    serialize concurrent callers. ``device`` defaults to "auto" (cuda → mps →
    cpu) via the shared device resolver.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self._device_pref = device
        self._pipe = None
        self._device: Optional[str] = None

    def load(self) -> "SD15Engine":
        """Load the pipeline onto the resolved device. Idempotent."""
        if self._pipe is not None:
            return self

        import torch
        from diffusers import StableDiffusionPipeline

        device = resolve_device(self._device_pref)
        torch_device = device.torch_str
        # fp16 on GPU (half the VRAM, faster); fp32 on CPU since CPU has no
        # native fp16 path and would crawl / error in half precision.
        dtype = torch.float16 if device.kind == "cuda" else torch.float32

        logger.info(
            "Loading SD1.5 pipeline %s on %s (%s)",
            self.model_id, torch_device, dtype,
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,  # disabled: otherwise it blanks out images to black
        )
        self._pipe = pipe.to(torch_device)
        self._device = torch_device
        logger.info("SD1.5 pipeline ready")
        return self

    def generate(
        self,
        *,
        prompt: str,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    ) -> "Image":
        """Render a single image and return it as a PIL ``Image``."""
        if self._pipe is None:
            raise RuntimeError("Call SD15Engine.load() before generate()")
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")

        logger.info(
            "Generating %dx%d steps=%d cfg=%s",
            width, height, num_inference_steps, guidance_scale,
        )
        result = self._pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        return result.images[0]
