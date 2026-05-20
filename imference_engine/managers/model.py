"""Minimal model manager for V1: single-GPU, single resident model, no LRU.

The richer CPU/GPU LRU + LoRA-aware caching from the production worker gets
lifted in a follow-up. For now the manager holds at most one loaded pipeline
and swaps it out (to CPU + free cache) when switching to another model.
"""
from __future__ import annotations
import gc
import logging
from dataclasses import dataclass
from typing import Any, Optional

from imference_engine.pipelines.base import PipelineBackend
from imference_engine.runtime.device import Device

logger = logging.getLogger(__name__)


@dataclass
class RegisteredModel:
    name: str
    backend: str
    weights_path: str
    base_model: Optional[str] = None


class ModelManager:
    """Registry + lazy load + single-GPU residency."""

    def __init__(
        self, backends: dict[str, PipelineBackend], device: Device
    ) -> None:
        self._backends = backends
        self._device = device
        self._registered: dict[str, RegisteredModel] = {}
        self._loaded_name: Optional[str] = None
        self._loaded_pipe: Any = None

    def register(self, model: RegisteredModel) -> None:
        if model.backend not in self._backends:
            raise ValueError(
                f"Unknown backend {model.backend!r}. "
                f"Available: {list(self._backends)}"
            )
        self._registered[model.name] = model
        logger.info(f"Registered model {model.name!r} (backend={model.backend})")

    def get_or_load(self, name: str) -> tuple[Any, PipelineBackend]:
        """Load the named model onto the active device. Returns (pipe, backend)."""
        if name not in self._registered:
            raise KeyError(
                f"Model {name!r} is not registered. "
                f"Known: {list(self._registered)}"
            )

        if self._loaded_name == name:
            return self._loaded_pipe, self._backends[self._registered[name].backend]

        if self._loaded_name is not None:
            self._unload_current()

        meta = self._registered[name]
        backend = self._backends[meta.backend]
        pipe = backend.load_pipeline(
            local_path=meta.weights_path,
            base_model=meta.base_model,
        )
        pipe.to(self._device.torch_str)

        # VAE memory optimizations — lower peak VRAM at decode time.
        if hasattr(pipe, "vae") and pipe.vae is not None:
            try:
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"VAE optimizations not available: {e}")

        self._loaded_name = name
        self._loaded_pipe = pipe
        logger.info(f"Loaded {name!r} onto {self._device.torch_str}")
        return pipe, backend

    def _unload_current(self) -> None:
        if self._loaded_pipe is None:
            return
        name = self._loaded_name
        logger.info(f"Unloading {name!r}")
        try:
            self._loaded_pipe.to("cpu")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to move {name!r} to CPU during unload: {e}")
        self._loaded_pipe = None
        self._loaded_name = None
        gc.collect()
        self._free_device_cache()

    def _free_device_cache(self) -> None:
        try:
            import torch
            if self._device.kind == "cuda":
                torch.cuda.empty_cache()
            elif self._device.kind == "mps" and hasattr(torch, "mps"):
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        except ImportError:
            pass
