"""Multi-tier model resident manager (GPU LRU + optional CPU LRU).

Lifted from gen-image-worker/workers/sdxl-multimodel/model.py (the OLD
worker — the v2 worker shim that uses this engine had regressed to a
single-resident model). The cost of that regression is ~10-30 s per
model swap for a worker juggling 3-5 SDXL checkpoints; with the CPU
tier restored, swaps drop back to ~0.5 s.

Configuration via Engine(RuntimeConfig(max_gpu_models=N, max_cpu_models=M)):

  - max_gpu_models = 1, max_cpu_models = 0 (defaults): single-resident,
    identical to the previous minimal manager. Used by the desktop sidecar.
  - max_gpu_models = 2, max_cpu_models = 8 (typical worker): up to 2 pipes
    concurrently in VRAM, up to 8 demoted-but-warm pipes in CPU RAM. Worker
    routes ~10-20 different models in production; the LRU keeps the
    frequently-hit ones hot.

Lifecycle hooks (on_loaded, on_evicted) let the caller plug in disk-cache
protection logic. Worker passes `disk_cache.protect`/`unprotect` so that
.safetensors files actively used in RAM aren't garbage-collected from disk.

The manager is intentionally NOT thread-safe — Engine docs state
"single-threaded and stateful, concurrent callers must serialize."
"""
from __future__ import annotations
import gc
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional

from imference_engine.pipelines.base import PipelineBackend
from imference_engine.runtime.device import Device

logger = logging.getLogger(__name__)


@dataclass
class RegisteredModel:
    """Metadata for a model — what to pass to backend.load_pipeline."""
    name: str
    backend: str
    weights_path: str
    base_model: Optional[str] = None


class ModelManager:
    """Two-tier LRU: GPU residents + optional CPU residents.

    GPU tier (`_gpu`): OrderedDict[name -> pipe], capped at max_gpu_models.
    Oldest in the OrderedDict is the LRU candidate for eviction.

    CPU tier (`_cpu`): OrderedDict[name -> pipe], capped at max_cpu_models.
    Models evicted from GPU land here (their reference is preserved so
    re-promotion to GPU is a cheap pipe.to(device) rather than a fresh
    disk read + load_pipeline). Models evicted from CPU are dropped
    entirely; on_evicted hook fires so the caller can unprotect disk.

    With max_cpu_models=0 the CPU tier is disabled — evicted-from-GPU
    models are dropped immediately, matching the previous minimal manager.
    """

    def __init__(
        self,
        backends: dict[str, PipelineBackend],
        device: Device,
        *,
        max_gpu_models: int = 1,
        max_cpu_models: int = 0,
        on_loaded: Optional[Callable[[str], None]] = None,
        on_evicted: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._backends = backends
        self._device = device
        self._registered: dict[str, RegisteredModel] = {}

        self._gpu: "OrderedDict[str, Any]" = OrderedDict()
        self._cpu: "OrderedDict[str, Any]" = OrderedDict()

        self._max_gpu = max(1, max_gpu_models)
        self._max_cpu = max(0, max_cpu_models)
        self._on_loaded = on_loaded
        self._on_evicted = on_evicted

        logger.info(
            f"ModelManager configured: max_gpu_models={self._max_gpu}, "
            f"max_cpu_models={self._max_cpu}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, model: RegisteredModel) -> None:
        if model.backend not in self._backends:
            raise ValueError(
                f"Unknown backend {model.backend!r}. "
                f"Available: {list(self._backends)}"
            )
        self._registered[model.name] = model
        logger.info(f"Registered model {model.name!r} (backend={model.backend})")

    def get_or_load(self, name: str) -> tuple[Any, PipelineBackend]:
        """Resolve a registered model to a (pipe, backend) tuple, with the
        pipe on the active device. Promotes from the CPU tier if available,
        otherwise loads from disk."""
        if name not in self._registered:
            raise KeyError(
                f"Model {name!r} is not registered. "
                f"Known: {list(self._registered)}"
            )

        meta = self._registered[name]
        backend = self._backends[meta.backend]

        # Already on GPU → touch LRU and return.
        if name in self._gpu:
            self._gpu.move_to_end(name)
            logger.debug(f"Model {name!r} already on GPU (LRU touched)")
            return self._gpu[name], backend

        # On CPU tier → swap to GPU (no disk read).
        if name in self._cpu:
            logger.info(f"Promoting {name!r}: CPU → GPU")
            pipe = self._cpu.pop(name)
            self._promote_cpu_to_gpu(name, pipe)
            return self._gpu[name], backend

        # Cold load from disk → CPU first, then promote to GPU.
        logger.info(f"Loading {name!r} from {meta.weights_path}")
        pipe = backend.load_pipeline(
            local_path=meta.weights_path,
            base_model=meta.base_model,
        )
        # The backend may return a pipe already configured for fp16/bfloat16
        # and possibly on CPU. We treat the load result as "on CPU" semantically
        # — the next step moves it to GPU under our slot accounting.
        self._safe_move_to_cpu(pipe)
        if self._on_loaded is not None:
            try:
                self._on_loaded(name)
            except Exception as e:
                logger.warning(f"on_loaded({name!r}) hook raised: {e}")

        self._promote_cpu_to_gpu(name, pipe)
        return self._gpu[name], backend

    # ------------------------------------------------------------------
    # Promotion / eviction
    # ------------------------------------------------------------------

    def _promote_cpu_to_gpu(self, name: str, pipe: Any) -> None:
        """Move a pipe from CPU RAM to GPU, making space first if needed.

        Evicts the GPU-LRU to CPU if max_gpu_models is hit. The freshly
        promoted pipe ends up at the MRU end of the GPU OrderedDict.
        """
        self._ensure_gpu_slot()
        self._swap_pipe_to_gpu(pipe, name)
        self._gpu[name] = pipe
        self._gpu.move_to_end(name)

    def _ensure_gpu_slot(self) -> None:
        """If the GPU tier is full, demote the GPU-LRU."""
        while len(self._gpu) >= self._max_gpu:
            evict_name = next(iter(self._gpu))  # oldest = LRU
            evict_pipe = self._gpu.pop(evict_name)
            logger.info(f"GPU LRU eviction: {evict_name!r} → CPU")
            self._swap_pipe_to_cpu(evict_pipe, evict_name)

            if self._max_cpu == 0:
                # CPU tier disabled — drop the pipe immediately.
                self._drop_pipe(evict_name, evict_pipe)
            else:
                # Park on CPU; may evict another CPU resident in turn.
                self._cpu[evict_name] = evict_pipe
                self._cpu.move_to_end(evict_name)
                self._enforce_cpu_cap()

    def _enforce_cpu_cap(self) -> None:
        """If the CPU tier is over capacity, drop oldest non-GPU residents."""
        while len(self._cpu) > self._max_cpu:
            # Oldest first. Models concurrently on GPU shouldn't be in _cpu
            # (we _gpu.pop() before _cpu insertion), so any entry is fair game.
            evict_name = next(iter(self._cpu))
            evict_pipe = self._cpu.pop(evict_name)
            self._drop_pipe(evict_name, evict_pipe)

    def _drop_pipe(self, name: str, _pipe: Any) -> None:
        """Release a pipe entirely. Fires on_evicted so the caller can
        unprotect disk cache / log eviction metrics / etc."""
        logger.info(f"Dropping {name!r} from memory")
        # No explicit unload — Python GC handles it when refs go away.
        # The caller's hook is the signal to free downstream resources.
        if self._on_evicted is not None:
            try:
                self._on_evicted(name)
            except Exception as e:
                logger.warning(f"on_evicted({name!r}) hook raised: {e}")
        self._free_device_cache()

    # ------------------------------------------------------------------
    # Low-level pipe<->device moves (lifted from v1, with the same VAE
    # tiling dance and OOM-retry that production already depended on)
    # ------------------------------------------------------------------

    def _swap_pipe_to_cpu(self, pipe: Any, name: str) -> None:
        """Demote a pipe to CPU. Disables VAE tiling/slicing first because
        those allocate CUDA tensors that explode on CPU-bound pipelines."""
        diffusers_logger = logging.getLogger("diffusers")
        original_level = diffusers_logger.level
        diffusers_logger.setLevel(logging.ERROR)
        try:
            if hasattr(pipe, "vae") and pipe.vae is not None:
                try:
                    pipe.vae.disable_slicing()
                    pipe.vae.disable_tiling()
                except Exception as e:
                    logger.debug(f"VAE pre-CPU swap cleanup failed for {name!r}: {e}")
            pipe.to("cpu")
        finally:
            diffusers_logger.setLevel(original_level)
        self._free_device_cache()

    def _swap_pipe_to_gpu(self, pipe: Any, name: str) -> None:
        """Promote a pipe to the active device, with OOM retry. Lifted
        from v1's swap_model_to_gpu — two retry attempts, then bail and
        purge the pipe entirely if VRAM is genuinely insufficient."""
        device = self._device.torch_str

        # Critical for big models (Z-Image ~12 GB barely fits in 20 GB VRAM):
        # force cleanup of residual allocations from the previous tenant
        # BEFORE attempting the move.
        self._free_device_cache()

        try:
            pipe.to(device)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            logger.warning(f"OOM moving {name!r} to {device}; aggressive cleanup + retry")
            try:
                pipe.to("cpu")
            except Exception:
                pass
            self._free_device_cache()
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                pipe.to(device)
            except RuntimeError as e2:
                if "out of memory" not in str(e2).lower():
                    raise
                logger.error(f"OOM retry failed for {name!r}; pipe will be dropped")
                try:
                    pipe.to("cpu")
                except Exception:
                    pass
                self._free_device_cache()
                raise RuntimeError(
                    f"OOM: {name!r} too large for {device}; pipe was purged from memory"
                ) from e2

        # Enable VAE optimizations AFTER the move — lowers peak VRAM
        # during batch decode (slicing) and large-resolution decode (tiling).
        if hasattr(pipe, "vae") and pipe.vae is not None:
            try:
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
            except Exception as e:
                logger.debug(f"VAE post-GPU swap setup failed for {name!r}: {e}")

        self._free_device_cache()

    def _safe_move_to_cpu(self, pipe: Any) -> None:
        """Cheap version of _swap_pipe_to_cpu for the fresh-from-load case:
        no VAE tiling teardown needed (the pipe was just constructed)."""
        try:
            pipe.to("cpu")
        except Exception as e:
            logger.debug(f"Initial move-to-CPU failed (non-fatal): {e}")

    def _free_device_cache(self) -> None:
        gc.collect()
        try:
            import torch
            if self._device.kind == "cuda":
                torch.cuda.empty_cache()
            elif self._device.kind == "mps" and hasattr(torch, "mps"):
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Introspection (used by tests and potentially callers)
    # ------------------------------------------------------------------

    def gpu_resident(self) -> list[str]:
        """Names of pipes currently on GPU, in LRU order (oldest first)."""
        return list(self._gpu)

    def cpu_resident(self) -> list[str]:
        """Names of pipes currently warm in CPU RAM, LRU order."""
        return list(self._cpu)
