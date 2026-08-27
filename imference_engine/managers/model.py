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
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.pipelines.base import PipelineBackend
from imference_engine.runtime.device import Device

logger = logging.getLogger(__name__)


def _leaf_offload(module: Any, **kwargs: Any) -> None:
    """Thin indirection over diffusers.hooks.apply_group_offloading so unit
    tests can monkeypatch the diffusers call without a GPU (the import stays
    lazy for the pure-Python install)."""
    from diffusers.hooks import apply_group_offloading
    apply_group_offloading(module, **kwargs)


def _memlock_limit_bytes() -> float:
    """Effective RLIMIT_MEMLOCK soft limit in bytes; +inf when unlimited or
    unknowable (Windows has no resource module — and no memlock limit)."""
    try:
        import resource
        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft == resource.RLIM_INFINITY:
            return float("inf")
        return float(soft)
    except Exception:  # noqa: BLE001 — no limit we can detect
        return float("inf")


@dataclass
class RegisteredModel:
    """Metadata for a model — what to pass to backend.load_pipeline, plus the
    per-model generation defaults (layer 2 of the precedence chain)."""
    name: str
    backend: str
    weights_path: str
    base_model: Optional[str] = None
    defaults: GenerationDefaults = field(default_factory=GenerationDefaults)


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
        enable_offload: bool = False,
        offload_mode: str = "model",
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
        self._enable_offload = enable_offload
        if offload_mode not in ("model", "group"):
            logger.warning(
                f"Unknown offload_mode {offload_mode!r} (expected 'model' or "
                "'group'); using 'model'")
            offload_mode = "model"
        self._offload_mode = offload_mode

        logger.info(
            f"ModelManager configured: max_gpu_models={self._max_gpu}, "
            f"max_cpu_models={self._max_cpu}, "
            f"enable_offload={self._enable_offload}, "
            f"offload_mode={self._offload_mode}"
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

    def config_for(self, name: str) -> RegisteredModel:
        """Return the registration metadata (incl. per-model defaults) for a
        registered model. Raises KeyError if unknown — same contract as
        get_or_load."""
        if name not in self._registered:
            raise KeyError(
                f"Model {name!r} is not registered. "
                f"Known: {list(self._registered)}"
            )
        return self._registered[name]

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
        # Make room BEFORE loading: a cold load transiently costs a full pipe
        # of host RAM, so the GPU-LRU eviction (and any CPU-tier drop it
        # cascades into) must run first — otherwise the transient holds
        # evictee + new pipe at once and small-RAM pods OOM. Same invariant as
        # VideoResidency.get_or_load and the pre-engine workers. Trade-off:
        # if load_pipeline raises, a slot was evicted for nothing.
        self._ensure_gpu_slot()
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

            if self._enable_offload or self._max_cpu == 0:
                # The pipe is going away entirely — do NOT round-trip it
                # through pipe.to("cpu"), which would materialize a full pipe
                # of host RAM just to free it a line later. Dropping the last
                # reference frees the CUDA blocks via refcounting, and
                # _drop_pipe's empty_cache returns them to the driver.
                # (Under enable_offload this is also correctness, not just
                # RAM: accelerate's hooks are attached to the submodules and
                # a manual .to("cpu") would corrupt the offloader's
                # device-pinning state.)
                reason = "cpu_offload" if self._enable_offload else "no CPU tier"
                logger.info(f"GPU LRU eviction ({reason}): {evict_name!r} → drop")
                del evict_pipe  # last ref — release BEFORE _drop_pipe's gc pass
                self._drop_pipe(evict_name)
                continue

            logger.info(f"GPU LRU eviction: {evict_name!r} → CPU")
            self._swap_pipe_to_cpu(evict_pipe, evict_name)

            # Park on CPU; may evict another CPU resident in turn.
            self._cpu[evict_name] = evict_pipe
            self._cpu.move_to_end(evict_name)
            del evict_pipe  # the dict holds it now; don't shadow the cap pass
            self._enforce_cpu_cap()

    def _enforce_cpu_cap(self) -> None:
        """If the CPU tier is over capacity, drop oldest non-GPU residents."""
        while len(self._cpu) > self._max_cpu:
            # Oldest first. Models concurrently on GPU shouldn't be in _cpu
            # (we _gpu.pop() before _cpu insertion), so any entry is fair game.
            # Pop without binding: a local ref would survive _drop_pipe's
            # gc.collect() and delay the actual free.
            evict_name = next(iter(self._cpu))
            self._cpu.pop(evict_name)
            self._drop_pipe(evict_name)

    def _drop_pipe(self, name: str) -> None:
        """Finalize the release of a pipe whose last reference is ALREADY gone
        (callers must pop/del it first, or the gc pass below can't free it).
        Fires on_evicted so the caller can unprotect disk cache / log
        eviction metrics / etc."""
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
        purge the pipe entirely if VRAM is genuinely insufficient.

        When `enable_offload=True`, takes an offload path instead, per
        `offload_mode`:

        - "model": diffusers' `pipe.enable_model_cpu_offload(device=...)`
          installs hooks that shuttle whole submodels (text_encoder,
          unet/transformer, vae) between CPU and GPU on demand. Peak VRAM
          drops to the largest single submodel (~5 GB for the SDXL unet —
          but the 12-20B DiT transformers stay 13-26 GB, hence "group").
        - "group": diffusers group offloading — the backend's compute module
          streams block-by-block with a CUDA-stream prefetch, the other
          nn.Module components (text encoders) go leaf-level, and the VAE
          stays resident. Peak VRAM ~4-5 GB even for the 12-20B DiTs.
          CUDA-only; falls back to "model" elsewhere or on failure.

        Either way the hooks own device placement afterwards — such a pipe
        must never be pipe.to()'d again, which the eviction path already
        guarantees (offload pipes are dropped, never demoted to the CPU tier).
        """
        device = self._device.torch_str
        self._free_device_cache()

        if self._enable_offload:
            offloaded = False
            if self._offload_mode == "group":
                if self._device.kind != "cuda":
                    logger.warning(
                        f"{name!r}: offload_mode='group' needs CUDA "
                        f"(device is {self._device.kind}); using 'model' instead")
                else:
                    try:
                        self._apply_group_offload(pipe, name, device)
                        offloaded = True
                    except Exception as e:
                        logger.warning(
                            f"group offload failed for {name!r}; falling back "
                            f"to model offload: {e}")
            if not offloaded:
                # Accelerate manages device placement per-submodel from here
                # on — we do NOT call pipe.to(device), the hook system would
                # conflict.
                try:
                    pipe.enable_model_cpu_offload(device=device)
                    logger.info(
                        f"{name!r}: enable_model_cpu_offload(device={device!r}) — "
                        "submodels will shuttle on demand"
                    )
                except Exception as e:
                    logger.warning(
                        f"enable_model_cpu_offload failed for {name!r}; "
                        f"falling back to direct .to({device}): {e}"
                    )
                    pipe.to(device)
        else:
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
        # Safe under cpu_offload too: tiling/slicing are about how the decode
        # chunks its work, orthogonal to device placement.
        if hasattr(pipe, "vae") and pipe.vae is not None:
            try:
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
            except Exception as e:
                logger.debug(f"VAE post-GPU swap setup failed for {name!r}: {e}")

        self._free_device_cache()

    def _apply_group_offload(self, pipe: Any, name: str, device: str) -> None:
        """Wire diffusers group offloading onto a pipe (offload_mode="group").

        The backend's compute module (unet / transformer — the multi-GB part)
        streams **block by block** onto the GPU with a CUDA-stream prefetch, so
        its VRAM footprint is a few blocks instead of the whole model. The
        other nn.Module components (text encoders — the next-biggest pieces)
        are leaf-offloaded; the VAE moves to the device outright (small, and
        tiling/slicing already bound its decode peak). This is the same recipe
        as the MiniMax-H3 loader (block + stream) and the Qwen-Image 32-GB-card
        validation harness (20B bf16 at 4.7 GB peak VRAM).

        Raises on anything unexpected — the caller falls back to model offload.
        NOTE: like accelerate's hooks, group offloading leaves hooks + pinned
        host buffers that a bare `del pipe` does not fully return; the eviction
        path drops such pipes (never .to()s them) and empty_cache reclaims the
        device side. Mirrors the known limitation documented for the video
        backends.
        """
        import torch

        meta = self._registered.get(name)
        backend = self._backends.get(meta.backend) if meta else None
        compute = backend.get_compute_module(pipe) if backend else None
        if compute is None:
            raise RuntimeError(
                f"backend for {name!r} exposes no compute module to group-offload")

        offload_kwargs = dict(
            onload_device=torch.device(device),
            offload_device=torch.device("cpu"),
            use_stream=True,
            # UNPINNED host buffers by default (low_cpu_mem_usage=True).
            # Pinning the full multi-GB pipe is how streamed group offload
            # dies in the field, in two different ways:
            #   - containers cap RLIMIT_MEMLOCK (8 MB on vast.ai/Docker) →
            #     the process is SIGKILLed with no traceback (observed,
            #     exit 137 during wiring);
            #   - Windows has no memlock limit to detect, but cudaHostAlloc
            #     of ~26 GB on a 32 GB laptop fails as a "CUDA error: out of
            #     memory" whose async report POISONS the CUDA context — every
            #     later CUDA call in the process fails (observed on the
            #     desktop sidecar; the model-offload fallback then dies too).
            # Measured cost of unpinned on a datacenter pod: ~none vs the
            # theoretical pinned path (108.8 s / 5.9 GB peak for Krea 2
            # 12.9B). Big-RAM Linux hosts can opt back in with
            # IMAGE_GROUP_PINNED=1 (refused when memlock says otherwise).
            low_cpu_mem_usage=True,
        )
        if os.environ.get("IMAGE_GROUP_PINNED", "").strip() in ("1", "true", "yes"):
            if _memlock_limit_bytes() < (1 << 30):
                logger.warning(
                    f"{name!r}: IMAGE_GROUP_PINNED=1 ignored — RLIMIT_MEMLOCK "
                    "is below 1 GiB and pinning the pipe would kill the process")
            else:
                del offload_kwargs["low_cpu_mem_usage"]
                logger.info(f"{name!r}: group offload with PINNED host buffers "
                            "(IMAGE_GROUP_PINNED=1)")
        if hasattr(compute, "enable_group_offload"):
            compute.enable_group_offload(
                offload_type="block_level", num_blocks_per_group=1, **offload_kwargs)
        else:  # non-ModelMixin compute module — use the free function
            _leaf_offload(compute, offload_type="block_level",
                          num_blocks_per_group=1, **offload_kwargs)

        import torch.nn as nn
        components = getattr(pipe, "components", None) or {}
        vae = getattr(pipe, "vae", None)
        for comp_name, module in components.items():
            if module is None or not isinstance(module, nn.Module):
                continue  # tokenizers / schedulers
            if module is compute:
                continue
            if module is vae:
                module.to(device)
                continue
            _leaf_offload(module, offload_type="leaf_level", **offload_kwargs)
        logger.info(
            f"{name!r}: group offload wired (compute block_level+stream, "
            f"encoders leaf_level, vae resident on {device})")

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
