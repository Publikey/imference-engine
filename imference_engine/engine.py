"""High-level Engine — public entry point.

Intentionally agnostic of transport: no Runqy, no FastAPI, no webhook URLs.
Callers wrap it in whatever surface they need (worker task handler, sidecar
HTTP server, in-process batch script).
"""
from __future__ import annotations
import gc
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from imference_engine.managers.batch import BatchSizer
from imference_engine.managers.model import ModelManager, RegisteredModel
from imference_engine.pipelines.base import PipelineBackend
from imference_engine.runtime.device import Device, resolve_device
from imference_engine.types import GenerationError, GenerationResult

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

# numpy int32 max — keeps seeds in a range schedulers accept
MAX_SEED = 2 ** 31 - 1


@dataclass
class RuntimeConfig:
    """Runtime knobs. All optional — defaults adapt to the host."""

    device: str = "auto"
    """auto | cuda | cuda:N | mps | cpu"""

    max_cpu_models: Optional[int] = None
    """Override CPU-resident model cap; None auto-detects (PR-next)."""

    max_gpu_models: Optional[int] = None
    """Override concurrently-on-GPU cap; None auto-detects (PR-next)."""

    model_cache_dir: Optional[Union[str, Path]] = None
    lora_cache_dir: Optional[Union[str, Path]] = None


class Engine:
    """High-level diffusion inference engine.

    V1 scope: single GPU, one model resident at a time, no LRU, no LoRA,
    no catalog YAML (use register_model). Substantial production-worker
    pieces — CPU/GPU LRU, LoRA stacking, model downloads, img2img — are
    lifted in follow-up PRs.

    Lifecycle:
        engine = Engine().load()
        engine.register_model("sdxl", backend="sdxl", weights_path="...")
        result = engine.generate(model="sdxl", prompt="cat", ...)

    Single-threaded and stateful. Concurrent callers must serialize.
    """

    def __init__(
        self,
        *,
        catalog_path: Optional[Union[str, Path]] = None,
        runtime: Optional[RuntimeConfig] = None,
    ) -> None:
        self._catalog_path = Path(catalog_path) if catalog_path else None
        self._runtime = runtime or RuntimeConfig()
        self._device: Optional[Device] = None
        self._backends: dict[str, PipelineBackend] = {}
        self._models: Optional[ModelManager] = None
        self._batch_sizer = BatchSizer()
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> "Engine":
        """Detect device, register default backends. Idempotent."""
        if self._loaded:
            return self

        self._device = resolve_device(self._runtime.device)
        logger.info(f"Engine device: {self._device.torch_str}")

        # Loud warning when we'd silently fall back to CPU: fp16 pipelines on CPU
        # hang at ~0 steps/s in PyTorch since CPU has no native fp16. Most users
        # hit this when they pip-install torch without CUDA in their venv.
        if self._device.kind == "cpu" and self._runtime.device == "auto":
            logger.warning(
                "Engine resolved device=cpu (no CUDA/MPS detected). "
                "fp16 pipelines will hang at 0%% on CPU. "
                "If you have a GPU, install CUDA torch: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )

        # Enable TF32 + flash SDPA when running on CUDA (free perf on Ampere+)
        if self._device.kind == "cuda":
            self._tune_cuda()

        from imference_engine.pipelines.sdxl import SDXLBackend
        from imference_engine.pipelines.zimage import ZImageBackend
        self._backends = {
            SDXLBackend.engine: SDXLBackend(),
            ZImageBackend.engine: ZImageBackend(),
        }

        self._models = ModelManager(self._backends, self._device)
        self._loaded = True
        return self

    @staticmethod
    def _tune_cuda() -> None:
        try:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"CUDA tuning skipped: {e}")

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def register_model(
        self,
        name: str,
        *,
        backend: str,
        weights_path: Union[str, Path],
        base_model: Optional[str] = None,
    ) -> None:
        """Register a model so generate(model=name, ...) can load it.

        For V1 the catalog YAML is not wired up; call this directly per model.
        """
        if not self._loaded:
            raise RuntimeError("Call Engine.load() before register_model")
        self._models.register(
            RegisteredModel(
                name=name,
                backend=backend,
                weights_path=str(weights_path),
                base_model=base_model,
            )
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 28,
        guidance_scale: float = 6.0,
        clip_skip: Optional[int] = None,
        scheduler: Optional[str] = None,
        batch: int = 1,
        seed: Optional[int] = None,
        source_image: Optional["Image"] = None,
        strength: float = 0.75,
        loras: Optional[list[dict]] = None,
        backend_options: Optional[dict] = None,
    ) -> GenerationResult:
        if not self._loaded:
            raise RuntimeError("Call Engine.load() before generate")
        if loras:
            logger.warning("LoRA support not yet wired in V1; ignoring loras=%s", loras)
        if source_image is not None:
            raise NotImplementedError("img2img lifted in a follow-up PR")

        pipe, backend = self._models.get_or_load(model)
        backend.apply_scheduler(pipe, scheduler, **(backend_options or {}))
        prompt_kwargs = backend.encode_prompts(pipe, prompt, negative_prompt)

        seeds = [
            (seed + i) if seed is not None else random.randint(0, MAX_SEED)
            for i in range(batch)
        ]

        return self._run_chunked(
            pipe=pipe,
            backend=backend,
            prompt_kwargs=prompt_kwargs,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            clip_skip=clip_skip,
            seeds=seeds,
            is_img2img=False,
        )

    def _run_chunked(
        self,
        *,
        pipe,
        backend: PipelineBackend,
        prompt_kwargs: dict,
        width: int,
        height: int,
        num_steps: int,
        guidance_scale: float,
        clip_skip: Optional[int],
        seeds: list[int],
        is_img2img: bool,
    ) -> GenerationResult:
        batch_total = len(seeds)
        max_gpu_batch = self._batch_sizer.get_max_batch_size(backend.engine, is_img2img)
        needs_profiling = max_gpu_batch == 0
        if needs_profiling:
            max_gpu_batch = 1

        images: list[Optional["Image"]] = []
        errors: list[GenerationError] = []
        idx = 0

        while idx < batch_total:
            chunk_size = min(max_gpu_batch, batch_total - idx)
            chunk_seeds = seeds[idx:idx + chunk_size]
            generators = [
                backend.make_generator(s, self._device.torch_str)
                for s in chunk_seeds
            ]
            generator = generators[0] if len(generators) == 1 else generators

            logger.info(
                f"Inference chunk {idx + 1}-{idx + chunk_size}/{batch_total} "
                f"seeds={chunk_seeds}"
            )

            baseline_vram = None
            total_vram = None
            if needs_profiling and self._device.kind == "cuda":
                import torch
                torch.cuda.reset_peak_memory_stats()
                baseline_vram = torch.cuda.memory_allocated()
                total_vram = torch.cuda.get_device_properties(0).total_memory

            try:
                kwargs = backend.build_inference_kwargs(
                    width=width,
                    height=height,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    clip_skip=clip_skip,
                    chunk_size=chunk_size,
                    generator=generator,
                )
                produced = pipe(**prompt_kwargs, **kwargs).images

                if needs_profiling and baseline_vram is not None:
                    import torch
                    peak_vram = torch.cuda.max_memory_allocated()
                    max_gpu_batch = self._batch_sizer.profile_and_compute(
                        backend.engine,
                        is_img2img,
                        total_vram=total_vram,
                        baseline_vram=baseline_vram,
                        peak_vram=peak_vram,
                    )
                    needs_profiling = False

                images.extend(produced)
                idx += chunk_size

            except Exception as e:  # noqa: BLE001
                if self._is_oom(e) and chunk_size > 1:
                    logger.warning(f"OOM at chunk_size={chunk_size}, halving and retrying")
                    self._free_cache()
                    max_gpu_batch = self._batch_sizer.report_oom(
                        backend.engine, is_img2img, chunk_size
                    )
                    needs_profiling = False
                    continue
                logger.error(
                    f"Inference failed for chunk {idx}-{idx + chunk_size}: {e}",
                    exc_info=True,
                )
                for j in range(chunk_size):
                    images.append(None)
                    errors.append(GenerationError(
                        error=str(e),
                        batch_index=idx + j,
                        seed=chunk_seeds[j],
                    ))
                idx += chunk_size

        return GenerationResult(images=images, seeds=seeds, errors=errors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        try:
            import torch
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:  # noqa: BLE001
            pass
        return "out of memory" in str(exc).lower()

    def _free_cache(self) -> None:
        gc.collect()
        try:
            import torch
            if self._device and self._device.kind == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass
