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
from typing import TYPE_CHECKING, Iterable, Optional, Union

from imference_engine.catalog.defaults import GLOBAL_DEFAULTS, GenerationDefaults
from imference_engine.managers.batch import BatchSizer
from imference_engine.managers.model import ModelManager, RegisteredModel
from imference_engine.pipelines.base import PipelineBackend
from imference_engine.runtime.device import Device, resolve_device
from imference_engine.types import GenerationError, GenerationResult

if TYPE_CHECKING:
    from PIL.Image import Image
    from typing import Callable

logger = logging.getLogger(__name__)

# numpy int32 max — keeps seeds in a range schedulers accept
MAX_SEED = 2 ** 31 - 1


@dataclass
class RuntimeConfig:
    """Runtime knobs. All optional — defaults adapt to the host."""

    device: str = "auto"
    """auto | cuda | cuda:N | mps | cpu"""

    max_cpu_models: Optional[int] = None
    """Cap on CPU-resident pipes kept warm for fast GPU re-promotion. None
    or 0 = no CPU tier (evicted-from-GPU pipes are dropped immediately).
    Workers should set this to fit available RAM (e.g. 8 SDXL pipes ~ 50 GB).
    """

    max_gpu_models: Optional[int] = None
    """Cap on pipes concurrently resident on GPU. None or 1 = single-resident
    (every model switch evicts the previous one). Workers with big VRAM
    (e.g. 2x SDXL on a 24 GB A10) can bump this to amortize swap cost."""

    model_cache_dir: Optional[Union[str, Path]] = None
    """Root of the flat, symlink-free offline model tree (IMAGE_MODEL_CACHE). The
    backends resolve shared base-components (SDXL config + tokenizers + fp16-fix
    VAE; Z-Image base text_encoder/vae) into it, so a cold load is fully offline
    under HF_HUB_OFFLINE=1 once the tree is populated (base tarball / prefetch)."""

    model_cdn: Optional[str] = None
    """Base URL of a CDN (R2) mirroring the same <repo>/<file> layout. When set,
    on-demand base-components download from the CDN instead of HF. Mirror of
    WanRuntimeConfig.model_cdn."""

    lora_cache_dir: Optional[Union[str, Path]] = None

    use_tiny_vae: bool = False
    """When True, SDXL backends substitute the full VAE with TAESDxl (Tiny
    AutoEncoder, ~5 MB). Decode goes from ~20 s to ~2 s on tight-VRAM GPUs,
    at the cost of slight quality loss. Recommended for previews / dev
    iteration; toggle off for final-quality renders. No effect on Z-Image
    (which uses a different VAE architecture without a TAESD equivalent)."""

    enable_cpu_offload: bool = False
    """When True, ModelManager calls `pipe.enable_model_cpu_offload(device=...)`
    instead of moving the whole pipe to GPU. Diffusers/accelerate then
    shuttles individual submodels (text_encoder, unet, vae) between CPU and
    GPU as inference progresses — peak VRAM drops to the largest single
    submodel (~5 GB for SDXL unet) instead of the sum (~7 GB).
    Cost: ~10-30 % slower per gen due to per-forward CPU↔GPU transfers.
    Strongly recommended on ≤8 GB VRAM where the model otherwise saturates.
    Incompatible with the CPU LRU tier (forces max_cpu_models=0)."""

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        """Build a config from the documented image-side env contract.

        Every field falls back to its dataclass default when the env var is
        unset, so this is safe to call with NO environment at all (the desktop
        path constructs ``RuntimeConfig(...)`` directly instead). Workers call
        ``from_env()`` then override ``max_gpu_models`` / ``max_cpu_models`` with
        hardware-detected values (``config/resource_detection.py``) — that is the
        decoupling seam: the engine honours a static env/param contract; the
        worker layers hardware detection on top.

        ``MAX_GPU_MODELS`` / ``MAX_CPU_MODELS`` accept an integer; ``auto`` or
        unset leaves the field ``None`` (engine default = 1 GPU / 0 CPU). See
        ``imference_engine/pipelines/README.md`` for the full table.
        """
        from imference_engine.runtime.env import env_bool, env_int_or_none, env_str
        return cls(
            device=env_str("IMAGE_DEVICE", "auto"),
            model_cache_dir=env_str("IMAGE_MODEL_CACHE"),
            model_cdn=env_str("IMAGE_MODEL_CDN"),
            max_gpu_models=env_int_or_none("MAX_GPU_MODELS"),
            max_cpu_models=env_int_or_none("MAX_CPU_MODELS"),
            use_tiny_vae=env_bool("IMAGE_USE_TINY_VAE", False),
            enable_cpu_offload=env_bool("IMAGE_ENABLE_CPU_OFFLOAD", False),
        )


class Engine:
    """High-level diffusion inference engine.

    Supports single-resident (desktop default) and multi-tier LRU (worker
    via RuntimeConfig.max_gpu_models / max_cpu_models). LoRA, catalog YAML,
    and img2img remain TBD.

    Lifecycle:
        engine = Engine(runtime=RuntimeConfig(...)).load()
        engine.set_lifecycle_hooks(on_model_loaded=..., on_model_evicted=...)
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
        # Lifecycle hooks may be set before OR after load(); we wire them
        # into the ModelManager at load() time, and set_lifecycle_hooks
        # reaches into the manager if it already exists.
        self._on_model_loaded: Optional["Callable[[str], None]"] = None
        self._on_model_evicted: Optional["Callable[[str], None]"] = None

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
        cache_dir = (str(self._runtime.model_cache_dir)
                     if self._runtime.model_cache_dir else None)
        cdn_base = self._runtime.model_cdn
        self._backends = {
            SDXLBackend.engine: SDXLBackend(
                use_tiny_vae=self._runtime.use_tiny_vae,
                cache_dir=cache_dir, cdn_base=cdn_base),
            ZImageBackend.engine: ZImageBackend(
                cache_dir=cache_dir, cdn_base=cdn_base),
        }

        max_gpu = self._runtime.max_gpu_models or 1
        max_cpu = self._runtime.max_cpu_models or 0
        if self._runtime.enable_cpu_offload and max_cpu > 0:
            logger.warning(
                "enable_cpu_offload=True forces max_cpu_models=0 — the CPU LRU tier "
                "would shadow accelerate's hook-based offloader and confuse residency."
            )
            max_cpu = 0
        self._models = ModelManager(
            self._backends,
            self._device,
            max_gpu_models=max_gpu,
            max_cpu_models=max_cpu,
            on_loaded=self._on_model_loaded,
            on_evicted=self._on_model_evicted,
            enable_cpu_offload=self._runtime.enable_cpu_offload,
        )
        self._loaded = True
        return self

    def set_lifecycle_hooks(
        self,
        *,
        on_model_loaded: Optional["Callable[[str], None]"] = None,
        on_model_evicted: Optional["Callable[[str], None]"] = None,
    ) -> "Engine":
        """Wire callbacks invoked when a model is loaded from disk
        (`on_model_loaded`) or dropped from memory (`on_model_evicted`).

        Workers plug `disk_cache.protect` / `disk_cache.unprotect` to
        prevent the disk cache from evicting a .safetensors that's
        currently in use. Desktop callers typically leave these None.

        Idempotent; can be called before or after load(). For consistent
        tracking, prefer calling BEFORE the first generate() so the load
        of the first model fires on_model_loaded properly.
        """
        self._on_model_loaded = on_model_loaded
        self._on_model_evicted = on_model_evicted
        if self._models is not None:
            # Manager already exists (load() ran); patch its hooks in place.
            self._models._on_loaded = on_model_loaded
            self._models._on_evicted = on_model_evicted
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
        defaults: Optional[GenerationDefaults] = None,
    ) -> None:
        """Register a model so generate(model=name, ...) can load it.

        ``defaults`` is layer 2 of the precedence chain (per-model sampler
        settings, e.g. ``GenerationDefaults(num_steps=8,
        backend_options={"shift": 3.0})`` for a Z-Image turbo checkpoint). When
        omitted, the model carries no opinion and generate() resolves against
        the engine defaults + request only. The catalog YAML loader populates
        this from a ``models.yml``; until then callers register per model.
        """
        if not self._loaded:
            raise RuntimeError("Call Engine.load() before register_model")
        self._models.register(
            RegisteredModel(
                name=name,
                backend=backend,
                weights_path=str(weights_path),
                base_model=base_model,
                defaults=defaults or GenerationDefaults(),
            )
        )

    def warm(self, specs: "Iterable[tuple[str, Optional[str]]]" = ()) -> "Engine":
        """Pre-download base-components for the given (backend, base_model) pairs
        WITHOUT loading any model — so a worker can warm the base at deploy time
        and a fresh pod is 'ready' with the shared base on disk (the first request
        then only pays for the checkpoint weights, which stay lazy).

        `specs` is typically the worker's catalog as distinct
        (config.engine, config.base_model) pairs. Deduped. Best-effort: a failed
        prefetch logs a warning and is skipped (the component falls back to its
        lazy fetch on first use) — warm() never raises, so it's safe in setup().
        """
        if not self._loaded:
            self.load()
        seen: set = set()
        for backend_name, base_model in specs:
            key = (backend_name, base_model)
            if key in seen:
                continue
            seen.add(key)
            backend = self._backends.get(backend_name)
            if backend is None:
                logger.warning("warm: unknown backend %r — skipping", backend_name)
                continue
            try:
                backend.prefetch_base(base_model)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "warm: prefetch_base(%s, %s) failed (%s); will fetch lazily "
                    "on first use", backend_name, base_model, e)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        scheduler: Optional[str] = None,
        batch: int = 1,
        seed: Optional[int] = None,
        source_image: Optional["Image"] = None,
        strength: Optional[float] = None,
        loras: Optional[list[dict]] = None,
        backend_options: Optional[dict] = None,
    ) -> GenerationResult:
        """Generate ``batch`` images for ``model``.

        The sampling params (``num_steps``, ``guidance_scale``, ``width``,
        ``height``, ``scheduler``, ``clip_skip``, ``negative_prompt``,
        ``strength``, ``backend_options``) default to ``None`` = "not set at the
        request layer". Each unset param is resolved through the precedence chain
        ``request > model defaults > engine defaults > GLOBAL_DEFAULTS`` (finest
        wins), so a per-model catalog default fills what the request omits. Pass
        a value to override at the request layer.
        """
        if not self._loaded:
            raise RuntimeError("Call Engine.load() before generate")
        if loras:
            logger.warning("LoRA support not yet wired in V1; ignoring loras=%s", loras)

        pipe, backend = self._models.get_or_load(model)

        # Resolve the effective sampling params through the precedence chain:
        # request > model defaults > engine defaults > GLOBAL_DEFAULTS. A None
        # request field does NOT shadow a lower layer (that is why the signature
        # defaults are None, not concrete values).
        request = GenerationDefaults(
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            scheduler=scheduler,
            clip_skip=clip_skip,
            negative_prompt=negative_prompt,
            strength=strength,
            backend_options=backend_options or {},
        )
        eff = backend.engine_defaults()                          # layer 1
        eff = self._models.config_for(model).defaults.merged_over(eff)  # layer 2
        eff = request.merged_over(eff)                           # layer 3
        eff = eff.merged_over(GLOBAL_DEFAULTS)                   # bottom fallback
        # img2img: wrap the resident t2i pipe in the backend's img2img pipeline.
        # make_img2img reuses the SAME in-memory modules (vae/text_encoder/unet|
        # transformer/scheduler), so it's a cheap reference wrapper on the already-
        # resident weights — no extra disk load and no separate device accounting
        # (the shared modules are already on the active device). Built per request
        # rather than cached: the wrapper is light and this avoids shadowing the
        # ModelManager's residency bookkeeping.
        is_img2img = source_image is not None
        if is_img2img:
            pipe = backend.make_img2img(pipe)
        backend.apply_scheduler(pipe, eff.scheduler, **eff.backend_options)
        prompt_kwargs = backend.encode_prompts(pipe, prompt, eff.negative_prompt)

        seeds = [
            (seed + i) if seed is not None else random.randint(0, MAX_SEED)
            for i in range(batch)
        ]

        return self._run_chunked(
            pipe=pipe,
            backend=backend,
            prompt_kwargs=prompt_kwargs,
            width=eff.width,
            height=eff.height,
            num_steps=eff.num_steps,
            guidance_scale=eff.guidance_scale,
            clip_skip=eff.clip_skip,
            seeds=seeds,
            is_img2img=is_img2img,
            source_image=source_image,
            strength=eff.strength,
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
        source_image: Optional["Image"] = None,
        strength: float = 0.75,
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
                    image=source_image,
                    strength=strength,
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
