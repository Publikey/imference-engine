"""Low-memory Qwen-Image-Edit: GGUF transformer + Lightning LoRA + NF4 encoder.

The recipe measured by ``experimental/qwen-edit-lite`` (RTX 4070 Laptop, 8 GB
VRAM: ~52 s per edit with group offload, 6.5 GB VRAM peak; Q4_K_M and Q5_K_M
visually identical) and chosen for Desktop AND cloud:

- transformer: an unsloth ``qwen-image-edit-2511-Q4_K_M.gguf`` (13.2 GB instead
  of 41 GB bf16), loaded with ``GGUFQuantizationConfig``;
- Lightning: lightx2v's 4-step LoRA, kept as an adapter (GGUF weights cannot
  absorb a fused delta), with lightx2v's scheduler — 4 steps, CFG 1.0;
- encoder: the 7B Qwen2.5-VL in NF4 (bitsandbytes, ~5.6 GB instead of 16.6 GB),
  quantized once and cached on disk (or pulled pre-quantized from the CDN),
  loaded for the encode phase and freed before denoising when VRAM is short
  (8-16 GB cards: the encoder and the transformer are never resident
  together). On a card with room for both (24 GB class, transformer resident)
  it stays loaded between edits, which saves reloading ~6 GB per edit
  (``QWEN_EDIT_KEEP_ENCODER``).

``QwenImageEditLitePipeline`` does the two phases inside one ``pipe(...)``
call, so the engine's generate loop, offload wiring and residency are untouched.
Without CUDA + bitsandbytes (Apple, CPU, AMD) the encoder stays bf16 in the
pipeline and the regular offload handles it.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import threading
from contextlib import contextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

LIGHTNING_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
LIGHTNING_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
# Derived artifact (NF4 save_pretrained of the base encoder): our mirror only,
# never on HuggingFace. Absent -> quantized locally from the bf16 encoder.
NF4_ENCODER_REPO = "imference/Qwen-Image-Edit-2511-text-encoder-nf4"

# lightx2v's scheduler for the Qwen-Image Lightning LoRAs (generate_with_diffusers.py).
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Base files for the lite build: the encoder's CONFIG only (its weights come
# from the NF4 cache), the transformer's config only (weights from the GGUF).
LITE_BASE_PATTERNS = [
    "model_index.json", "scheduler/*",
    "tokenizer/*", "processor/*",
    "vae/*",
    "transformer/config.json",
    "text_encoder/config.json", "text_encoder/generation_config.json",
]


def lightning_setting() -> str:
    """``QWEN_EDIT_LIGHTNING``: "4" (default) or "none"."""
    v = os.environ.get("QWEN_EDIT_LIGHTNING", "4").strip().lower()
    return "none" if v in ("none", "0", "off", "false") else "4"


def encoder_mode() -> str:
    """``QWEN_EDIT_ENCODER``: auto (default) | nf4 | bf16. auto = NF4 when
    CUDA (not ROCm) and bitsandbytes are available, else bf16."""
    want = os.environ.get("QWEN_EDIT_ENCODER", "auto").strip().lower()
    if want == "bf16":
        return "bf16"
    try:
        import torch
        cuda = torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
    except ImportError:
        cuda = False
    try:
        import bitsandbytes  # noqa: F401
        bnb = True
    except Exception:  # noqa: BLE001 — a broken bnb install counts as absent
        bnb = False
    if cuda and bnb:
        return "nf4"
    if want == "nf4":
        logger.warning("QWEN_EDIT_ENCODER=nf4 needs CUDA + bitsandbytes; using the bf16 encoder")
    return "bf16"


def keep_encoder_setting() -> str:
    """``QWEN_EDIT_KEEP_ENCODER``: auto (default) | 1 | 0. auto keeps the NF4
    encoder loaded across edits when the free VRAM left at denoise time covers
    the denoise reserve (``denoise_reserve_bytes``); 1 always keeps it, 0 always
    frees it (the original two-phase behaviour)."""
    v = os.environ.get("QWEN_EDIT_KEEP_ENCODER", "auto").strip().lower()
    if v in ("1", "true", "on", "yes", "always"):
        return "1"
    if v in ("0", "false", "off", "no", "never"):
        return "0"
    return "auto"


def denoise_reserve_bytes(height: Optional[int], width: Optional[int]) -> int:
    """VRAM the denoise + VAE decode need on top of the resident weights:
    ``QWEN_EDIT_DENOISE_RESERVE_GB`` (default 2.5) plus
    ``QWEN_EDIT_RESERVE_GB_PER_MP`` (default 1.5) per output megapixel."""
    def env_f(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except ValueError:
            return default

    mp = (height or 1024) * (width or 1024) / 1e6
    gb = env_f("QWEN_EDIT_DENOISE_RESERVE_GB", 2.5) + env_f("QWEN_EDIT_RESERVE_GB_PER_MP", 1.5) * mp
    return int(gb * (1 << 30))


MEGAPIXEL = 1024 * 1024  # diffusers' edit target area (VAE_IMAGE_SIZE)


def edit_area_bounds() -> tuple:
    """(min, max) work area in pixels, from ``QWEN_EDIT_MIN_MP`` (default 0.5)
    and ``QWEN_EDIT_MAX_MP`` (default 1.0), 1 MP = 1024x1024 px."""
    def env_f(name: str, default: float) -> float:
        try:
            v = float(os.environ.get(name, default))
            return v if v > 0 else default
        except ValueError:
            return default

    lo = env_f("QWEN_EDIT_MIN_MP", 0.5)
    hi = env_f("QWEN_EDIT_MAX_MP", 1.0)
    return int(min(lo, hi) * MEGAPIXEL), int(hi * MEGAPIXEL)


def edit_dimensions(width: int, height: int) -> Optional[tuple]:
    """(width, height) to edit an input of this size at, or None to keep
    diffusers' default (~1 MP, whatever the input).

    The input's own area, clamped to ``edit_area_bounds``: a small input is
    denoised (and VAE-conditioned) at its size instead of being blown up to
    1 MP, which is where the time goes (the transformer cost grows with the
    token count). Same rounding as diffusers (multiples of 32)."""
    if not width or not height:
        return None
    lo, hi = edit_area_bounds()
    area = max(lo, min(hi, width * height))
    if area >= MEGAPIXEL:
        return None
    ratio = width / height
    w = math.sqrt(area * ratio)
    return max(32, round(w / 32) * 32), max(32, round(w / ratio / 32) * 32)


_VAE_AREA_LOCK = threading.Lock()


@contextmanager
def _vae_condition_area(area: Optional[int]):
    """EditPlus VAE-encodes the condition images at a module constant
    (``VAE_IMAGE_SIZE``, 1 MP), not at the output size: set it to the output
    area for the call so the reference tokens shrink with the output."""
    if not area:
        yield
        return
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as mod

    with _VAE_AREA_LOCK:
        saved = mod.VAE_IMAGE_SIZE
        mod.VAE_IMAGE_SIZE = area
        try:
            yield
        finally:
            mod.VAE_IMAGE_SIZE = saved


def _pipeline_class():
    from diffusers import QwenImageEditPlusPipeline

    class QwenImageEditLitePipeline(QwenImageEditPlusPipeline):
        """EditPlus whose text encoder is loaded on demand for the encode phase.

        An explicit ``height``/``width`` under 1 MP (``edit_dimensions``) also
        shrinks the VAE condition images to that area.

        ``_encoder_loader`` returns a ready encoder (NF4, on the GPU). It is
        registered when ``encode_prompt`` first needs it. In ``prepare_latents``
        (the first step after encoding) it is released unless the free VRAM
        covers the denoise reserve (``QWEN_EDIT_KEEP_ENCODER``): then it stays
        loaded and the next edit skips the reload.
        """

        _encoder_loader: Optional[Callable[[], Any]] = None

        def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None, dtype=None):
            if self.text_encoder is None and self._encoder_loader is not None:
                self.register_modules(text_encoder=self._encoder_loader())
            target = device or self._execution_device
            enc_device = getattr(self.text_encoder, "device", None) or target
            embeds, mask = super()._get_qwen_prompt_embeds(prompt, image, enc_device, dtype)
            return embeds.to(target), mask.to(target)

        def prepare_latents(self, *args, **kwargs):
            height = kwargs.get("height", args[3] if len(args) > 3 else None)
            width = kwargs.get("width", args[4] if len(args) > 4 else None)
            if not self._keep_encoder(height, width):
                self.release_encoder()
            return super().prepare_latents(*args, **kwargs)

        def _keep_encoder(self, height, width) -> bool:
            if self._encoder_loader is None or self.text_encoder is None:
                return False
            setting = keep_encoder_setting()
            if setting != "auto":
                return setting == "1"
            try:
                import torch
                if not torch.cuda.is_available():
                    return False
                free, _ = torch.cuda.mem_get_info()
                # Memory torch already holds but isn't using (the encode
                # activations' cache) is free for the denoise too.
                free += max(0, torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
            except Exception:  # noqa: BLE001 — unknown: be safe, free it
                return False
            need = denoise_reserve_bytes(height, width)
            keep = free >= need
            logger.info("Qwen-Image-Edit: %.1f GiB free, denoise needs ~%.1f GiB -> encoder %s",
                        free / (1 << 30), need / (1 << 30), "kept" if keep else "released")
            return keep

        def __call__(self, *args, **kwargs):
            peak = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    peak = torch.cuda
            except Exception:  # noqa: BLE001
                peak = None
            ok = False
            h, w = kwargs.get("height"), kwargs.get("width")
            area = h * w if h and w and h * w < MEGAPIXEL else None
            try:
                with _vae_condition_area(area):
                    out = super().__call__(*args, **kwargs)
                ok = True
                return out
            finally:
                # A failed call (OOM...) never leaves the encoder behind.
                if not ok or keep_encoder_setting() == "0":
                    self.release_encoder()
                if peak is not None:
                    logger.info("Qwen-Image-Edit: peak VRAM %.2f GiB (encoder %s)",
                                peak.max_memory_allocated() / (1 << 30),
                                "resident" if self.text_encoder is not None else "freed")

        def release_encoder(self) -> None:
            if self._encoder_loader is None or self.text_encoder is None:
                return
            self.register_modules(text_encoder=None)
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    return QwenImageEditLitePipeline


def load_gguf_transformer(path: str, base_dir: str):
    import torch
    from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel

    logger.info("Loading Qwen-Image-Edit GGUF transformer %s", os.path.basename(path))
    return QwenImageTransformer2DModel.from_single_file(
        path, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=base_dir, subfolder="transformer", torch_dtype=torch.bfloat16)


def apply_lightning(pipe, *, cache_dir: Optional[str], cdn_base: Optional[str]) -> None:
    """Load the 4-step Lightning LoRA as an adapter + lightx2v's scheduler."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    from imference_engine.runtime.offline import fetch

    path = fetch(LIGHTNING_REPO, LIGHTNING_FILE, cdn_base, cache_dir, namespace="image")
    pipe.load_lora_weights(path, adapter_name="lightning")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    logger.info("Qwen-Image-Edit: Lightning 4-step LoRA active (adapter, not fused)")


def ensure_text_encoder_weights(base_repo: str, base_dir: str, *, cdn_base: Optional[str]) -> None:
    """The bf16 encoder shards in ``base_dir`` (the lite base pull skips them)."""
    index = os.path.join(base_dir, "text_encoder", "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        if all(os.path.isfile(os.path.join(base_dir, "text_encoder", s)) for s in shards):
            return
    if os.path.isdir(base_repo):
        return  # a local base dir is taken as-is
    if cdn_base:
        from imference_engine.runtime.download import download_parallel

        base = f"{cdn_base.rstrip('/')}/{base_repo}/text_encoder"
        os.makedirs(os.path.dirname(index), exist_ok=True)
        if not os.path.isfile(index):
            download_parallel(f"{base}/model.safetensors.index.json", index, 4)
        with open(index, encoding="utf-8") as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for s in shards:
            dest = os.path.join(base_dir, "text_encoder", s)
            if not os.path.isfile(dest):
                download_parallel(f"{base}/{s}", dest, 8)
        return
    from huggingface_hub import snapshot_download
    snapshot_download(base_repo, allow_patterns=["text_encoder/*"], local_dir=base_dir)


def nf4_encoder_dir(cache_dir: Optional[str]) -> str:
    from imference_engine.runtime.offline import flat_root

    return os.path.join(flat_root(cache_dir, namespace="image"), NF4_ENCODER_REPO)


def nf4_encoder_ready(cached: str) -> bool:
    return os.path.isfile(os.path.join(cached, "config.json")) and any(
        n.endswith(".safetensors") for n in os.listdir(cached))


def prefetch_lite_extras(*, cache_dir: Optional[str], cdn_base: Optional[str]) -> None:
    """Download (no load) what the lite build fetches on its first edit: the
    Lightning LoRA and, in NF4 mode, the pre-quantized encoder from the CDN.
    For a worker warm at deploy time. Best-effort: a miss stays lazy."""
    from imference_engine.runtime.offline import fetch

    if lightning_setting() != "none":
        try:
            fetch(LIGHTNING_REPO, LIGHTNING_FILE, cdn_base, cache_dir, namespace="image")
        except Exception as e:  # noqa: BLE001
            logger.warning("prefetch: Lightning LoRA not fetched (%s); lazy on first edit", e)
    if encoder_mode() == "nf4" and cdn_base:
        cached = nf4_encoder_dir(cache_dir)
        if os.path.isdir(cached) and nf4_encoder_ready(cached):
            return
        try:
            from imference_engine.runtime.offline import _cdn_snapshot
            _cdn_snapshot(cdn_base, NF4_ENCODER_REPO, cached)
        except Exception as e:  # noqa: BLE001
            logger.warning("prefetch: NF4 encoder not fetched (%s); lazy on first edit", e)


def nf4_encoder_loader(base_repo: str, base_dir: str, *, cache_dir: Optional[str],
                       cdn_base: Optional[str]) -> Callable[[], Any]:
    """A loader for the NF4 encoder: the cached quantized copy, else the CDN
    copy, else quantize the bf16 encoder once and cache it."""
    cached = nf4_encoder_dir(cache_dir)

    def ready() -> bool:
        return os.path.isdir(cached) and nf4_encoder_ready(cached)

    def load():
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration

        device = f"cuda:{torch.cuda.current_device()}"
        if not ready() and cdn_base:
            try:
                from imference_engine.runtime.offline import _cdn_snapshot
                _cdn_snapshot(cdn_base, NF4_ENCODER_REPO, cached)
            except Exception as e:  # noqa: BLE001 — not mirrored: quantize locally
                logger.info("NF4 encoder not on the CDN (%s); quantizing locally", e)
        if ready():
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(cached, device_map={"": device})
        from transformers import BitsAndBytesConfig

        ensure_text_encoder_weights(base_repo, base_dir, cdn_base=cdn_base)
        logger.info("Quantizing the Qwen2.5-VL encoder to NF4 (once; cached in %s)", cached)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
        te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_dir, subfolder="text_encoder", dtype=torch.bfloat16, quantization_config=q,
            device_map={"": device})
        try:
            te.save_pretrained(cached)
        except Exception as e:  # noqa: BLE001 — still usable, just not cached
            logger.warning("Could not cache the NF4 encoder: %s", e)
        return te

    return load


def load_lite_pipeline(gguf_path: str, base_repo: str, *, cache_dir: Optional[str],
                       cdn_base: Optional[str], full_base_patterns: list) -> Any:
    """GGUF transformer + Lightning + (NF4 two-phase | bf16) encoder."""
    import torch
    from diffusers import QwenImageEditPlusPipeline

    from imference_engine.runtime.offline import local_repo_dir

    mode = encoder_mode()
    if os.path.isdir(base_repo):
        base_dir = base_repo
    else:
        patterns = LITE_BASE_PATTERNS if mode == "nf4" else full_base_patterns
        base_dir = local_repo_dir(base_repo, patterns, cache_dir, namespace="image", cdn_base=cdn_base)
    transformer = load_gguf_transformer(gguf_path, base_dir)
    if mode == "nf4":
        pipe = _pipeline_class().from_pretrained(
            base_dir, transformer=transformer, text_encoder=None, local_files_only=True, dtype=torch.bfloat16)
        pipe._encoder_loader = nf4_encoder_loader(base_repo, base_dir, cache_dir=cache_dir, cdn_base=cdn_base)
    else:
        ensure_text_encoder_weights(base_repo, base_dir, cdn_base=cdn_base)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            base_dir, transformer=transformer, local_files_only=True, dtype=torch.bfloat16)
    if lightning_setting() != "none":
        apply_lightning(pipe, cache_dir=cache_dir, cdn_base=cdn_base)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.enable_tiling()
    logger.info("Qwen-Image-Edit lite pipeline ready (encoder=%s, lightning=%s)", mode, lightning_setting())
    return pipe
