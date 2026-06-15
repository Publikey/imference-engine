"""Loading the validated Wan recipe: GGUF experts + shared components + LoRA-no-fuse.

Encodes exactly what spikes P1d/P1e/P1g proved:
- GGUF Q8 transformers via ``from_single_file`` (high → transformer, low → transformer_2);
- one shared UMT5 ``text_encoder`` + Wan ``vae`` reused across variants (P1g);
- Lightning LoRA applied with ``set_adapters`` and **never fused** (dodges #12047);
- ``enable_model_cpu_offload`` so VRAM stays ~one expert regardless of residency.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SharedComponents:
    """Big components shared across variants of the same base (loaded once).

    P1g confirmed the UMT5 text_encoder + Wan VAE are interchangeable between the
    T2V and I2V pipelines, so one set can serve every variant.
    """

    base_repo: str
    text_encoder: Any
    tokenizer: Any
    vae: Any


def load_shared_components(base_repo: str, *, cache_dir: Optional[str] = None) -> SharedComponents:
    import torch
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    logger.info("Loading shared components (text_encoder + vae) from %s", base_repo)
    kw = {"cache_dir": cache_dir} if cache_dir else {}
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_repo, subfolder="text_encoder", torch_dtype=torch.bfloat16, **kw)
    tokenizer = AutoTokenizer.from_pretrained(base_repo, subfolder="tokenizer", **kw)
    vae = AutoencoderKLWan.from_pretrained(
        base_repo, subfolder="vae", torch_dtype=torch.float32, **kw)
    return SharedComponents(base_repo, text_encoder, tokenizer, vae)


def _resolve_gguf(
    repo: str, quant: str, which: str,
    explicit_name: Optional[str] = None, template: Optional[str] = None,
) -> str:
    """Return a local path to the high/low GGUF for a quant level.

    Resolution order:
      1. ``explicit_name`` — exact path (quant-fixed).
      2. ``template`` with a ``{quant}`` placeholder — DETERMINISTIC, no API call,
         so it works under ``HF_HUB_OFFLINE=1`` from a mirrored cache.
      3. online auto-discovery via ``list_repo_files`` (needs network + auth).

    Only path 3 contacts the tree API; prefer 1/2 for offline/CDN deployments.
    """
    from huggingface_hub import hf_hub_download

    if explicit_name:
        logger.info("  %s [%s] -> %s", repo, which, explicit_name)
        return hf_hub_download(repo, explicit_name)
    if template:
        name = template.format(quant=quant)
        logger.info("  %s [%s] -> %s (template)", repo, which, name)
        return hf_hub_download(repo, name)

    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(repo) if f.endswith(".gguf")]
    cand = [f for f in files
            if which in f.lower() and f.lower().endswith(f"-{quant.lower()}.gguf")]
    if not cand:
        cand = [f for f in files if which in f.lower() and quant.lower() in f.lower()]
    if not cand:
        raise FileNotFoundError(
            f"no '{which}' '{quant}' .gguf in {repo}; available: {files}")
    logger.info("  %s [%s] -> %s", repo, which, cand[0])
    return hf_hub_download(repo, cand[0])


def _load_gguf_transformer(path: str, base_repo: str, subfolder: str) -> Any:
    import torch
    from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

    try:
        return WanTransformer3DModel.from_single_file(
            path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config=base_repo,
            subfolder=subfolder,
        )
    except ValueError as e:
        if "patch_embedding" in str(e):
            raise ValueError(
                f"GGUF at {path} stores patch_embedding in a ComfyUI-flattened shape "
                "diffusers can't load (e.g. BigDannyPt conversion). Use a known-good "
                "converter (QuantStack / Bedovyy) or a bf16 source. See doc P1e."
            ) from e
        raise


def _apply_loras(pipe: Any, loras: list) -> None:
    """Stack LoRAs onto the (possibly quantized) experts with set_adapters — no fuse.

    Validated on GGUF in P1d: each LoRA's high file → transformer, low file →
    transformer_2 (load_into_transformer_2=True); set_adapters keeps them as live
    bf16 adapters over the quantized weights.
    """
    names: list[str] = []
    weights: list[float] = []
    for i, lora in enumerate(loras):
        if lora.high_weight_name:
            adapter = f"lora{i}_high"
            pipe.load_lora_weights(lora.repo, weight_name=lora.high_weight_name,
                                   adapter_name=adapter)
            names.append(adapter)
            weights.append(lora.high_weight)
        if lora.low_weight_name:
            adapter = f"lora{i}_low"
            pipe.load_lora_weights(lora.repo, weight_name=lora.low_weight_name,
                                   adapter_name=adapter, load_into_transformer_2=True)
            names.append(adapter)
            weights.append(lora.low_weight)
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
        logger.info("  applied %d LoRA adapter(s), not fused: %s", len(names), names)


def build_pipeline(
    variant,
    *,
    quant: str,
    shared: SharedComponents,
    device: str,
    enable_offload: bool,
    vae_tiling: bool,
    cache_dir: Optional[str] = None,
) -> Any:
    """Build a ready-to-run Wan pipeline for ``variant`` at GGUF ``quant``."""
    import torch
    from diffusers import (UniPCMultistepScheduler, WanImageToVideoPipeline,
                           WanPipeline)

    logger.info("Building variant %r (%s, gguf=%s)", variant.name, variant.mode, quant)
    high = _resolve_gguf(variant.gguf_repo, quant, "high",
                         variant.gguf_high_name, variant.gguf_high_template)
    low = _resolve_gguf(variant.gguf_repo, quant, "low",
                        variant.gguf_low_name, variant.gguf_low_template)
    transformer = _load_gguf_transformer(high, variant.base_repo, "transformer")
    transformer_2 = _load_gguf_transformer(low, variant.base_repo, "transformer_2")

    pipe_cls = WanPipeline if variant.mode == "t2v" else WanImageToVideoPipeline
    kw = {"cache_dir": cache_dir} if cache_dir else {}
    pipe = pipe_cls.from_pretrained(
        variant.base_repo,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=shared.vae,
        text_encoder=shared.text_encoder,
        tokenizer=shared.tokenizer,
        torch_dtype=torch.bfloat16,
        **kw,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=variant.flow_shift)

    if variant.lightning_baked:
        logger.info("  lightning baked in checkpoint — no LoRA applied")
    else:
        _apply_loras(pipe, variant.loras)

    if enable_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    if vae_tiling:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    return pipe
