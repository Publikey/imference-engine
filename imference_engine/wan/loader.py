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


def _cdn_download(cdn_base: str, repo: str, filename: str, cache_dir: Optional[str]) -> str:
    """Download <cdn_base>/<repo>/<filename> to a local cache (skip if present).

    The CDN must mirror the HF repo/filename layout. Uses urllib (not
    huggingface_hub), so it works alongside HF_HUB_OFFLINE=1.
    """
    import os
    import urllib.error
    import urllib.request

    # CDN files cache under cache_dir, else $HF_HOME/wan-cdn (same big volume as
    # the HF cache), else ~/.cache/wan-cdn. Avoids filling a small container disk.
    if cache_dir:
        root = cache_dir
    elif os.environ.get("HF_HOME"):
        root = os.path.join(os.environ["HF_HOME"], "wan-cdn")
    else:
        root = os.path.join(os.path.expanduser("~"), ".cache", "wan-cdn")
    dest = os.path.join(root, repo, filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        import sys
        url = f"{cdn_base.rstrip('/')}/{repo}/{filename}"
        logger.info("  CDN fetch %s", url)
        tmp = dest + ".part"
        # Send a browser-like User-Agent: Cloudflare (R2 custom domains) and many
        # CDNs return 403 to the default "Python-urllib/x" agent.
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (imference-engine wan)"})
        label = os.path.basename(filename)
        try:
            with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    buf = r.read(8 * 1024 * 1024)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    pct = f" ({done * 100 // total}%)" if total else ""
                    print(f"\r  CDN {label}: {done // 1024 // 1024} MiB{pct}   ",
                          end="", file=sys.stderr, flush=True)
                print("", file=sys.stderr, flush=True)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"CDN {e.code} for {url}") from e
        os.replace(tmp, dest)
    return dest


def _fetch(repo: str, filename: str, cdn_base: Optional[str], cache_dir: Optional[str]) -> str:
    """Local path for a model file — from the CDN mirror if set, else HF."""
    if cdn_base:
        return _cdn_download(cdn_base, repo, filename, cache_dir)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, **({"cache_dir": cache_dir} if cache_dir else {}))


def _resolve_gguf(
    repo: str, quant: str, which: str,
    explicit_name: Optional[str] = None, template: Optional[str] = None,
    cdn_base: Optional[str] = None, cache_dir: Optional[str] = None,
) -> str:
    """Return a local path to the high/low GGUF for a quant level.

    Resolution order:
      1. ``explicit_name`` — exact path (quant-fixed).
      2. ``template`` with a ``{quant}`` placeholder — DETERMINISTIC, no API call.
      3. online auto-discovery via ``list_repo_files`` (HF only; needs network).

    The resolved file is fetched from ``cdn_base`` if set, else HF. CDN mode needs
    a name/template (no listing over a CDN).
    """
    name = explicit_name or (template.format(quant=quant) if template else None)
    if name:
        logger.info("  %s [%s] -> %s%s", repo, which, name, " (cdn)" if cdn_base else "")
        return _fetch(repo, name, cdn_base, cache_dir)
    if cdn_base:
        raise ValueError(
            f"{repo}: CDN mode needs gguf_*_name/template for the {which} expert")

    from huggingface_hub import hf_hub_download, list_repo_files
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


def _apply_loras(pipe: Any, loras: list,
                 cdn_base: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
    """Stack LoRAs onto the (possibly quantized) experts with set_adapters — no fuse.

    Validated on GGUF in P1d: each LoRA's high file → transformer, low file →
    transformer_2 (load_into_transformer_2=True); set_adapters keeps them as live
    bf16 adapters over the quantized weights. When ``cdn_base`` is set, the LoRA
    file is fetched from the CDN to a local path and loaded from there.
    """
    names: list[str] = []
    weights: list[float] = []
    for i, lora in enumerate(loras):
        for wn, weight, into2, suffix in (
            (lora.high_weight_name, lora.high_weight, False, "high"),
            (lora.low_weight_name, lora.low_weight, True, "low"),
        ):
            if not wn:
                continue
            adapter = f"lora{i}_{suffix}"
            if cdn_base:
                path = _cdn_download(cdn_base, lora.repo, wn, cache_dir)
                pipe.load_lora_weights(path, adapter_name=adapter,
                                       load_into_transformer_2=into2)
            else:
                pipe.load_lora_weights(lora.repo, weight_name=wn, adapter_name=adapter,
                                       load_into_transformer_2=into2)
            names.append(adapter)
            weights.append(weight)
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
    cdn_base: Optional[str] = None,
) -> Any:
    """Build a ready-to-run Wan pipeline for ``variant`` at GGUF ``quant``.

    ``cdn_base`` (env WAN_MODEL_CDN) makes the GGUF experts + LoRAs come from a CDN
    mirror instead of HF — the shared base still loads from the (cached) HF repo.
    """
    import torch
    from diffusers import (UniPCMultistepScheduler, WanImageToVideoPipeline,
                           WanPipeline)

    logger.info("Building variant %r (%s, gguf=%s)", variant.name, variant.mode, quant)
    high = _resolve_gguf(variant.gguf_repo, quant, "high",
                         variant.gguf_high_name, variant.gguf_high_template,
                         cdn_base=cdn_base, cache_dir=cache_dir)
    low = _resolve_gguf(variant.gguf_repo, quant, "low",
                        variant.gguf_low_name, variant.gguf_low_template,
                        cdn_base=cdn_base, cache_dir=cache_dir)
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
        _apply_loras(pipe, variant.loras, cdn_base=cdn_base, cache_dir=cache_dir)

    if enable_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    if vae_tiling:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    return pipe
