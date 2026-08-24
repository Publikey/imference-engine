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


# Files needed from a diffusers base repo. NEVER the transformer weights (we use
# GGUF). The shared base also needs the UMT5 text_encoder + tokenizer + VAE.
_SHARED_PATTERNS = ["model_index.json", "scheduler/*", "text_encoder/*",
                    "tokenizer/*", "vae/*",
                    "transformer/config.json", "transformer_2/config.json"]
_BASE_CFG_PATTERNS = ["model_index.json", "scheduler/*",
                      "transformer/config.json", "transformer_2/config.json"]


# The flat-tree resolver + downloader now live in imference_engine.runtime.offline
# (shared with the image side). These thin wrappers pin namespace="wan" so the Wan
# tree stays at <HF_HOME>/wan exactly as before — internal callers keep the local
# names and signatures, so nothing else in this module changes.
def _flat_root(cache_dir: Optional[str]) -> str:
    from imference_engine.runtime.offline import flat_root
    return flat_root(cache_dir, namespace="wan")


def _local_repo_dir(repo: str, patterns: list, cache_dir: Optional[str],
                    cdn_base: Optional[str] = None) -> str:
    from imference_engine.runtime.offline import local_repo_dir
    return local_repo_dir(repo, patterns, cache_dir, namespace="wan",
                          cdn_base=cdn_base)


def _quantize_text_encoder(te: Any, quant: Optional[str]) -> Any:
    """Best-effort weight-only int8 quant of the UMT5 text encoder.

    The encoder runs once per generation, so int8 weights hold quality while
    cutting ~4-6 GB CPU RAM (validated spike P2a). GRACEFUL: torchao missing,
    API drift, or any failure -> keep the bf16 encoder (never breaks a host)."""
    if not quant or quant == "none":
        return te
    if quant != "int8":
        logger.warning("text_encoder_quant=%r unsupported (use int8|none); bf16", quant)
        return te
    try:
        from torchao.quantization import quantize_
        try:
            from torchao.quantization import Int8WeightOnlyConfig
            cfg = Int8WeightOnlyConfig()
        except ImportError:  # older torchao exposes a factory fn instead
            from torchao.quantization import int8_weight_only
            cfg = int8_weight_only()
        quantize_(te, cfg)
        logger.info("text_encoder quantized to int8 (torchao weight-only)")
    except Exception as e:  # noqa: BLE001
        logger.warning("text_encoder int8 quant unavailable (%s); using bf16", e)
    return te


def _tie_umt5_input_embedding(text_encoder: Any) -> None:
    """Re-tie the UMT5 input embedding to ``shared`` after load.

    UMT5/T5 tie the input embedding: the checkpoint stores only ``shared.weight``
    (``encoder.embed_tokens`` is tied to it and NOT saved separately). transformers
    5.1 does not re-tie ``UMT5EncoderModel`` on ``from_pretrained`` — it leaves
    ``encoder.embed_tokens.weight`` zero-initialized, so the encoder emits garbage
    and generation IGNORES the prompt (confirmed: embed norm 0.0 vs shared norm
    262741.8). Point embed_tokens at ``shared`` explicitly. Guarded + idempotent:
    a no-op when the loader already tied them, and best-effort (never raises)."""
    try:
        shared = getattr(text_encoder, "shared", None)
        enc = getattr(text_encoder, "encoder", None)
        embed = getattr(enc, "embed_tokens", None) if enc is not None else None
        if shared is None or embed is None:
            return
        if embed.weight.data_ptr() != shared.weight.data_ptr():
            embed.weight = shared.weight
            logger.info("Re-tied UMT5 encoder.embed_tokens -> shared (transformers 5.1 load fix)")
    except Exception as e:  # noqa: BLE001
        logger.warning("UMT5 embedding re-tie skipped (%s)", e)


def load_shared_components(base_repo: str, *, cache_dir: Optional[str] = None,
                           text_encoder_quant: str = "none",
                           cdn_base: Optional[str] = None) -> SharedComponents:
    import torch
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    # cdn_base (WAN_MODEL_CDN) pulls the shared UMT5/tokenizer/VAE from the CDN
    # manifest instead of a HF snapshot — so a cold load is fully offline.
    d = _local_repo_dir(base_repo, _SHARED_PATTERNS, cache_dir, cdn_base=cdn_base)
    logger.info("Loading shared components (text_encoder + vae) from %s", d)
    text_encoder = UMT5EncoderModel.from_pretrained(
        d, subfolder="text_encoder", dtype=torch.bfloat16)
    _tie_umt5_input_embedding(text_encoder)
    text_encoder = _quantize_text_encoder(text_encoder, text_encoder_quant)
    tokenizer = AutoTokenizer.from_pretrained(d, subfolder="tokenizer")
    vae = AutoencoderKLWan.from_pretrained(
        d, subfolder="vae", dtype=torch.float32)
    return SharedComponents(base_repo, text_encoder, tokenizer, vae)



def _cdn_download(cdn_base: str, repo: str, filename: str, cache_dir: Optional[str]) -> str:
    from imference_engine.runtime.offline import cdn_download
    return cdn_download(cdn_base, repo, filename, cache_dir, namespace="wan")


def _fetch(repo: str, filename: str, cdn_base: Optional[str], cache_dir: Optional[str]) -> str:
    """Local path for a model file — CDN if set else HF. Branches locally (rather
    than delegating wholesale) so the local ``_cdn_download`` stays the seam that
    callers/tests monkeypatch."""
    if cdn_base:
        return _cdn_download(cdn_base, repo, filename, cache_dir)
    import os
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo, filename, local_dir=os.path.join(_flat_root(cache_dir), repo))


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
    return _fetch(repo, cand[0], None, cache_dir)


def _load_gguf_transformer(path: str, base_repo: str, subfolder: str) -> Any:
    import torch
    from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

    try:
        return WanTransformer3DModel.from_single_file(
            path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            dtype=torch.bfloat16,
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
    bf16 adapters over the quantized weights. The LoRA file is fetched into the
    flat tree (CDN or HF) and loaded from that local path.
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
            path = _fetch(lora.repo, wn, cdn_base, cache_dir)
            # Pass (dir, weight_name) — a bare path makes diffusers guess the file
            # name, which it refuses to do under HF_HUB_OFFLINE=1.
            import os
            pipe.load_lora_weights(
                os.path.dirname(path), weight_name=os.path.basename(path),
                adapter_name=adapter, load_into_transformer_2=into2)
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
    # base configs (model_index, scheduler, transformer configs) in a FLAT dir —
    # also used as the GGUF transformer `config=` so loading is fully local/offline.
    base_dir = _local_repo_dir(variant.base_repo, _BASE_CFG_PATTERNS, cache_dir,
                               cdn_base=cdn_base)
    high = _resolve_gguf(variant.gguf_repo, quant, "high",
                         variant.gguf_high_name, variant.gguf_high_template,
                         cdn_base=cdn_base, cache_dir=cache_dir)
    low = _resolve_gguf(variant.gguf_repo, quant, "low",
                        variant.gguf_low_name, variant.gguf_low_template,
                        cdn_base=cdn_base, cache_dir=cache_dir)
    transformer = _load_gguf_transformer(high, base_dir, "transformer")
    transformer_2 = _load_gguf_transformer(low, base_dir, "transformer_2")

    pipe_cls = WanPipeline if variant.mode == "t2v" else WanImageToVideoPipeline
    pipe = pipe_cls.from_pretrained(
        base_dir,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=shared.vae,
        text_encoder=shared.text_encoder,
        tokenizer=shared.tokenizer,
        dtype=torch.bfloat16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=variant.flow_shift)

    if variant.lightning_baked:
        logger.info("  lightning baked in checkpoint — no LoRA applied")
    else:
        _apply_loras(pipe, variant.loras, cdn_base=cdn_base, cache_dir=cache_dir)

    _log_build_diagnostics(pipe, shared, variant)

    if enable_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    if vae_tiling:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    return pipe


def _log_build_diagnostics(pipe: Any, shared: SharedComponents, variant: Any) -> None:
    """Best-effort INFO diagnostics to localize a bad render (all guarded — never
    affects the pipeline). Answers the three recurring i2v/t2v questions in ONE
    run: is the prompt encoder alive (embed tied to shared, non-zero norm — the
    transformers 5.1 UMT5 regression), are the Lightning adapters actually active
    (downloaded != applied), and which pipeline / MoE boundary is in effect."""
    try:
        te = shared.text_encoder
        emb = te.encoder.embed_tokens.weight
        shd = te.shared.weight
        logger.info(
            "DIAG text_encoder: embed_norm=%.1f shared_norm=%.1f tied=%s "
            "(embed_norm≈0 => prompt ignored)",
            float(emb.norm()), float(shd.norm()), emb.data_ptr() == shd.data_ptr())
    except Exception as e:  # noqa: BLE001
        logger.info("DIAG text_encoder: unavailable (%s)", e)
    try:
        active = pipe.get_active_adapters()
    except Exception:  # noqa: BLE001
        active = "n/a"
    logger.info(
        "DIAG pipeline=%s mode=%s boundary_ratio=%s active_adapters=%s baked=%s",
        type(pipe).__name__, variant.mode, getattr(pipe, "boundary_ratio", None),
        active, variant.lightning_baked)
    # i2v conditioning surface: does the transformer want a CLIP image encoder
    # (image_dim set), and did we actually load one? If image_dim is set but
    # image_encoder is None, the CLIP conditioning is silently missing -> shape
    # survives (VAE first-frame concat) but detail/motion is mush.
    if variant.mode == "i2v":
        try:
            tc = pipe.transformer.config
            logger.info(
                "DIAG i2v-cond: image_dim=%s in_channels=%s image_encoder=%s "
                "image_processor=%s",
                getattr(tc, "image_dim", "?"), getattr(tc, "in_channels", "?"),
                type(getattr(pipe, "image_encoder", None)).__name__,
                type(getattr(pipe, "image_processor", None)).__name__)
        except Exception as e:  # noqa: BLE001
            logger.info("DIAG i2v-cond: unavailable (%s)", e)
