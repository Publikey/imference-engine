"""MiniMax-H3 pipeline builder — modular load, int8 quant, group offload.

UPSTREAM STATUS: MiniMax-H3 shipped in diffusers 0.40.0 (PR #14355); the
``[minimax-h3]`` extra pins that release — the same repo-wide pin every other
extra uses, so H3 no longer needs a dedicated venv. This loader still guards
every entry point with ``require_h3_support()`` so a stale install (an old venv
on diffusers <= 0.39) fails with an actionable message instead of a deep
AttributeError.

Load path (mirrors the upstream loading recipes, consumer-card variant):

1. Resolve the repo into the flat offline tree (CDN mirror when ``H3_MODEL_CDN``
   is set — same contract as every other backend). Only the FL2VA half is
   fetched: ``transformer/``, never ``transformer_ref/`` and never the original
   (non-diffusers) checkpoint folders the repository also carries.
2. ``ModularPipeline.from_pretrained(local_dir)`` + repoint each component spec
   at the local tree (the modular index names components by hub repo id — the
   exact issue the Anima backend documents; same fix, strict per-component
   check).
3. Quantization, by profile:
   - tree already carries torchao-serialized int8 weights (its
     ``transformer/config.json`` has a ``quantization_config``) -> plain load;
     this is the pre-staged-mirror fast path (``validation/stage_h3_int8.py``).
   - profile "int8" on a bf16 tree -> quantize AS the components load
     (``TorchAoConfig(Int8WeightOnlyConfig(version=2))`` on the transformer and
     the Qwen3-VL conditioner, with the upstream modules-to-not-convert lists).
   - profile "bf16" -> no quantization.
4. Freeze both big components (version-2 int8 tensors are pinnable, which
   streamed offload needs; freezing removes the one autograd path they cannot
   serve) and wire the offload mode: "block" (24-32 GB), "leaf" (12-16 GB,
   video VAE offloaded too), "none" (multi-GPU / 80 GB+).
"""
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from imference_engine.minimax_h3.presets import H3Variant

logger = logging.getLogger(__name__)

# The FL2VA half + shared components, and nothing else: no ``transformer_ref/*``
# (the Ref2VA partition, ~40 GB we never load) and none of the original
# checkpoint folders the official repo carries next to the converted ones.
# ``video_processor`` is config-created (no weights) but harmless to include in
# case a future conversion ships files for it.
_H3_PATTERNS = [
    "modular_model_index.json",
    "transformer/*",
    "text_encoder/*",
    "tokenizer/*",
    "processor/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
    "video_processor/*",
]
_SENTINEL = "modular_model_index.json"

# Upstream modules-to-not-convert lists (docs/source/en/api/pipelines/minimax_h3.md
# at the minimax-h3 branch): projections, embedders and norms stay bf16.
_TRANSFORMER_SKIP = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
    "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
_TEXT_ENCODER_SKIP = [
    "model.visual", "model.language_model.embed_tokens",
    "model.language_model.norm", "lm_head",
]

# Components the modular index declares but this engine NEVER loads: the
# Ref2VA partition is deliberately out of scope (presets.py), and its subfolder
# is never mirrored — left in ``load_components``' hands it would stream the
# ~66 GB bf16 partition straight from the Hub spec.
_EXCLUDED_COMPONENTS = ("transformer_ref",)

_INSTALL_HINT = (
    "MiniMax-H3 requires diffusers >= 0.40.0 (its integration, PR #14355, "
    "shipped in the 0.40.0 release). This venv's diffusers is older — a stale "
    "install predating the repo-wide 0.40.0 pin. Reinstall with:\n"
    "  pip install \"imference-engine[minimax-h3]\"   # pins diffusers==0.40.0"
)


def require_h3_support() -> None:
    """Raise ImportError with an actionable message unless the installed
    diffusers carries the MiniMax-H3 modular blocks."""
    try:
        from diffusers.modular_pipelines import MiniMaxH3Blocks  # noqa: F401
    except ImportError as e:
        raise ImportError(_INSTALL_HINT) from e


def _tree_is_prequantized(local_dir: str) -> bool:
    """True when the tree's transformer already carries serialized quantized
    weights (a staged int8 mirror) — then loading must NOT re-quantize."""
    cfg_path = os.path.join(local_dir, "transformer", "config.json")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            return "quantization_config" in json.load(f)
    except (OSError, ValueError):
        return False


def build_pipeline(
    spec: "H3Variant",
    *,
    profile: str,
    device: str,
    offload_mode: str,
    vae_tiling: bool,
    cache_dir: Optional[str] = None,
    cdn_base: Optional[str] = None,
    attention_backend: Optional[str] = None,
) -> Any:
    """Build a ready-to-run ``MiniMaxH3ModularPipeline`` for ``spec``.

    ``profile`` is "int8" | "bf16"; ``offload_mode`` is "block" | "leaf" |
    "none" (already resolved from "auto" by the engine).
    """
    require_h3_support()
    from diffusers import ModularPipeline

    from imference_engine.runtime.offline import local_repo_dir

    src = spec.repo
    if not os.path.isdir(src):
        src = local_repo_dir(
            spec.repo, _H3_PATTERNS, cache_dir,
            namespace="video", sentinel=_SENTINEL, cdn_base=cdn_base)

    logger.info("Loading MiniMax-H3 modular pipeline from %s (profile=%s offload=%s)",
                src, profile, offload_mode)
    pipe = ModularPipeline.from_pretrained(src)

    prequantized = _tree_is_prequantized(src)
    if profile == "int8" and not prequantized:
        _quantize_large_components(pipe, src)
    elif profile == "int8":
        logger.info("MiniMax-H3 tree is pre-quantized (serialized int8) — loading as-is")

    _load_components(pipe, src)

    # Freeze the two big components: version-2 int8 tensors are pinnable (which
    # streamed offload needs) and freezing removes the one autograd path
    # quantized tensors cannot serve. Free on bf16 too.
    for name in ("transformer", "text_encoder"):
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "requires_grad_"):
            module.requires_grad_(False)

    _wire_memory(pipe, device=device, offload_mode=offload_mode,
                 vae_tiling=vae_tiling)

    if attention_backend:
        try:
            pipe.transformer.set_attention_backend(attention_backend)
            logger.info("MiniMax-H3 attention backend: %s", attention_backend)
        except Exception as e:  # noqa: BLE001 — perf knob, never fatal
            logger.warning("set_attention_backend(%r) failed (%s); keeping SDPA",
                           attention_backend, e)
    return pipe


def _quantize_large_components(pipe: Any, local_dir: str) -> None:
    """Load the transformer + Qwen3-VL conditioner with on-the-fly torchao int8
    (weight-only, version 2) and inject them, so ``_load_components`` skips the
    bf16 loads of the two ~62 GB components. Exact upstream consumer-card recipe."""
    import torch
    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    try:
        from torchao.quantization import Int8WeightOnlyConfig
    except ImportError as e:
        raise ImportError(
            "H3_PROFILE=int8 needs torchao (pip install torchao) — or point the "
            "variant at a pre-quantized mirror staged with "
            "validation/stage_h3_int8.py.") from e

    logger.info("Quantizing MiniMax-H3 transformer + text encoder to int8 at load "
                "(one-time cost per cold load; stage_h3_int8.py serializes this)")
    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        local_dir, subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(version=2), modules_to_not_convert=_TRANSFORMER_SKIP),
        # NOTE: low_cpu_mem_usage must stay at its default (True) — diffusers
        # (0.40.0, unchanged from the PR #14355 head) REJECTS
        # low_cpu_mem_usage=False with quantization (earlier drafts of the PR
        # docs said the opposite).
    )
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        local_dir, subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2), modules_to_not_convert=_TEXT_ENCODER_SKIP),
    )
    pipe.update_components(transformer=transformer, text_encoder=text_encoder)


def _load_components(pipe: Any, local_dir: str) -> None:
    """Load the remaining modular components off the local tree, strictly.

    Same two modular-pipeline gotchas the Anima backend documents:
    ``modular_model_index.json`` records hub repo ids (repoint at the local tree
    or every load fails under ``HF_HUB_OFFLINE=1``), and ``load_components``
    only *warns* per-component (check + raise here, while the cause is in hand).
    Components already injected (quantized transformer/text_encoder) are not in
    ``null_component_names`` and are skipped automatically."""
    import torch

    names, paths = [], {}
    for name in pipe.null_component_names:
        if name in _EXCLUDED_COMPONENTS:
            logger.info("MiniMax-H3 component %r is out of scope — not loaded", name)
            continue
        spec = pipe.get_component_spec(name)
        src = getattr(spec, "pretrained_model_name_or_path", None)
        if not src or spec.default_creation_method != "from_pretrained":
            continue  # config-created (video_processor etc.) — load_components skips too
        names.append(name)
        if os.path.isdir(src):
            continue  # already local
        subfolder = getattr(spec, "subfolder", "") or ""
        if os.path.isdir(os.path.join(local_dir, subfolder)):
            paths[name] = local_dir
        else:
            logger.warning(
                "MiniMax-H3 component %r is sourced from %r, which is not mirrored "
                "under %s; leaving its spec untouched (needs network)",
                name, src, local_dir)

    pipe.load_components(names=names, pretrained_model_name_or_path=paths,
                         dtype=torch.bfloat16)

    missing = [n for n in names if getattr(pipe, n, None) is None]
    if missing:
        raise RuntimeError(
            f"MiniMax-H3 components failed to load from {local_dir}: "
            f"{', '.join(missing)}. See the warnings above for each component's "
            f"traceback — offline, this usually means the mirrored tree is "
            f"incomplete (expected subfolders: {', '.join(missing)})."
        )


def _wire_memory(pipe: Any, *, device: str, offload_mode: str,
                 vae_tiling: bool) -> None:
    """Place/offload the components per ``offload_mode`` (module docstring)."""
    import torch

    if offload_mode == "none":
        pipe.to(device)
    else:
        from diffusers.hooks import apply_group_offloading
        offload = dict(onload_device=torch.device(device),
                       offload_device=torch.device("cpu"), use_stream=True)
        pipe.transformer.enable_group_offload(
            offload_type="block_level", num_blocks_per_group=1, **offload)
        # ``.model`` — the inner Qwen3-VL; its lm_head is unused (H3 reads
        # layer-50 hidden states) and stays wherever it loaded.
        apply_group_offloading(pipe.text_encoder.model,
                               offload_type="leaf_level", **offload)
        if offload_mode == "leaf":
            # 12-16 GB recipe: the video VAE streams too (leaf level, no stream).
            apply_group_offloading(
                pipe.vae, offload_type="leaf_level",
                onload_device=torch.device(device),
                offload_device=torch.device("cpu"))
        else:
            pipe.vae.to(device)
        pipe.audio_vae.to(device)

    if vae_tiling:
        for fn in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, fn)()
            except (AttributeError, TypeError):
                pass  # best-effort — the H3 video VAE may not expose these


def teardown(pipe: Any) -> None:
    """Release a built pipeline's RAM before the next variant builds. Group
    offload leaves hooks + pinned host buffers a bare ``del`` does not return —
    remove hooks, then drop the big modules."""
    for target in (pipe, getattr(pipe, "transformer", None),
                   getattr(pipe, "text_encoder", None)):
        if target is None:
            continue
        for fn in ("remove_all_hooks", "maybe_free_model_hooks"):
            try:
                getattr(target, fn)()
            except Exception:  # noqa: BLE001
                pass
    for attr in ("transformer", "text_encoder", "vae", "audio_vae"):
        try:
            if getattr(pipe, attr, None) is not None:
                setattr(pipe, attr, None)
        except Exception:  # noqa: BLE001
            pass
