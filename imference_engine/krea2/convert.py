"""Krea 2 checkpoint normalization — ComfyUI/native single-files → diffusers.

diffusers 0.40 ships ``Krea2Pipeline`` / ``Krea2Transformer2DModel`` but NO
``from_single_file`` for them (upstream issue huggingface/diffusers#14122; PRs
#14126/#14264 unmerged as of 2026-08-27), and no scaled-fp8 handling for ANY
model. The civitai/ComfyUI ecosystem, meanwhile, ships Krea 2 Turbo finetunes
exclusively as transformer-only single-files in the NATIVE key layout
(``blocks.N.attn.wq`` / ``txtfusion.*``), predominantly quantized as ComfyUI
"scaled fp8" (float8_e4m3fn weights + a per-tensor float32 ``<name>.weight_scale``
+ a ``_quantization_metadata`` header entry).

This module bridges that gap IN MEMORY at load time (no on-disk conversion, no
offline step):

  1. strip an optional ``model.diffusion_model.`` / ``diffusion_model.`` prefix
     (all-in-one ComfyUI checkpoints),
  2. dequantize scaled-fp8 weights exactly: ``w = w_fp8.float() * weight_scale``
     (lossless w.r.t. the distributed file, computed in fp32, stored per-tensor
     in the compute dtype so the 12.9B model never materializes in fp32),
  3. remap the native key layout onto diffusers ``Krea2Transformer2DModel``
     module names.

The key mapping and dequant are vendored from InvokeAI (Apache-2.0,
``invokeai/backend/model_manager/load/model_loaders/krea2.py``, which itself
matches the mapping in the unmerged diffusers PR #14126). When upstream ships
single-file + scaled-fp8 support, this module can be dropped in favor of
``Krea2Transformer2DModel.from_single_file``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

# Default Krea2Transformer2DModel config (mirrors krea/Krea-2-Turbo
# transformer/config.json). Fallback for a bare single-file load when the base
# repo's transformer config is unavailable; the base_model path prefers the
# mirrored config.json.
KREA2_TRANSFORMER_CONFIG: Dict[str, Any] = {
    "attention_head_dim": 128,
    "axes_dims_rope": [32, 48, 48],
    "in_channels": 64,
    "intermediate_size": 16384,
    "norm_eps": 1e-05,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "num_layers": 28,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "num_text_layers": 12,
    "rope_theta": 1000.0,
    "text_hidden_dim": 2560,
    "text_intermediate_size": 6912,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "timestep_embed_dim": 256,
}


def strip_comfyui_prefix(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Strip a ``model.diffusion_model.`` / ``diffusion_model.`` key prefix if
    present (all-in-one ComfyUI checkpoints; bare Comfy-Org files have none).
    In-memory — unlike the Z-Image backend's on-disk variant."""
    prefix_to_strip = None
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if any(isinstance(k, str) and k.startswith(prefix) for k in sd):
            prefix_to_strip = prefix
            break
    if not prefix_to_strip:
        return sd
    logger.info("Krea 2: stripping ComfyUI %r key prefix", prefix_to_strip)
    return {
        (k[len(prefix_to_strip):] if isinstance(k, str) and k.startswith(prefix_to_strip) else k): v
        for k, v in sd.items()
    }


def is_native_krea2_format(sd: Dict[str, Any]) -> bool:
    """Detect the native/ComfyUI Krea 2 key naming vs. the diffusers naming."""
    return any(
        isinstance(k, str)
        and (k.startswith(("blocks.", "txtfusion.", "first.")) or ".mod.lin" in k)
        for k in sd
    )


def has_fp8_weights(sd: Dict[str, Any]) -> bool:
    """True when the checkpoint is fp8 on disk — ComfyUI scaled-fp8
    (``.weight_scale`` companions) or raw float8 tensors."""
    import torch

    if any(isinstance(k, str) and k.endswith(".weight_scale") for k in sd):
        return True
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    return any(getattr(v, "dtype", None) in fp8_dtypes for v in sd.values())


def dequantize_scaled_fp8(sd: Dict[str, Any], dtype: "Any") -> Dict[str, Any]:
    """Dequantize ComfyUI 'scaled fp8' weights: ``w = w_fp8.float() * weight_scale``.

    Each quantized layer stores an fp8 ``<name>.weight`` plus a (scalar)
    ``<name>.weight_scale``; the multiply runs in float32 for precision but each
    result is stored as ``dtype`` immediately, so the ~12.9B model never peaks
    at fp32 size (~50 GB) in host RAM. No-op when there are no scale keys.
    """
    import torch

    scale_keys = [k for k in sd if isinstance(k, str) and k.endswith(".weight_scale")]
    if not scale_keys:
        return sd
    out = dict(sd)
    for scale_key in scale_keys:
        weight_key = scale_key[: -len(".weight_scale")] + ".weight"
        if weight_key in out:
            weight = torch.as_tensor(out[weight_key]).float()
            scale = torch.as_tensor(out[scale_key]).float()
            out[weight_key] = (weight * scale).to(dtype)
            del weight
        del out[scale_key]
    logger.info("Krea 2: dequantized %d scaled-fp8 tensors -> %s", len(scale_keys), dtype)
    return out


def _drop_quant_metadata_keys(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ComfyUI quantization side-channel keys (``.comfy_quant`` /
    ``scale_input``) that have no diffusers counterpart."""
    return {
        k: v
        for k, v in sd.items()
        if not (isinstance(k, str) and (k.endswith(".comfy_quant") or "scale_input" in k))
    }


def _put_unique_key(
    dest: Dict[Any, Any], key: Any, value: Any, *, source: Any, source_of: Dict[Any, Any]
) -> None:
    """Assign ``dest[key] = value``, rejecting a collision produced by a
    different source key. A malformed MIXED-layout checkpoint (both native
    ``blocks.0.attn.wq.weight`` and diffusers ``transformer_blocks.0.attn.to_q.weight``)
    would otherwise make the surviving tensor depend on dict iteration order."""
    if key in dest:
        raise RuntimeError(
            f"Krea 2 checkpoint: source keys {source_of.get(key)!r} and {source!r} "
            f"both normalize to {key!r}. The checkpoint appears to mix native and "
            "diffusers key layouts; refusing to silently drop one of the tensors."
        )
    dest[key] = value
    source_of[key] = source


def convert_krea2_native_to_diffusers(sd: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a native/ComfyUI-format Krea 2 state dict to diffusers
    ``Krea2Transformer2DModel`` keys.

    Top-level module renames::

        blocks.N.*                  -> transformer_blocks.N.*
        txtfusion.*                 -> text_fusion.*
        first.*                     -> img_in.*
        tmlp.0/2.*                  -> time_embed.linear_1/2.*
        tproj.1.*                   -> time_mod_proj.*
        txtmlp.0/1/3.*              -> txt_in.norm / linear_1 / linear_2.*
        last.linear/norm/modulation -> final_layer.linear / norm.weight / scale_shift_table

    Within every transformer / text-fusion block::

        attn.wq/wk/wv/wo              -> attn.to_q/to_k/to_v/to_out.0
        attn.gate                     -> attn.to_gate
        attn.qknorm.qnorm/knorm.scale -> attn.norm_q/norm_k.weight
        mlp.gate/up/down              -> ff.gate/up/down
        prenorm/postnorm.scale        -> norm1/norm2.weight
        mod.lin (6*H,)                -> scale_shift_table (6, H)

    The native final-block ``last.down``/``last.up`` projections have no
    counterpart in the diffusers ``Krea2FinalLayer`` and are dropped.
    """
    import torch

    new_sd: Dict[str, Any] = {}
    source_of: Dict[Any, Any] = {}
    for key, value in sd.items():
        if not isinstance(key, str):
            _put_unique_key(new_sd, key, value, source=key, source_of=source_of)
            continue
        # Native-only final-block projections: no diffusers equivalent.
        if key in ("last.down.weight", "last.up.weight"):
            continue

        k = key
        # Top-level module prefixes.
        if k.startswith("blocks."):
            k = "transformer_blocks." + k[len("blocks."):]
        elif k.startswith("txtfusion."):
            k = "text_fusion." + k[len("txtfusion."):]
        elif k.startswith("first."):
            k = "img_in." + k[len("first."):]
        elif k.startswith("tmlp.0."):
            k = "time_embed.linear_1." + k[len("tmlp.0."):]
        elif k.startswith("tmlp.2."):
            k = "time_embed.linear_2." + k[len("tmlp.2."):]
        elif k.startswith("tproj.1."):
            k = "time_mod_proj." + k[len("tproj.1."):]
        elif k == "txtmlp.0.scale":
            k = "txt_in.norm.weight"
        elif k.startswith("txtmlp.1."):
            k = "txt_in.linear_1." + k[len("txtmlp.1."):]
        elif k.startswith("txtmlp.3."):
            k = "txt_in.linear_2." + k[len("txtmlp.3."):]
        elif k == "last.linear.weight":
            k = "final_layer.linear.weight"
        elif k == "last.linear.bias":
            k = "final_layer.linear.bias"
        elif k == "last.norm.scale":
            k = "final_layer.norm.weight"
        elif k == "last.modulation.lin":
            # Krea2FinalLayer.scale_shift_table is (2, hidden). Reshape the flat
            # native table like the per-block (6, H) tables below — otherwise
            # load_state_dict(assign=True) installs a wrong-shaped 1-D parameter
            # that only fails at inference.
            k = "final_layer.scale_shift_table"
            value = torch.as_tensor(value).reshape(2, -1)

        # Within-block sub-module renames (transformer_blocks.* and text_fusion.*).
        k = k.replace(".attn.wq.weight", ".attn.to_q.weight")
        k = k.replace(".attn.wk.weight", ".attn.to_k.weight")
        k = k.replace(".attn.wv.weight", ".attn.to_v.weight")
        k = k.replace(".attn.wo.weight", ".attn.to_out.0.weight")
        k = k.replace(".attn.gate.weight", ".attn.to_gate.weight")
        k = k.replace(".attn.qknorm.qnorm.scale", ".attn.norm_q.weight")
        k = k.replace(".attn.qknorm.knorm.scale", ".attn.norm_k.weight")
        k = k.replace(".mlp.gate.weight", ".ff.gate.weight")
        k = k.replace(".mlp.up.weight", ".ff.up.weight")
        k = k.replace(".mlp.down.weight", ".ff.down.weight")
        k = k.replace(".prenorm.scale", ".norm1.weight")
        k = k.replace(".postnorm.scale", ".norm2.weight")

        # Per-block modulation table: flat (6*H,) -> (6, H).
        if k.endswith(".mod.lin"):
            k = k[: -len(".mod.lin")] + ".scale_shift_table"
            value = torch.as_tensor(value).reshape(6, -1)

        _put_unique_key(new_sd, k, value, source=key, source_of=source_of)
    return new_sd


def prepare_krea2_state_dict(sd: Dict[str, Any], dtype: "Any") -> Tuple[Dict[str, Any], bool]:
    """Full normalization pipeline for a Krea 2 single-file state dict.

    Returns ``(diffusers-keyed state dict in ``dtype``, source_was_fp8)`` —
    the flag drives the backend's fp8-resident storage decision.
    """
    import torch

    sd = strip_comfyui_prefix(sd)
    was_fp8 = has_fp8_weights(sd)
    sd = dequantize_scaled_fp8(sd, dtype)
    sd = _drop_quant_metadata_keys(sd)
    # Plain-fp8 files (raw float8 weights, NO weight_scale — some community
    # quants): upcast to the compute dtype, else nn.Linear gets raw float8
    # parameters and fails at forward. Scaled-fp8 weights are already handled
    # above; every other tensor keeps its file dtype (bf16 weights stay bf16,
    # fp32 norm scales stay fp32).
    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    sd = {
        k: (v.to(dtype) if getattr(v, "dtype", None) in fp8_dtypes else v)
        for k, v in sd.items()
    }
    if is_native_krea2_format(sd):
        sd = convert_krea2_native_to_diffusers(sd)
        logger.info("Krea 2: native/ComfyUI key layout remapped to diffusers (%d tensors)", len(sd))
    return sd, was_fp8


def reject_incomplete_load(model: Any, *, what: str) -> None:
    """Raise if ``load_state_dict(strict=False)`` left required tensors on the
    meta device — an incomplete/misidentified checkpoint should fail AT LOAD
    with an actionable message, not mid-inference with a meta-tensor crash.
    Buffers are checked too (``init_empty_weights`` puts them on meta as well).
    """
    still_meta = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if getattr(tensor, "is_meta", False)
    ]
    if still_meta:
        raise RuntimeError(
            f"{what} is incomplete: {len(still_meta)} tensor(s) were not provided "
            f"by the checkpoint and remain uninitialized (meta device). First few: "
            f"{still_meta[:8]}. The file is likely incomplete, misidentified, or "
            "uses a key layout that needs conversion."
        )
