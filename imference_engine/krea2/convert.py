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
  2. dequantize quantized weights exactly —
     * ComfyUI **scaled fp8**: ``w = w_fp8.float() * weight_scale``,
     * ComfyUI **int8 "ConvRot"** (``<base>.weight`` int8 + per-channel
       ``.weight_scale`` + ``.comfy_quant`` JSON config): per-channel de-scale
       then the involutive block-Hadamard un-rotation, reusing the vendored
       comfy-kitchen dequant from ``minimax_h3/comfy_convert.py`` (same format,
       verified byte-level against Comfy-Org/Krea-2 ``int8_convrot``) —
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


def has_int8_convrot(sd: Dict[str, Any]) -> bool:
    """True when the checkpoint carries ComfyUI int8 'ConvRot' layers: an int8
    ``<base>.weight`` next to a ``.comfy_quant`` config or a ``.weight_scale``.
    Gated on the WEIGHT dtype, not the companion keys alone — scaled-fp8 files
    can carry ``.comfy_quant`` entries too. Must be checked BEFORE the
    scaled-fp8 path: the ``.weight_scale`` companions look identical, but int8
    weights also need the block-Hadamard un-rotation — a plain ``int8 * scale``
    would produce silently wrong weights."""
    import torch

    for k in sd:
        if not isinstance(k, str):
            continue
        for suffix in (".comfy_quant", ".weight_scale"):
            if k.endswith(suffix):
                w = sd.get(k[: -len(suffix)] + ".weight")
                if getattr(w, "dtype", None) is torch.int8:
                    return True
    return False


class _DequantWorker:
    """Streams per-tensor dequant math through an accelerator when one is
    given: ONE tensor at a time is moved to ``device``, dequantized there, and
    the result comes back to CPU in the target dtype — peak device memory stays
    under ~1 GB (largest Krea 2 layer ≈ 100 MB quantized + fp32 intermediate),
    never the whole 12.9B model. Any device error (OOM included) permanently
    falls back to the CPU path for the remaining tensors — correctness never
    depends on the device."""

    def __init__(self, device: "Any" = None) -> None:
        self.device = None if device in (None, "cpu") else device
        self.fell_back = False

    def run(self, fn, *tensors):
        """``fn(*tensors_on_work_device)`` -> result moved back to CPU."""
        if self.device is not None and not self.fell_back:
            try:
                return fn(*(t.to(self.device) for t in tensors)).cpu()
            except ValueError:
                raise  # unsupported-format errors are not device failures
            except Exception as e:  # noqa: BLE001 — CPU fallback beats a hard fail
                logger.warning(
                    "Krea 2 dequant on %s failed (%s); falling back to CPU",
                    self.device, e)
                self.fell_back = True
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return fn(*tensors)


def _strip_key_prefix(key: str) -> str:
    """Strip the ComfyUI all-in-one prefix from ONE key (header metadata keys
    may or may not carry it, independently of the tensor keys)."""
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def parse_header_quant_layers(quant_metadata: Any) -> Dict[str, dict]:
    """Per-layer quant configs from a safetensors ``__metadata__`` dict.

    Comfy-Org files carry a per-layer ``.comfy_quant`` TENSOR; third-party
    converters (e.g. "ComfyUI Kitchen NVFP4 Converter", used by several civitai
    int8 finetunes) instead put ONE ``_quantization_metadata`` JSON blob in the
    file header: ``{"layers": {"blocks.0.attn.wq": {"format": ...}, ...}}``.
    ``safetensors.torch.load_file`` DROPS the header — callers must read it via
    ``safe_open(...).metadata()`` and pass it through. Returns ``{}`` when
    absent/unparseable (never raises: a quantized layer without any config is
    rejected later, with context).
    """
    import json

    if not isinstance(quant_metadata, dict):
        return {}
    raw = quant_metadata.get("_quantization_metadata")
    if not raw:
        return {}
    try:
        layers = json.loads(raw).get("layers", {})
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning("Krea 2: unparseable _quantization_metadata header (%s)", e)
        return {}
    if not isinstance(layers, dict):
        return {}
    return {
        _strip_key_prefix(k): v
        for k, v in layers.items()
        if isinstance(k, str) and isinstance(v, dict)
    }


def dequantize_int8_convrot(
    sd: Dict[str, Any],
    dtype: "Any",
    device: "Any" = None,
    header_layers: Dict[str, dict] | None = None,
) -> Dict[str, Any]:
    """Dequantize ComfyUI int8 'ConvRot' layers exactly: per-output-channel
    de-scale, then the involutive block-Hadamard un-rotation. The per-layer
    config comes from the layer's ``.comfy_quant`` tensor (Comfy-Org files) or,
    failing that, from ``header_layers`` (third-party converters that put one
    ``_quantization_metadata`` blob in the file header — see
    ``parse_header_quant_layers``). An int8 layer with NO config from either
    source raises: silently skipping it would hand the raw rotated weights to
    the scaled-fp8 path, which multiplies WITHOUT un-rotating — noise renders.
    The math is the vendored comfy-kitchen dequant shared with the MiniMax-H3
    converter; fp32, each result stored as ``dtype`` immediately; ``device``
    streams the per-tensor math through an accelerator (see ``_DequantWorker``).
    Unsupported quant formats (int4/nvfp4) raise with an actionable message.
    """
    import torch

    from imference_engine.minimax_h3.comfy_convert import (
        dequantize_convrot,
        parse_comfy_quant,
    )

    header_layers = header_layers or {}
    weight_keys = [
        k for k, v in sd.items()
        if isinstance(k, str) and k.endswith(".weight")
        and getattr(v, "dtype", None) is torch.int8
    ]
    if not weight_keys:
        return sd
    worker = _DequantWorker(device)
    out = dict(sd)
    count = 0
    for weight_key in weight_keys:
        base = weight_key[: -len(".weight")]
        scale_key, quant_key = base + ".weight_scale", base + ".comfy_quant"
        config = None
        if quant_key in out:
            config = parse_comfy_quant(torch.as_tensor(out[quant_key]))
            del out[quant_key]
        elif base in header_layers:
            config = header_layers[base]
        if config is None or scale_key not in out:
            raise RuntimeError(
                f"Krea 2 checkpoint: int8 layer {weight_key!r} has no "
                "quantization config (no .comfy_quant tensor and no "
                "_quantization_metadata header entry)"
                + ("" if scale_key in out else " / no .weight_scale")
                + ". Refusing to guess — a wrong dequant renders pure noise. "
                "Supported: ComfyUI int8-ConvRot (Comfy-Org or Kitchen-converter "
                "files), scaled-fp8, plain fp8, bf16."
            )
        try:
            out[weight_key] = worker.run(
                lambda w, s: dequantize_convrot(w, s, config, dtype),
                torch.as_tensor(out[weight_key]),
                torch.as_tensor(out[scale_key]),
            )
        except ValueError as e:
            raise RuntimeError(
                f"Krea 2 checkpoint: cannot dequantize {weight_key!r}: {e}"
            ) from e
        del out[scale_key]
        count += 1
    logger.info("Krea 2: dequantized %d int8-ConvRot tensors -> %s%s",
                count, dtype,
                f" (via {device})" if device and not worker.fell_back else "")
    return out


def dequantize_scaled_fp8(
    sd: Dict[str, Any], dtype: "Any", device: "Any" = None
) -> Dict[str, Any]:
    """Dequantize ComfyUI 'scaled fp8' weights: ``w = w_fp8.float() * weight_scale``.

    Each quantized layer stores an fp8 ``<name>.weight`` plus a (scalar)
    ``<name>.weight_scale``; the multiply runs in float32 for precision but each
    result is stored as ``dtype`` immediately, so the ~12.9B model never peaks
    at fp32 size (~50 GB) in host RAM. ``device`` streams the per-tensor math
    through an accelerator (see ``_DequantWorker``). No-op when there are no
    scale keys.
    """
    import torch

    scale_keys = [k for k in sd if isinstance(k, str) and k.endswith(".weight_scale")]
    if not scale_keys:
        return sd
    worker = _DequantWorker(device)
    out = dict(sd)
    for scale_key in scale_keys:
        weight_key = scale_key[: -len(".weight_scale")] + ".weight"
        if weight_key in out:
            out[weight_key] = worker.run(
                lambda w, s: (w.float() * s.float()).to(dtype),
                torch.as_tensor(out[weight_key]),
                torch.as_tensor(out[scale_key]),
            )
        del out[scale_key]
    logger.info("Krea 2: dequantized %d scaled-fp8 tensors -> %s%s",
                len(scale_keys), dtype,
                f" (via {device})" if device and not worker.fell_back else "")
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


def prepare_krea2_state_dict(
    sd: Dict[str, Any],
    dtype: "Any",
    device: "Any" = None,
    quant_metadata: Any = None,
) -> Tuple[Dict[str, Any], bool]:
    """Full normalization pipeline for a Krea 2 single-file state dict.

    ``device`` (optional) streams the per-tensor dequant math through an
    accelerator — one tensor at a time, <1 GB peak, CPU fallback on any device
    error (see ``_DequantWorker``). ``quant_metadata`` (optional) is the
    safetensors ``__metadata__`` dict from ``safe_open(...).metadata()`` —
    required for third-party int8 files whose quant config lives in the header
    (``load_file`` drops it; see ``parse_header_quant_layers``).

    Returns ``(diffusers-keyed state dict in ``dtype``, source_was_quantized)``
    — True for fp8 (scaled or plain) AND int8-ConvRot sources; the flag drives
    the backend's fp8-resident storage decision (a user shipping a quantized
    file chose the small-footprint trade already).
    """
    import torch

    sd = strip_comfyui_prefix(sd)
    # int8-ConvRot MUST run before the scaled-fp8 path: both formats ship
    # ``.weight_scale`` companions, but int8 weights additionally need the
    # block-Hadamard un-rotation — the fp8 multiply alone would yield silently
    # wrong weights.
    was_int8 = has_int8_convrot(sd)
    if was_int8:
        sd = dequantize_int8_convrot(
            sd, dtype, device,
            header_layers=parse_header_quant_layers(quant_metadata))
    was_fp8 = has_fp8_weights(sd)
    sd = dequantize_scaled_fp8(sd, dtype, device)
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
    # Any int8 tensor left at this point followed an unknown quantization
    # scheme (no .comfy_quant config) — refuse rather than load garbage.
    leftover_int8 = [
        k for k, v in sd.items() if getattr(v, "dtype", None) is torch.int8
    ]
    if leftover_int8:
        raise RuntimeError(
            f"Krea 2 checkpoint: {len(leftover_int8)} int8 tensor(s) without a "
            f"recognized quantization config (first few: {leftover_int8[:5]}). "
            "Supported quantized formats: ComfyUI scaled-fp8, plain fp8, and "
            "int8 ConvRot (.comfy_quant). int4/nvfp4/mxfp8 files are not."
        )
    if is_native_krea2_format(sd):
        sd = convert_krea2_native_to_diffusers(sd)
        logger.info("Krea 2: native/ComfyUI key layout remapped to diffusers (%d tensors)", len(sd))
    return sd, (was_fp8 or was_int8)


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
