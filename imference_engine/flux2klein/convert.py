"""Single-file normalization for FLUX.2 [klein] transformer checkpoints.

BFL ships the klein transformer as native-key single files in three flavors:

- **bf16** (``flux-2-klein-4b.safetensors``) — diffusers' own
  ``from_single_file`` converts it, nothing to do here;
- **scaled fp8** (``flux-2-klein-4b-fp8.safetensors``, the official
  half-size build) — the big linear weights are ``float8_e4m3fn`` with a
  per-tensor ``<layer>.weight_scale`` (and an ``<layer>.input_scale`` meant for
  activation quantization at inference). diffusers 0.40 has no path for it, so
  ``dequantize_scaled_fp8`` rebuilds the bf16 state dict IN MEMORY
  (``weight * weight_scale``; ``input_scale`` dropped — compute stays bf16);
- **nvfp4** (``…-nvfp4.safetensors``, NVIDIA 4-bit float, RTX 50 only) — no
  generic dequant: refused loudly rather than loaded as noise.

``detect_format`` reads only the safetensors JSON header (no torch, no tensor
I/O), so the format check is unit-testable without a GPU. Nothing is rewritten
on disk.
"""
from __future__ import annotations

import json
import struct
from typing import Any

# Header metadata key BFL / ComfyUI write on quantized single files.
QUANT_METADATA_KEY = "_quantization_metadata"

_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")


def read_safetensors_header(path: str) -> dict:
    """The JSON header of a ``.safetensors`` file: ``{tensor_name: {dtype,
    shape, data_offsets}, "__metadata__": {...}}``."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def detect_format(header: dict) -> str:
    """Classify a klein transformer single file from its header.

    Returns ``"bf16"`` (plain float weights), ``"fp8-scaled"`` (float8 weights
    + ``weight_scale``), ``"fp8"`` (float8 weights, no scale tensors) or
    ``"nvfp4"``.
    """
    meta = header.get("__metadata__") or {}
    quant = meta.get(QUANT_METADATA_KEY) or ""
    names = [k for k in header if k != "__metadata__"]
    if '"nvfp4"' in quant or any(k.endswith(".weight_scale_2") for k in names):
        return "nvfp4"
    has_fp8 = any(header[k].get("dtype") == "F8_E4M3" for k in names)
    if not has_fp8:
        return "bf16"
    return "fp8-scaled" if any(k.endswith(".weight_scale") for k in names) else "fp8"


def dequantize_scaled_fp8(sd: dict, dtype: Any) -> dict:
    """Rebuild a float state dict from BFL's scaled-fp8 single file.

    ``<layer>.weight`` (float8_e4m3fn) is multiplied by ``<layer>.weight_scale``
    in fp32 and cast to ``dtype``; the ``*_scale`` helper tensors are dropped.
    Tensors that are not float8 are only cast. Keys stay in the native layout
    (diffusers' flux2 converter remaps them afterwards).
    """
    import torch

    out: dict = {}
    for key, value in sd.items():
        if key.endswith(_SCALE_SUFFIXES):
            continue
        if value.dtype == torch.float8_e4m3fn:
            scale = sd.get(key[: -len("weight")] + "weight_scale") if key.endswith(".weight") else None
            value = value.to(torch.float32)
            if scale is not None:
                value = value * scale.to(torch.float32)
        out[key] = value.to(dtype)
    return out
