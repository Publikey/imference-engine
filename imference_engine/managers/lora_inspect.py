"""Identify which model family a LoRA file targets — header-only, no torch.

Reads the safetensors header (8-byte length + JSON: tensor names, shapes,
``__metadata__``) without touching the tensor data, so it is cheap enough to
run on every load and safe to call from a UI import flow.

Detection order, most to least trustworthy:

1. **Trainer metadata** — kohya ``ss_base_model_version`` (``sdxl_base_v1-0``,
   ``sd_v1``, ``flux1``…) and the modelspec ``modelspec.architecture``
   (``stable-diffusion-xl-v1-base/lora``…).
2. **Key layout** — SDXL has a second text encoder (``lora_te2_`` /
   ``text_encoder_2.``) and kohya names its UNet in the ldm style
   (``lora_unet_input_blocks_…``) where SD1.x uses diffusers names
   (``lora_unet_down_blocks_…``); FLUX-style DiTs have ``double_blocks`` /
   ``single_blocks``.
3. **Cross-attention width** — the input dim of an ``attn2.to_k`` down
   projection is the text-embedding width: 2048 SDXL, 768 SD1.x, 1024 SD2.x.

Returns ``None`` when nothing matches: unknown layouts (LoKr factors, DiT
families whose key layout is not mapped here yet) are let through and left for
diffusers to accept or reject — a guess would block valid files.
"""
from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# A real header is a few hundred KB; anything past this is not a safetensors file.
_MAX_HEADER_BYTES = 100 * 1024 * 1024

# Families a LoRA is detected as. "flux" covers the FLUX-derived DiTs (Chroma
# LoRAs share the FLUX block layout).
SDXL, SD15, SD2, FLUX = "sdxl", "sd15", "sd2", "flux"

_CROSS_ATTN_WIDTH = {2048: SDXL, 768: SD15, 1024: SD2}


@dataclass
class LoraInfo:
    family: Optional[str]
    """Detected target family, or None when unknown."""
    reason: str
    """What the detection was based on (for logs / UI tooltips)."""
    metadata: dict = field(default_factory=dict)
    """The file's ``__metadata__`` block (trainer info, trigger words, …)."""


def read_safetensors_header(path: str) -> dict:
    """Return the parsed JSON header of a ``.safetensors`` file.

    Raises ``ValueError`` for files that are not safetensors (pickled ``.pt`` /
    ``.ckpt`` LoRAs included — those are refused rather than unpickled).
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path!r} is not a safetensors file (too short)")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > _MAX_HEADER_BYTES:
            raise ValueError(f"{path!r} is not a safetensors file (bad header length)")
        blob = f.read(n)
    try:
        header = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path!r} is not a safetensors file ({e})") from e
    if not isinstance(header, dict):
        raise ValueError(f"{path!r} is not a safetensors file (header is not an object)")
    return header


def inspect_lora(path: str) -> LoraInfo:
    """Detect the family a LoRA file was trained for (see module docstring)."""
    header = read_safetensors_header(path)
    metadata = header.get("__metadata__") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    keys = [k for k in header if k != "__metadata__"]

    family = _family_from_metadata(metadata)
    if family:
        return LoraInfo(family, "trainer metadata", metadata)
    family = _family_from_keys(keys)
    if family:
        return LoraInfo(family, "key layout", metadata)
    family = _family_from_cross_attention(header, keys)
    if family:
        return LoraInfo(family, "cross-attention width", metadata)
    return LoraInfo(None, "unrecognized layout", metadata)


def is_compatible(lora_family: Optional[str], backend_engine: str) -> bool:
    """True unless the LoRA was positively identified for another family."""
    return lora_family is None or lora_family == backend_engine


def _family_from_metadata(md: dict) -> Optional[str]:
    base = str(md.get("ss_base_model_version") or "").lower()
    if base.startswith("sdxl"):
        return SDXL
    if base.startswith("sd_v1"):
        return SD15
    if base.startswith("sd_v2"):
        return SD2
    if base.startswith("flux"):
        return FLUX

    arch = str(md.get("modelspec.architecture") or "").lower()
    if arch.startswith("stable-diffusion-xl"):
        return SDXL
    if arch.startswith("stable-diffusion-v1"):
        return SD15
    if arch.startswith("stable-diffusion-v2"):
        return SD2
    if arch.startswith("flux"):
        return FLUX
    return None


def _family_from_keys(keys: list[str]) -> Optional[str]:
    if any(k.startswith("lora_te2_") or "text_encoder_2." in k for k in keys):
        return SDXL
    if any(k.startswith(("lora_unet_input_blocks_", "lora_unet_output_blocks_",
                         "lora_unet_middle_block_")) for k in keys):
        return SDXL
    if any("double_blocks" in k or "single_blocks" in k
           or "single_transformer_blocks" in k for k in keys):
        return FLUX
    return None


def _family_from_cross_attention(header: dict, keys: list[str]) -> Optional[str]:
    for k in keys:
        if "attn2" not in k or "to_k" not in k:
            continue
        if not (k.endswith("lora_down.weight") or k.endswith("lora_A.weight")):
            continue
        shape = (header.get(k) or {}).get("shape") or []
        if len(shape) >= 2:
            return _CROSS_ATTN_WIDTH.get(int(shape[1]))
    return None
