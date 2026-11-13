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

Two pairs of families share a key layout and are told apart by tensor shapes:
Lumina-Image 2.0 / Z-Image (hidden size 2304 vs 3840) and FLUX.1 / FLUX.2 klein
(block count and single-block projection width).

Returns ``None`` when nothing matches: unknown layouts (LoKr factors, DiT
families whose key layout is not mapped here yet) are let through and left for
diffusers to accept or reject — a guess would block valid files.
"""
from __future__ import annotations

import json
import re
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
ZIMAGE, KREA2, ANIMA = "zimage", "krea2", "anima"
FLUX2_KLEIN, LUMINA2 = "flux2-klein", "lumina2"

# Z-Image inherited Lumina-Image 2.0's block names (layers / context_refiner /
# noise_refiner), so the key layout alone cannot tell their LoRAs apart: the
# width of a down projection can (the transformer's hidden size).
_LUMINA_FAMILY_WIDTH = {2304: LUMINA2, 3840: ZIMAGE}
# FLUX.2 klein reuses FLUX.1's double/single block names. FLUX.1 has 19 double
# and 38 single blocks and a 15360-wide single-block projection input; klein 4B
# has 5 and 20 (klein 9B: 8 and 24) and a 12288-wide one.
_FLUX1_SINGLE_IN = 15360
_KLEIN_MAX_DOUBLE, _KLEIN_MAX_SINGLE = 8, 24
_BLOCK_INDEX = re.compile(
    r"(double_blocks|single_blocks|transformer_blocks|single_transformer_blocks)[._](\d+)[._]")

# Key layouts of the DiT families (original / ComfyUI / kohya / diffusers
# spellings, from diffusers' lora_conversion_utils). Checked before the UNet
# heuristics: an Anima LoRA in diffusers format has attn2.to_k too.
_ZIMAGE_KEYS = re.compile(r"(noise_refiner|context_refiner|(^|[._])layers[._]\d+[._](attention|feed_forward)[._])")
_KREA2_KEYS = re.compile(r"(txtfusion\.|text_fusion\.|\.attn\.(wq|wk|wv|wo|to_gate)\.|(^|\.)(tmlp|txtmlp)\.)")
_ANIMA_KEYS = re.compile(
    r"(llm_adapter\.|text_conditioner\.|adaln_modulation_(self_attn|cross_attn|mlp)|"
    r"\.(self_attn|cross_attn)\.(q_proj|k_proj|v_proj|output_proj)\.|transformer_blocks\.\d+\.norm[123]\.linear_[12])"
)
_UNET_KEYS = ("down_blocks", "up_blocks", "mid_block", "input_blocks", "output_blocks", "middle_block")

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
    family = _family_from_keys(keys, header)
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
    if base.startswith("flux2") or base.startswith("flux.2") or "klein" in base:
        return FLUX2_KLEIN
    if base.startswith("flux"):
        return FLUX
    if base.startswith("lumina"):
        return LUMINA2
    for prefix, fam in (("zimage", ZIMAGE), ("z_image", ZIMAGE), ("z-image", ZIMAGE),
                        ("krea", KREA2), ("anima", ANIMA)):
        if base.startswith(prefix):
            return fam

    arch = str(md.get("modelspec.architecture") or "").lower()
    if arch.startswith("stable-diffusion-xl"):
        return SDXL
    if arch.startswith("stable-diffusion-v1"):
        return SD15
    if arch.startswith("stable-diffusion-v2"):
        return SD2
    if arch.startswith(("flux-2", "flux2", "flux.2")) or "klein" in arch:
        return FLUX2_KLEIN
    if arch.startswith("flux"):
        return FLUX
    if "lumina" in arch:
        return LUMINA2
    for needle, fam in (("z-image", ZIMAGE), ("zimage", ZIMAGE), ("krea", KREA2), ("anima", ANIMA)):
        if needle in arch:
            return fam
    return None


def _down_widths(header: Optional[dict], keys: list[str]) -> set:
    """Input widths of the LoRA down projections (``lora_A`` / ``lora_down``)."""
    widths: set = set()
    for k in keys:
        if not k.endswith(("lora_A.weight", "lora_down.weight")):
            continue
        shape = ((header or {}).get(k) or {}).get("shape") or []
        if len(shape) >= 2:
            widths.add(int(shape[1]))
    return widths


def _flux_generation(keys: list[str], widths: set) -> str:
    """FLUX.1 vs FLUX.2 klein for a LoRA in the double/single block layout."""
    if any("to_qkv_mlp_proj" in k for k in keys):     # FLUX.2-only module name
        return FLUX2_KLEIN
    if _FLUX1_SINGLE_IN in widths:
        return FLUX
    double = single = -1
    for k in keys:
        m = _BLOCK_INDEX.search(k)
        if not m:
            continue
        idx = int(m.group(2))
        if m.group(1) in ("double_blocks", "transformer_blocks"):
            double = max(double, idx)
        else:
            single = max(single, idx)
    if double >= _KLEIN_MAX_DOUBLE or single >= _KLEIN_MAX_SINGLE:
        return FLUX
    # Fits klein's block count. A FLUX.1 LoRA trained on its first blocks only
    # would too, so the width of the single-block projection settles it when
    # the file has one (12288 on klein 4B).
    if 12288 in widths or 16384 in widths:
        return FLUX2_KLEIN
    return FLUX


def _family_from_keys(keys: list[str], header: Optional[dict] = None) -> Optional[str]:
    if any(_ZIMAGE_KEYS.search(k) or re.search(r"(^|[._])layers[._]\d+[._]attn[._]", k)
           for k in keys):
        widths = _down_widths(header, keys)
        for width, fam in _LUMINA_FAMILY_WIDTH.items():
            if width in widths:
                return fam
        return ZIMAGE
    for pattern, fam in ((_KREA2_KEYS, KREA2), (_ANIMA_KEYS, ANIMA)):
        if any(pattern.search(k) for k in keys):
            return fam
    if any(k.startswith("lora_te2_") or "text_encoder_2." in k for k in keys):
        return SDXL
    if any(k.startswith(("lora_unet_input_blocks_", "lora_unet_output_blocks_",
                         "lora_unet_middle_block_")) for k in keys):
        return SDXL
    if any("double_blocks" in k or "single_blocks" in k
           or "single_transformer_blocks" in k for k in keys):
        return _flux_generation(keys, _down_widths(header, keys))
    return None


def _family_from_cross_attention(header: dict, keys: list[str]) -> Optional[str]:
    for k in keys:
        # UNet cross-attention only: DiTs have attn2.to_k too, with other widths.
        if "attn2" not in k or "to_k" not in k or not any(u in k for u in _UNET_KEYS):
            continue
        if not (k.endswith("lora_down.weight") or k.endswith("lora_A.weight")):
            continue
        shape = (header.get(k) or {}).get("shape") or []
        if len(shape) >= 2:
            return _CROSS_ATTN_WIDTH.get(int(shape[1]))
    return None
