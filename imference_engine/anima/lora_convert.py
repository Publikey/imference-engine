"""Anima LoRA formats diffusers 0.40 does not read, normalized in memory.

diffusers' ``AnimaLoraLoaderMixin`` converts ONE non-diffusers layout: ComfyUI
keys (``diffusion_model.blocks.0.cross_attn.k_proj.lora_A.weight``). Two common
community layouts load NOTHING through it — ``load_lora_weights`` returns
without injecting an adapter, and the failure only surfaces later as
"Adapter name(s) {...} not in the list of present adapters":

- kohya-ss / sd-scripts (``networks.lora_anima``): flattened module names
  (``lora_unet_blocks_0_cross_attn_k_proj``), ``lora_down`` / ``lora_up``
  and a per-module ``.alpha``;
- ComfyUI keys with ``lora_down`` / ``lora_up`` (+ ``.alpha``) instead of
  ``lora_A`` / ``lora_B``.

Both are rewritten into the ComfyUI ``lora_A`` / ``lora_B`` form diffusers
does convert, with the kohya scale ``alpha / rank`` baked into ``lora_B``
(peft then loads the adapter at scale 1, as the trainer intended).
"""
from __future__ import annotations
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# kohya flattened module name (after "lora_unet_") -> original Anima module path.
_KOHYA_MODULES = [
    (re.compile(r"^blocks_(\d+)_(self_attn|cross_attn)_(q_proj|k_proj|v_proj|output_proj)$"),
     r"blocks.\1.\2.\3"),
    (re.compile(r"^blocks_(\d+)_mlp_(layer1|layer2)$"), r"blocks.\1.mlp.\2"),
    (re.compile(r"^blocks_(\d+)_adaln_modulation_(self_attn|cross_attn|mlp)_(1|2)$"),
     r"blocks.\1.adaln_modulation_\2.\3"),
    (re.compile(r"^final_layer_linear$"), r"final_layer.linear"),
    (re.compile(r"^final_layer_adaln_modulation_(1|2)$"), r"final_layer.adaln_modulation.\1"),
    (re.compile(r"^x_embedder_proj_1$"), r"x_embedder.proj.1"),
]


def needs_conversion(keys) -> bool:
    """True for a layout diffusers' Anima loader would silently ignore."""
    return any(k.startswith("lora_unet_") or ".lora_down." in k or ".lora_up." in k for k in keys)


def to_comfy_lora_ab(state_dict: dict) -> dict:
    """kohya / lora_down-up Anima LoRA -> ComfyUI ``diffusion_model.* lora_A/B``.

    Raises ValueError when no module maps (an unknown layout must fail the
    request, not render without the LoRA); unmapped modules among mapped ones
    are dropped with a warning."""
    modules: dict[str, dict[str, Any]] = {}
    for key, value in state_dict.items():
        base, _, leaf = key.partition(".")
        if key.startswith("diffusion_model."):
            # diffusion_model.<path>.lora_down.weight / .alpha
            m = re.match(r"^diffusion_model\.(.+?)\.(lora_down\.weight|lora_up\.weight|lora_A\.weight|lora_B\.weight|alpha)$", key)
            if not m:
                continue
            base, leaf = "diffusion_model." + m.group(1), m.group(2)
        modules.setdefault(base, {})[leaf] = value

    out: dict = {}
    unmapped: list[str] = []
    for base, parts in modules.items():
        path = _module_path(base)
        if path is None:
            unmapped.append(base)
            continue
        down = parts.get("lora_down.weight", parts.get("lora_A.weight"))
        up = parts.get("lora_up.weight", parts.get("lora_B.weight"))
        if down is None or up is None:
            unmapped.append(base)
            continue
        alpha = parts.get("alpha")
        if alpha is not None:
            rank = down.shape[0]
            scale = float(alpha) / rank if rank else 1.0
            if scale != 1.0:
                up = (up.float() * scale).to(up.dtype)
        out[f"diffusion_model.{path}.lora_A.weight"] = down
        out[f"diffusion_model.{path}.lora_B.weight"] = up

    if not out:
        raise ValueError(
            f"Unrecognized Anima LoRA layout: none of its {len(modules)} modules map "
            f"to the Anima DiT (e.g. {sorted(modules)[:3]})")
    if unmapped:
        logger.warning("Anima LoRA: %d module(s) not mapped and skipped (e.g. %s)",
                       len(unmapped), sorted(unmapped)[:3])
    return out


def _module_path(base: str) -> Optional[str]:
    if base.startswith("diffusion_model."):
        return base.removeprefix("diffusion_model.")
    if base.startswith("lora_unet_"):
        name = base.removeprefix("lora_unet_")
        for pattern, repl in _KOHYA_MODULES:
            if pattern.match(name):
                return pattern.sub(repl, name)
    return None
