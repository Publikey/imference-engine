"""fp8 storage for text encoders (float8_e4m3fn weights, bf16 compute).

Halves an encoder's footprint with diffusers' layerwise casting. One catch
makes a helper worth having: diffusers pipelines read ``text_encoder.dtype``
(the dtype of the model's FIRST parameter) to cast the prompt embeddings and,
through them, to create the latents. With every weight in float8 that dtype is
float8 — lossy embeddings, and ``torch.randn`` cannot sample float8 at all.
Keeping the module that owns the first parameter (in practice the token
embedding) in bf16 keeps ``.dtype`` at bf16 for a few hundred MB.
"""
from __future__ import annotations

import re
from typing import Any


def apply_fp8_storage(model: Any, *, keep: tuple = ("embed_tokens", "norm")) -> Any:
    """Store ``model``'s weights in float8_e4m3fn, computing in bf16.

    ``keep`` are module-name patterns left in bf16 (norm layers, the token
    embedding). The module holding the first parameter is always kept, so
    ``model.dtype`` still reports bf16. Returns ``model``.
    """
    import torch
    from diffusers.hooks import apply_layerwise_casting

    first = next(iter(model.named_parameters()))[0].rsplit(".", 1)[0]
    patterns = tuple(dict.fromkeys((*keep, re.escape(first))))
    apply_layerwise_casting(
        model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
        skip_modules_pattern=patterns)
    return model


def resolve_storage(env_var: str, recommended: str, vram_gb: Any) -> str:
    """``"bf16"`` or ``"fp8"`` for one component of a storage build.

    The ``env_var`` override (``bf16`` | ``fp8``; anything else = auto) wins,
    then the build table's ``recommended`` value. fp8 storage needs CUDA
    (float8 kernels): without a VRAM reading the answer is always bf16.
    """
    import os

    if vram_gb is None:
        return "bf16"
    setting = os.environ.get(env_var, "auto").strip().lower()
    if setting in ("bf16", "fp8"):
        return setting
    return recommended or "bf16"


def fp8_transformer(transformer: Any, label: str, logger: Any) -> Any:
    """Store a diffusers transformer in float8_e4m3fn with bf16 compute
    (diffusers' own skip patterns keep norms and embeddings in bf16). Falls
    back to bf16 with a warning if layerwise casting is unavailable."""
    import torch

    try:
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        logger.info("%s transformer: fp8-resident storage enabled", label)
    except Exception as e:  # noqa: BLE001 — bf16 beats a hard fail
        logger.warning("%s fp8-resident storage unavailable (%s); staying bf16", label, e)
    return transformer
