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
