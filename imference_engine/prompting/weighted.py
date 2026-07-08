"""Weighted prompt embeddings for SDXL with optional BREAK chunking.

Wraps `sd_embed.get_weighted_text_embeddings_sdxl` to overcome the 77-token
CLIP limit, plus A1111/ComfyUI-style BREAK keyword: split the prompt on BREAK,
encode each chunk independently, concatenate along the sequence dimension.
"""
from __future__ import annotations
import logging
import warnings
from typing import Optional

logger = logging.getLogger(__name__)


def encode_sdxl_weighted(pipe, prompt: str, negative_prompt: Optional[str]) -> dict:
    """Encode prompts using sd_embed weighted embeddings, with BREAK support.

    Returns kwargs suitable for `pipe(**kwargs, ...)`:
        prompt_embeds, negative_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds

    Falls back to raw {prompt, negative_prompt} kwargs if sd_embed is missing —
    loses BREAK + weighting + >77 token support.
    """
    try:
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
    except ImportError:
        logger.warning("sd_embed not installed; using raw prompt strings")
        return {"prompt": prompt, "negative_prompt": negative_prompt or ""}

    import torch
    neg = negative_prompt or ""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

        if "BREAK" in prompt:
            chunks = [c.strip() for c in prompt.split("BREAK") if c.strip()]
            logger.info(f"BREAK keyword: encoding {len(chunks)} CLIP chunks separately")

            chunk_embeds = []
            chunk_neg_embeds = []
            pooled = None
            neg_pooled = None

            for i, chunk in enumerate(chunks):
                # Negative is only attached to the first chunk to avoid amplification
                e, ne, p, np_ = get_weighted_text_embeddings_sdxl(
                    pipe, prompt=chunk, neg_prompt=neg if i == 0 else ""
                )
                chunk_embeds.append(e)
                chunk_neg_embeds.append(ne)
                if i == 0:
                    pooled = p
                    neg_pooled = np_

            prompt_embeds = torch.cat(chunk_embeds, dim=1)
            neg_prompt_embeds = torch.cat(chunk_neg_embeds, dim=1)

            # Pad negative to match positive sequence length
            if neg_prompt_embeds.shape[1] < prompt_embeds.shape[1]:
                pad_size = prompt_embeds.shape[1] - neg_prompt_embeds.shape[1]
                padding = torch.zeros(
                    neg_prompt_embeds.shape[0],
                    pad_size,
                    neg_prompt_embeds.shape[2],
                    dtype=neg_prompt_embeds.dtype,
                    device=neg_prompt_embeds.device,
                )
                neg_prompt_embeds = torch.cat([neg_prompt_embeds, padding], dim=1)
        else:
            (
                prompt_embeds,
                neg_prompt_embeds,
                pooled,
                neg_pooled,
            ) = get_weighted_text_embeddings_sdxl(pipe, prompt=prompt, neg_prompt=neg)

    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": neg_prompt_embeds,
        "pooled_prompt_embeds": pooled,
        "negative_pooled_prompt_embeds": neg_pooled,
    }


def encode_sd15_weighted(pipe, prompt: str, negative_prompt: Optional[str]) -> dict:
    """Weighted prompt embeddings for SD 1.5 (single CLIP-L encoder).

    Unlike SDXL, SD 1.5 has ONE text encoder and NO pooled embeddings, so this
    returns just ``prompt_embeds`` / ``negative_prompt_embeds``. sd_embed's sd15
    path already overcomes the 77-token limit and supports A1111-style weighting;
    BREAK chunking is not wired here (SDXL-only for now).

    Falls back to raw {prompt, negative_prompt} kwargs if sd_embed is missing.
    """
    try:
        from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15
    except ImportError:
        logger.warning("sd_embed not installed; using raw prompt strings")
        return {"prompt": prompt, "negative_prompt": negative_prompt or ""}

    neg = negative_prompt or ""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_sd15(
            pipe, prompt=prompt, neg_prompt=neg
        )
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": neg_prompt_embeds,
    }
