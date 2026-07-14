"""Weighted prompt embeddings for SDXL with optional BREAK chunking.

Wraps `sd_embed.get_weighted_text_embeddings_sdxl` to overcome the 77-token
CLIP limit, plus A1111/ComfyUI-style BREAK keyword: split the prompt on BREAK,
encode each chunk independently, concatenate along the sequence dimension.

VRAM note — why the encoding runs under ``torch.no_grad()``:
    sd_embed runs the CLIP text encoders with autograd ENABLED, and we call it
    OUTSIDE ``pipe.__call__`` (which diffusers decorates ``@torch.no_grad()``, so
    the pipeline's own ``encode_prompt`` never builds a graph). Without a no_grad
    context here, the returned embeddings carry a ``.grad_fn`` that pins the
    encoders' CUDA weight tensors (saved-for-backward) plus intermediate
    activations — and since those embeddings are held for the entire denoise
    loop, ~1.85 GB stays resident on the GPU the whole time. Under
    ``enable_model_cpu_offload`` on an 8 GB card that tips peak VRAM past capacity
    into WDDM shared-memory spill (measured: baseline 0 -> 1981 MB, ~50x slower).
    It's a retained autograd graph, not stranded weights: moving an encoder to
    CPU frees almost nothing because the graph still references the original cuda
    tensors. Encoding never needs gradients, so a no_grad context drops the graph
    entirely, lets the offload hooks reclaim the encoders normally, and keeps
    both sd_embed weighting/BREAK and model_cpu_offload's speed.
"""
from __future__ import annotations
import logging
import warnings
from typing import Optional

logger = logging.getLogger(__name__)


def _log_encode_vram(where: str) -> None:
    """One-line INFO of current GPU allocation after a weighted encode, so the
    sidecar log shows the baseline stayed low (no retained-graph residue)."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("weighted-encode(%s): VRAM allocated %.0f MB",
                        where, torch.cuda.memory_allocated() / 1e6)
    except Exception:  # noqa: BLE001
        pass


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

    # no_grad is load-bearing here — see the module docstring. Without it the
    # returned embeds keep a grad graph that pins ~1.85 GB of encoder tensors on
    # the GPU for the whole denoise loop and defeats enable_model_cpu_offload.
    with torch.no_grad(), warnings.catch_warnings():
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

    _log_encode_vram("sdxl")

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

    import torch
    neg = negative_prompt or ""
    # no_grad for the same reason as SDXL — keep the encoder off the autograd graph
    # so offload can reclaim it (see module docstring).
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")
        prompt_embeds, neg_prompt_embeds = get_weighted_text_embeddings_sd15(
            pipe, prompt=prompt, neg_prompt=neg
        )
    _log_encode_vram("sd15")
    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": neg_prompt_embeds,
    }
