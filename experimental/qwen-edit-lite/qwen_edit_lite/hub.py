"""Resolve model files: local paths, URLs, or files picked in a Hugging Face repo.

Default repos (all Apache-2.0):
    base       Qwen/Qwen-Image-Edit-2511            (diffusers format)
    gguf       unsloth/Qwen-Image-Edit-2511-GGUF    (Q2_K .. Q8_0, BF16)
    lightning  lightx2v/Qwen-Image-Edit-2511-Lightning (4-step LoRA)

File names inside repos change between uploads, so they are matched by
pattern (quant tag, "4steps", "bf16") against the live listing rather than
hard-coded. Override any repo with env QEL_BASE_REPO / QEL_GGUF_REPO /
QEL_LIGHTNING_REPO.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Optional

BASE_REPO = os.environ.get("QEL_BASE_REPO", "Qwen/Qwen-Image-Edit-2511")
GGUF_REPO = os.environ.get("QEL_GGUF_REPO", "unsloth/Qwen-Image-Edit-2511-GGUF")
LIGHTNING_REPO = os.environ.get("QEL_LIGHTNING_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning")

QUANT_RE = re.compile(r"^(Q[2-8](_[0K](_[SML])?)?|IQ\d\w*|BF16|F16)$", re.I)


class ResolveError(RuntimeError):
    pass


def list_files(repo: str) -> list:
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(repo)


def pick_gguf(files: list, quant: str) -> str:
    """The .gguf file for ``quant`` (e.g. Q5_K_M). Exact tag match, case-insensitive,
    tag delimited by -, _ or . so Q5_K does not match Q5_K_M."""
    q = quant.upper()
    cands = [f for f in files if f.lower().endswith(".gguf")
             and re.search(rf"(^|[-_.]){re.escape(q)}([-_.]|$)", os.path.basename(f).upper().replace(".GGUF", ".END"))]
    if not cands:
        avail = sorted({m.group(0) for f in files if f.lower().endswith(".gguf")
                        for m in [re.search(r"(Q\d(_[0K](_[SML])?)?|BF16|F16|IQ\w+)", os.path.basename(f).upper())] if m})
        raise ResolveError(f"no {quant} .gguf in the repo; available: {avail}")
    # Prefer the shortest name (main file over e.g. "-mmproj" companions) at the repo root.
    return sorted(cands, key=lambda f: (f.count("/"), len(f)))[0]


def pick_lightning(files: list, steps: int) -> str:
    st = [f for f in files if f.endswith(".safetensors") and re.search(rf"{steps}[-_ ]?steps?", f, re.I)]
    # A LoRA, not a fused checkpoint: skip the fp8 / fused full-model files.
    lora = [f for f in st if not re.search(r"fp8|e4m3|scaled|fused|merged", f, re.I)]
    if not lora:
        raise ResolveError(f"no {steps}-step Lightning LoRA in the repo (files: {[f for f in files if f.endswith('.safetensors')]})")
    # Prefer bf16 over fp32 (half the download, same result after cast).
    lora.sort(key=lambda f: (0 if "bf16" in f.lower() else 1, f.count("/"), len(f)))
    return lora[0]


def download(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename)


def resolve_gguf(arg: str, *, lister: Callable = list_files, fetch: Callable = download) -> str:
    if os.path.isfile(arg) or arg.startswith(("http://", "https://")):
        return arg
    if QUANT_RE.match(arg):
        return fetch(GGUF_REPO, pick_gguf(lister(GGUF_REPO), arg))
    if "/" in arg and arg.count("/") >= 2:            # repo/owner/file.gguf
        owner, name, rest = arg.split("/", 2)
        return fetch(f"{owner}/{name}", rest)
    raise ResolveError(f"cannot resolve GGUF {arg!r}: give a quant tag (Q5_K_M), a path, a URL or owner/repo/file.gguf")


def resolve_lightning(arg: str, *, lister: Callable = list_files, fetch: Callable = download) -> Optional[str]:
    if arg == "none":
        return None
    if os.path.isfile(arg):
        return arg
    if arg in ("4", "8"):
        return fetch(LIGHTNING_REPO, pick_lightning(lister(LIGHTNING_REPO), int(arg)))
    if arg.count("/") >= 2:
        owner, name, rest = arg.split("/", 2)
        return fetch(f"{owner}/{name}", rest)
    raise ResolveError(f"cannot resolve Lightning {arg!r}")


# Always needed: configs, tokenizer/processor, scheduler, VAE (~250 MB).
BASE_PATTERNS = ["model_index.json", "scheduler/*", "tokenizer/*", "processor/*", "vae/*",
                 "transformer/config.json", "text_encoder/config.json", "*.md"]


def base_patterns(*, transformer_weights: bool, encoder_weights: bool) -> list:
    pats = list(BASE_PATTERNS)
    if transformer_weights:
        pats.append("transformer/*")      # ~41 GB bf16: only for bf16 / fp8 / torchao presets
    if encoder_weights:
        pats.append("text_encoder/*")     # ~16.6 GB bf16: once, then the NF4 copy is cached
    return pats


def base_snapshot(repo: str, *, transformer_weights: bool, encoder_weights: bool, fetch=None) -> str:
    """Local directory of the base repo with only the needed files. A local
    directory is returned as-is."""
    if os.path.isdir(repo):
        return repo
    if fetch is None:
        from huggingface_hub import snapshot_download
        fetch = snapshot_download
    return fetch(repo, allow_patterns=base_patterns(transformer_weights=transformer_weights,
                                                    encoder_weights=encoder_weights))
