"""Loading the validated Wan recipe: GGUF experts + shared components + LoRA-no-fuse.

Encodes exactly what spikes P1d/P1e/P1g proved:
- GGUF Q8 transformers via ``from_single_file`` (high → transformer, low → transformer_2);
- one shared UMT5 ``text_encoder`` + Wan ``vae`` reused across variants (P1g);
- Lightning LoRA applied with ``set_adapters`` and **never fused** (dodges #12047);
- ``enable_model_cpu_offload`` so VRAM stays ~one expert regardless of residency.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SharedComponents:
    """Big components shared across variants of the same base (loaded once).

    P1g confirmed the UMT5 text_encoder + Wan VAE are interchangeable between the
    T2V and I2V pipelines, so one set can serve every variant.
    """

    base_repo: str
    text_encoder: Any
    tokenizer: Any
    vae: Any


# Files needed from a diffusers base repo. NEVER the transformer weights (we use
# GGUF). The shared base also needs the UMT5 text_encoder + tokenizer + VAE.
_SHARED_PATTERNS = ["model_index.json", "scheduler/*", "text_encoder/*",
                    "tokenizer/*", "vae/*",
                    "transformer/config.json", "transformer_2/config.json"]
_BASE_CFG_PATTERNS = ["model_index.json", "scheduler/*",
                      "transformer/config.json", "transformer_2/config.json"]


def _flat_root(cache_dir: Optional[str]) -> str:
    """Root of the FLAT, symlink-free model tree (<root>/<repo>/<file>).

    Unified for the base configs AND the GGUF/LoRA. Defaults next to the HF cache
    on the big volume; overridable via cache_dir (WAN_MODEL_CACHE)."""
    import os
    if cache_dir:
        return cache_dir
    hf = os.environ.get("HF_HOME")
    return os.path.join(hf, "wan") if hf else os.path.join(
        os.path.expanduser("~"), ".cache", "wan")


def _local_repo_dir(repo: str, patterns: list, cache_dir: Optional[str]) -> str:
    """Ensure a base repo's needed files are in a FLAT local dir and return it.

    Uses ``snapshot_download(local_dir=...)``, which writes real files (NO
    symlinks) on every OS -> the tree is portable Windows<->Mac<->Linux, unlike
    the symlinked HF cache. Idempotent; offline once populated (from_pretrained
    then just reads local files)."""
    import os
    from huggingface_hub import snapshot_download
    d = os.path.join(_flat_root(cache_dir), repo)
    snapshot_download(repo, allow_patterns=patterns, local_dir=d)
    return d


def load_shared_components(base_repo: str, *, cache_dir: Optional[str] = None) -> SharedComponents:
    import torch
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    d = _local_repo_dir(base_repo, _SHARED_PATTERNS, cache_dir)
    logger.info("Loading shared components (text_encoder + vae) from %s", d)
    text_encoder = UMT5EncoderModel.from_pretrained(
        d, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(d, subfolder="tokenizer")
    vae = AutoencoderKLWan.from_pretrained(
        d, subfolder="vae", torch_dtype=torch.float32)
    return SharedComponents(base_repo, text_encoder, tokenizer, vae)



def _cdn_download(cdn_base: str, repo: str, filename: str, cache_dir: Optional[str]) -> str:
    """Download <cdn_base>/<repo>/<filename> to a local cache (skip if present).

    The CDN must mirror the HF repo/filename layout. Uses urllib (not
    huggingface_hub), so it works alongside HF_HUB_OFFLINE=1.
    """
    import os
    import urllib.error
    import urllib.request

    # Same FLAT tree as the base configs: <root>/<repo>/<file>. Unified so one
    # shippable, symlink-free tree holds everything (base + GGUF + LoRA).
    dest = os.path.join(_flat_root(cache_dir), repo, filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        import concurrent.futures
        import sys
        import threading
        import time
        url = f"{cdn_base.rstrip('/')}/{repo}/{filename}"
        logger.info("  CDN fetch %s", url)
        tmp = dest + ".part"
        label = os.path.basename(filename)
        # Browser-like UA: Cloudflare 403s the default "Python-urllib/x" agent.
        ua = "Mozilla/5.0 (imference-engine wan)"
        nstreams = max(1, int(os.environ.get("WAN_CDN_THREADS", "8")))

        def _head_total() -> int:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": ua})
            with urllib.request.urlopen(req) as r:
                return int(r.headers.get("Content-Length") or 0)

        last_err = None
        for attempt in range(1, 4):
            try:
                total = _head_total()
                if total <= 0:
                    raise IOError("CDN gave no Content-Length")
                got = [0]
                lock = threading.Lock()
                fd = os.open(tmp, os.O_CREAT | os.O_WRONLY, 0o644)
                try:
                    os.ftruncate(fd, total)

                    def _range(start: int, end: int) -> None:
                        req = urllib.request.Request(
                            url, headers={"User-Agent": ua, "Range": f"bytes={start}-{end}"})
                        with urllib.request.urlopen(req) as r:
                            off = start
                            while True:
                                buf = r.read(4 * 1024 * 1024)
                                if not buf:
                                    break
                                os.pwrite(fd, buf, off)  # thread-safe positional write
                                off += len(buf)
                                with lock:
                                    got[0] += len(buf)

                    # Parallel range requests saturate the link (a single GET is slow).
                    size = (total + nstreams - 1) // nstreams
                    spans = [(i * size, min((i + 1) * size - 1, total - 1))
                             for i in range(nstreams) if i * size < total]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(spans)) as ex:
                        futs = [ex.submit(_range, s, e) for s, e in spans]
                        while not all(f.done() for f in futs):
                            d = got[0]
                            print(f"\r  CDN {label}: {d//1024//1024}/{total//1024//1024} MiB "
                                  f"({d*100//total}%) [{len(spans)} streams]   ",
                                  end="", file=sys.stderr, flush=True)
                            time.sleep(0.5)
                        for f in futs:
                            f.result()  # surface any worker error
                    print("", file=sys.stderr, flush=True)
                finally:
                    os.close(fd)
                if os.path.getsize(tmp) != total:
                    raise IOError(f"incomplete: {os.path.getsize(tmp)} of {total} bytes")
                os.replace(tmp, dest)
                return dest
            except (urllib.error.URLError, IOError) as e:
                last_err = e
                logger.warning("  CDN attempt %d/3 failed (%s); retrying", attempt, e)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                time.sleep(2 * attempt)
        raise RuntimeError(f"CDN download failed for {url}: {last_err}")
    return dest


def _fetch(repo: str, filename: str, cdn_base: Optional[str], cache_dir: Optional[str]) -> str:
    """Local path for a model file — into the FLAT tree, from the CDN if set else HF.

    Both paths land at <flat_root>/<repo>/<filename> (no symlinks), so the whole
    tree is portable + shippable + offline once present."""
    if cdn_base:
        return _cdn_download(cdn_base, repo, filename, cache_dir)
    import os
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo, filename, local_dir=os.path.join(_flat_root(cache_dir), repo))


def _resolve_gguf(
    repo: str, quant: str, which: str,
    explicit_name: Optional[str] = None, template: Optional[str] = None,
    cdn_base: Optional[str] = None, cache_dir: Optional[str] = None,
) -> str:
    """Return a local path to the high/low GGUF for a quant level.

    Resolution order:
      1. ``explicit_name`` — exact path (quant-fixed).
      2. ``template`` with a ``{quant}`` placeholder — DETERMINISTIC, no API call.
      3. online auto-discovery via ``list_repo_files`` (HF only; needs network).

    The resolved file is fetched from ``cdn_base`` if set, else HF. CDN mode needs
    a name/template (no listing over a CDN).
    """
    name = explicit_name or (template.format(quant=quant) if template else None)
    if name:
        logger.info("  %s [%s] -> %s%s", repo, which, name, " (cdn)" if cdn_base else "")
        return _fetch(repo, name, cdn_base, cache_dir)
    if cdn_base:
        raise ValueError(
            f"{repo}: CDN mode needs gguf_*_name/template for the {which} expert")

    from huggingface_hub import list_repo_files
    files = [f for f in list_repo_files(repo) if f.endswith(".gguf")]
    cand = [f for f in files
            if which in f.lower() and f.lower().endswith(f"-{quant.lower()}.gguf")]
    if not cand:
        cand = [f for f in files if which in f.lower() and quant.lower() in f.lower()]
    if not cand:
        raise FileNotFoundError(
            f"no '{which}' '{quant}' .gguf in {repo}; available: {files}")
    logger.info("  %s [%s] -> %s", repo, which, cand[0])
    return _fetch(repo, cand[0], None, cache_dir)


def _load_gguf_transformer(path: str, base_repo: str, subfolder: str) -> Any:
    import torch
    from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

    try:
        return WanTransformer3DModel.from_single_file(
            path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            config=base_repo,
            subfolder=subfolder,
        )
    except ValueError as e:
        if "patch_embedding" in str(e):
            raise ValueError(
                f"GGUF at {path} stores patch_embedding in a ComfyUI-flattened shape "
                "diffusers can't load (e.g. BigDannyPt conversion). Use a known-good "
                "converter (QuantStack / Bedovyy) or a bf16 source. See doc P1e."
            ) from e
        raise


def _apply_loras(pipe: Any, loras: list,
                 cdn_base: Optional[str] = None, cache_dir: Optional[str] = None) -> None:
    """Stack LoRAs onto the (possibly quantized) experts with set_adapters — no fuse.

    Validated on GGUF in P1d: each LoRA's high file → transformer, low file →
    transformer_2 (load_into_transformer_2=True); set_adapters keeps them as live
    bf16 adapters over the quantized weights. The LoRA file is fetched into the
    flat tree (CDN or HF) and loaded from that local path.
    """
    names: list[str] = []
    weights: list[float] = []
    for i, lora in enumerate(loras):
        for wn, weight, into2, suffix in (
            (lora.high_weight_name, lora.high_weight, False, "high"),
            (lora.low_weight_name, lora.low_weight, True, "low"),
        ):
            if not wn:
                continue
            adapter = f"lora{i}_{suffix}"
            path = _fetch(lora.repo, wn, cdn_base, cache_dir)
            pipe.load_lora_weights(path, adapter_name=adapter,
                                   load_into_transformer_2=into2)
            names.append(adapter)
            weights.append(weight)
    if names:
        pipe.set_adapters(names, adapter_weights=weights)
        logger.info("  applied %d LoRA adapter(s), not fused: %s", len(names), names)


def build_pipeline(
    variant,
    *,
    quant: str,
    shared: SharedComponents,
    device: str,
    enable_offload: bool,
    vae_tiling: bool,
    cache_dir: Optional[str] = None,
    cdn_base: Optional[str] = None,
) -> Any:
    """Build a ready-to-run Wan pipeline for ``variant`` at GGUF ``quant``.

    ``cdn_base`` (env WAN_MODEL_CDN) makes the GGUF experts + LoRAs come from a CDN
    mirror instead of HF — the shared base still loads from the (cached) HF repo.
    """
    import torch
    from diffusers import (UniPCMultistepScheduler, WanImageToVideoPipeline,
                           WanPipeline)

    logger.info("Building variant %r (%s, gguf=%s)", variant.name, variant.mode, quant)
    # base configs (model_index, scheduler, transformer configs) in a FLAT dir —
    # also used as the GGUF transformer `config=` so loading is fully local/offline.
    base_dir = _local_repo_dir(variant.base_repo, _BASE_CFG_PATTERNS, cache_dir)
    high = _resolve_gguf(variant.gguf_repo, quant, "high",
                         variant.gguf_high_name, variant.gguf_high_template,
                         cdn_base=cdn_base, cache_dir=cache_dir)
    low = _resolve_gguf(variant.gguf_repo, quant, "low",
                        variant.gguf_low_name, variant.gguf_low_template,
                        cdn_base=cdn_base, cache_dir=cache_dir)
    transformer = _load_gguf_transformer(high, base_dir, "transformer")
    transformer_2 = _load_gguf_transformer(low, base_dir, "transformer_2")

    pipe_cls = WanPipeline if variant.mode == "t2v" else WanImageToVideoPipeline
    pipe = pipe_cls.from_pretrained(
        base_dir,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=shared.vae,
        text_encoder=shared.text_encoder,
        tokenizer=shared.tokenizer,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=variant.flow_shift)

    if variant.lightning_baked:
        logger.info("  lightning baked in checkpoint — no LoRA applied")
    else:
        _apply_loras(pipe, variant.loras, cdn_base=cdn_base, cache_dir=cache_dir)

    if enable_offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    if vae_tiling:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    return pipe
