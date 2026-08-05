"""MiniMax-H3 ComfyUI/civitai single-file checkpoints -> diffusers layout.

The community distributes MiniMax-H3 as ComfyUI-style single ``.safetensors``
files (Comfy-Org/MiniMax-H3 on the Hub, mirrored on civitai): a transformer
partition, a 50-layer-truncated Qwen3-VL text encoder, and the two VAEs —
~67 GB total at int8 versus ~124 GB for the official bf16 repository. This
module converts those files into the diffusers modular layout our loader
consumes. It is a pure-torch library; ``validation/stage_h3_from_comfy.py`` is
the CLI that drives it against a real download.

Three separate concerns live here:

1. **ConvRot int8 dequantization.** The quantized files store, per layer,
   ``<base>.weight`` (int8, block-Hadamard-rotated along the input dim),
   ``<base>.weight_scale`` (fp32 per-output-channel) and ``<base>.comfy_quant``
   (a uint8-encoded JSON config). The Hadamard is the deterministic "regular"
   H4 Kronecker construction, normalized and symmetric — so it is orthogonal
   AND involutive, and dequantization is just ``rotate(int8 * scale)`` with the
   same rotation the quantizer applied. Adapted from Comfy-Org/comfy-kitchen
   (``comfy_kitchen/tensor/int8_utils.py``, Apache-2.0).

2. **Key-layout conversion.** Adapted from diffusers'
   ``scripts/convert_minimax_h3_to_diffusers.py`` (Apache-2.0, ``minimax-h3``
   branch — PR #14355; not importable from an installed diffusers, hence
   vendored). One deliberate difference, verified against the ComfyUI model
   code and the shipped safetensors headers: the Comfy **DiT** repackage stores
   fused QKV already in the reference ``[q_all; k_all; v_all]`` layout (its
   forward splits contiguous thirds), while the Comfy **video-VAE** file keeps
   the raw per-head interleave (its forward chunks per head) — so only the VAE
   path applies the interleave reorder before splitting.

3. **Streaming.** Files are read via ``safe_open`` (memory-mapped) and written
   in bounded shards, so peak RAM stays near the largest single tensor, not the
   34 GB file.

The "pruned" Comfy variants are NOT convertible: they replace the timestep
embedder + full-width AdaLN projections with a low-rank ``adaln_t_table``
lookup — a modified architecture ``MiniMaxH3Transformer3DModel`` cannot load.
``convert_transformer_file`` detects that marker and says so instead of
emitting a broken tree.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ConvRot dequantization (format: Comfy-Org/comfy-kitchen, Apache-2.0)
# ---------------------------------------------------------------------------

_QUANT_SUFFIXES = (".weight_scale", ".comfy_quant")
_HADAMARD_CACHE: dict[int, torch.Tensor] = {}

# The one architecture marker of the "pruned" (low-rank AdaLN) DiT repackages.
PRUNED_MARKER_KEY = "adaln_t_table"


def build_hadamard(size: int) -> torch.Tensor:
    """The normalized *regular* Hadamard matrix ConvRot uses (H4 Kronecker
    powers -> sizes are powers of 4). Symmetric + orthogonal, hence involutive:
    applying `rotate_weight` twice is the identity."""
    cached = _HADAMARD_CACHE.get(size)
    if cached is not None:
        return cached
    if size < 4 or size & (size - 1) or size.bit_length() % 2 == 0:
        # power of 4 <=> power of 2 with an odd bit_length (4 -> 100, 16 -> 10000)
        raise ValueError(f"ConvRot Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[size] = h
    return h


def rotate_weight(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Block-rotate ``weight`` along its input (last) dim: ``W @ H_block^T`` per
    ``group_size`` slice. Because H is symmetric orthogonal this both rotates
    and un-rotates; dequantization calls it on ``int8 * scale``."""
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(
            f"in_features {in_features} not divisible by group_size {group_size}")
    h_t = build_hadamard(group_size).T.to(weight.dtype)
    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    return torch.matmul(grouped, h_t).reshape(out_features, in_features)


def parse_comfy_quant(raw: torch.Tensor) -> dict[str, Any]:
    """Decode a ``<base>.comfy_quant`` uint8 tensor into its JSON config."""
    return json.loads(bytes(raw.tolist()).decode("utf-8"))


def dequantize_convrot(
    weight_int8: torch.Tensor,
    scale: torch.Tensor,
    quant_config: dict[str, Any],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct a ConvRot-quantized weight: per-channel de-scale, then the
    involutive block-Hadamard un-rotation. fp32 math, cast to ``dtype`` last."""
    fmt = quant_config.get("format")
    if fmt != "int8_tensorwise":
        raise ValueError(f"unsupported comfy_quant format {fmt!r} (int8 ConvRot only; "
                         f"int4/nvfp4 files need ComfyUI kernels)")
    w = weight_int8.to(torch.float32) * scale.to(torch.float32)
    if quant_config.get("convrot", False):
        w = rotate_weight(w, int(quant_config.get("convrot_groupsize", 256)))
    return w.to(dtype)


class ComfyCheckpoint:
    """A lazily-read Comfy single-file checkpoint with transparent ConvRot
    dequantization: ``tensor(key)`` returns plain weights, quant sidecar keys
    are folded away from ``keys()``."""

    def __init__(self, path: str, dtype: torch.dtype = torch.bfloat16) -> None:
        from safetensors import safe_open

        self.path = path
        self.dtype = dtype
        self._f = safe_open(path, framework="pt", device="cpu")
        all_keys = list(self._f.keys())
        self._quantized = {k[:-len(".comfy_quant")] for k in all_keys
                           if k.endswith(".comfy_quant")}
        self._keys = [k for k in all_keys if not k.endswith(_QUANT_SUFFIXES)]

    def keys(self) -> list[str]:
        return list(self._keys)

    @property
    def quantized_bases(self) -> set[str]:
        return set(self._quantized)

    def tensor(self, key: str) -> torch.Tensor:
        """The tensor for ``key`` — dequantized to ``self.dtype`` when ConvRot
        sidecars are present, verbatim (source dtype) otherwise."""
        t = self._f.get_tensor(key)
        base = key[:-len(".weight")] if key.endswith(".weight") else None
        if base is not None and base in self._quantized:
            return dequantize_convrot(
                t, self._f.get_tensor(base + ".weight_scale"),
                parse_comfy_quant(self._f.get_tensor(base + ".comfy_quant")),
                self.dtype)
        return t


# ---------------------------------------------------------------------------
# Streaming shard writer (flush pattern from the diffusers convert script)
# ---------------------------------------------------------------------------

class ShardWriter:
    """Accumulate tensors and write ``<stem>-XXXXX-of-XXXXX.safetensors``
    shards + ``<stem>.safetensors.index.json``; a run that fits one shard is
    written as plain ``<stem>.safetensors`` with no index."""

    def __init__(self, out_dir: str, stem: str, max_shard_bytes: int = 5 * 1024 ** 3):
        self.out_dir = out_dir
        self.stem = stem
        self.max_shard_bytes = max_shard_bytes
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self.total_bytes = 0
        self._written: list[str] = []
        self._weight_map: dict[str, str] = {}
        os.makedirs(out_dir, exist_ok=True)

    def add(self, key: str, tensor: torch.Tensor) -> None:
        self._buffer[key] = tensor
        n = tensor.numel() * tensor.element_size()
        self._buffer_bytes += n
        self.total_bytes += n
        if self._buffer_bytes >= self.max_shard_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        from safetensors.torch import save_file

        path = os.path.join(self.out_dir, f".tmp-shard-{len(self._written):05d}.safetensors")
        save_file(self._buffer, path, metadata={"format": "pt"})
        for key in self._buffer:
            self._weight_map[key] = path
        self._written.append(path)
        self._buffer = {}
        self._buffer_bytes = 0

    def finalize(self) -> int:
        """Flush, rename shards to their final names, write the index (unless
        single-shard). Returns the shard count."""
        self._flush()
        if len(self._written) == 1:
            final = os.path.join(self.out_dir, f"{self.stem}.safetensors")
            os.replace(self._written[0], final)
            return 1
        renames = {
            path: os.path.join(
                self.out_dir,
                f"{self.stem}-{i + 1:05d}-of-{len(self._written):05d}.safetensors")
            for i, path in enumerate(self._written)
        }
        for old, new in renames.items():
            os.replace(old, new)
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": {k: os.path.basename(renames[p])
                           for k, p in self._weight_map.items()},
        }
        with open(os.path.join(self.out_dir, f"{self.stem}.safetensors.index.json"),
                  "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        return len(self._written)


# ---------------------------------------------------------------------------
# Transformer (DiT) key conversion
# (adapted from diffusers scripts/convert_minimax_h3_to_diffusers.py, Apache-2.0)
# ---------------------------------------------------------------------------

# Dropped: recomputed by MiniMaxH3RotaryPosEmbed into a non-persistent buffer.
TRANSFORMER_DROPPED_KEYS = ("rope.inv_freq",)

# Original keys that are (and must stay) float32 in the mixed-precision layout.
TRANSFORMER_FP32_PREFIXES = (
    "video_patch_proj.", "audio_patch_proj.", "time_embedder.",
    "final_layer.video_out.", "final_layer.audio_out.",
)


def split_fused_qkv(weight: torch.Tensor, inner_dim: int) -> tuple[torch.Tensor, ...]:
    """Split a fused ``[q_all; k_all; v_all]`` tensor into contiguous thirds
    along dim 0 (weights and biases alike)."""
    if weight.shape[0] != 3 * inner_dim:
        raise ValueError(
            f"fused qkv has {weight.shape[0]} rows, expected 3 * {inner_dim}")
    return tuple(t.contiguous() for t in weight.split(inner_dim, dim=0))


def reorder_interleaved_qkv(
    weight: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Reorder a per-head-interleaved fused QKV (``[h0: q k v, h1: q k v, ...]``)
    into ``[q_all; k_all; v_all]``. Works on weights and 1-D biases."""
    expected = num_heads * 3 * head_dim
    if weight.shape[0] != expected:
        raise ValueError(
            f"fused qkv has {weight.shape[0]} rows, expected "
            f"{expected} = {num_heads} heads * 3 * {head_dim}")
    grouped = weight.reshape(num_heads, 3 * head_dim, *weight.shape[1:])
    q, k, v = grouped.split(head_dim, dim=1)
    return torch.cat(
        [t.reshape(num_heads * head_dim, *weight.shape[1:]) for t in (q, k, v)],
        dim=0)


def convert_transformer_key(
    source_key: str, tensor: torch.Tensor, num_heads: int, head_dim: int
) -> list[tuple[str, torch.Tensor]]:
    """One original-layout DiT key/tensor -> the diffusers key/tensor pair(s).

    Assumes the reference ``[q; k; v]`` fused-QKV layout (what the Comfy DiT
    repackage ships — its forward splits contiguous thirds); raw interleaved
    checkpoints must go through ``reorder_interleaved_qkv`` first.
    """
    if source_key in TRANSFORMER_DROPPED_KEYS:
        return []

    target = source_key
    if target.startswith("token_refiner.blocks."):
        target = target.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif target.startswith("blocks."):
        target = target.replace("blocks.", "transformer_blocks.", 1)
    target = target.replace("time_embedder.proj_in.", "time_embedder.linear_1.")
    target = target.replace("time_embedder.proj_out.", "time_embedder.linear_2.")
    target = target.replace("video_patch_proj.", "proj_in.")
    target = target.replace("audio_patch_proj.", "audio_proj_in.")
    target = target.replace("condition_proj.", "context_embedder.")
    target = target.replace("final_layer.norm.", "norm_out.norm.")
    target = target.replace("final_layer.adaln_proj.linear.", "norm_out.linear.")
    target = target.replace("final_layer.video_out.", "proj_out.")
    target = target.replace("final_layer.audio_out.", "audio_proj_out.")
    target = target.replace(".attn.q_norm.", ".attn.norm_q.")
    target = target.replace(".attn.k_norm.", ".attn.norm_k.")
    target = target.replace(".attn.out_proj.", ".attn.to_out.0.")

    if target.endswith(".attn.qkv_proj.weight"):
        q, k, v = split_fused_qkv(tensor, num_heads * head_dim)
        prefix = target[:-len("qkv_proj.weight")]
        return [(f"{prefix}to_q.weight", q), (f"{prefix}to_k.weight", k),
                (f"{prefix}to_v.weight", v)]

    if target.endswith(".mlp.fc1.weight"):
        # reference SwiGLU stores [gate; value], diffusers' reads [value; gate]
        gate, value = tensor.chunk(2, dim=0)
        target = target.replace(".mlp.fc1.weight", ".ff.net.0.proj.weight")
        return [(target, torch.cat([value, gate], dim=0).contiguous())]

    target = target.replace(".mlp.fc2.", ".ff.net.2.")
    return [(target, tensor)]


def convert_transformer_file(
    src_path: str,
    out_dir: str,
    transformer_config: dict[str, Any],
    max_shard_bytes: int = 5 * 1024 ** 3,
    progress: Optional[Callable[[str], None]] = None,
) -> int:
    """Convert a Comfy DiT single-file into ``out_dir`` diffusers shards.
    Returns the tensor count written. ``transformer_config`` is the official
    ``transformer/config.json`` (for head geometry + layer-count validation)."""
    ckpt = ComfyCheckpoint(src_path, dtype=torch.bfloat16)
    keys = ckpt.keys()

    if PRUNED_MARKER_KEY in keys:
        raise ValueError(
            f"{os.path.basename(src_path)} is a 'pruned' repackage: it replaces "
            f"the timestep embedder + full AdaLN with a low-rank adaln_t_table, "
            f"an architecture MiniMaxH3Transformer3DModel cannot load. Use the "
            f"non-pruned int8_convrot file instead (~34 GB).")

    num_heads = int(transformer_config["num_attention_heads"])
    head_dim = int(transformer_config["attention_head_dim"])
    block_count = _max_block_index(keys, r"^blocks\.(\d+)\.") + 1
    if block_count != int(transformer_config["num_layers"]):
        raise ValueError(
            f"{os.path.basename(src_path)} has {block_count} blocks but the "
            f"official config says {transformer_config['num_layers']} — layout "
            f"drift; refusing to emit a mismatched tree.")

    writer = ShardWriter(out_dir, "diffusion_pytorch_model", max_shard_bytes)
    written = 0
    for i, key in enumerate(keys):
        tensor = ckpt.tensor(key)
        expected = (torch.float32 if key.startswith(TRANSFORMER_FP32_PREFIXES)
                    else torch.bfloat16)
        if key not in TRANSFORMER_DROPPED_KEYS and tensor.dtype != expected:
            raise ValueError(f"{key}: expected {expected}, got {tensor.dtype}")
        for target_key, out in convert_transformer_key(key, tensor, num_heads, head_dim):
            writer.add(target_key, out)
            written += 1
        if progress and i % 100 == 0:
            progress(f"transformer {i}/{len(keys)} source keys")
    shards = writer.finalize()
    logger.info("transformer: %d source keys -> %d diffusers keys in %d shard(s), %.2f GiB",
                len(keys), written, shards, writer.total_bytes / 1024 ** 3)
    return written


def _max_block_index(keys: list[str], pattern: str) -> int:
    rx = re.compile(pattern)
    indices = [int(m.group(1)) for k in keys if (m := rx.match(k))]
    if not indices:
        raise ValueError(f"no keys match {pattern!r} — wrong file?")
    return max(indices)


# ---------------------------------------------------------------------------
# Text encoder (truncated Qwen3-VL) key conversion
# ---------------------------------------------------------------------------

def convert_text_encoder_key(source_key: str) -> str:
    """Comfy TE layout -> ``Qwen3VLForConditionalGeneration`` state dict:
    ``model.*`` is the language model, ``visual.*`` the vision tower."""
    if source_key.startswith("visual."):
        return "model." + source_key
    if source_key.startswith("model."):
        return "model.language_model." + source_key[len("model."):]
    raise ValueError(f"unexpected text-encoder key root: {source_key}")


def convert_text_encoder_file(
    src_path: str,
    out_dir: str,
    official_config: dict[str, Any],
    max_shard_bytes: int = 5 * 1024 ** 3,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Convert the Comfy truncated Qwen3-VL single-file into ``out_dir`` shards
    and return the patched ``config.json`` dict to write next to them.

    The Comfy repackage keeps only the decoder layers H3 actually reads (the
    conditioner taps the raw ``hidden_states[N]`` of an N=50-layer prefix) and
    drops the final norm + LM head. A checkpoint truncated to *exactly* N
    layers cannot reproduce that conditioning through transformers: the last
    ``hidden_states`` entry of a full stack is the POST-norm output, and the
    upstream H3 encoder guards against it. So the staged checkpoint gets N+1
    layers: the real N, plus one **dummy layer** (ones for its norms, ~1e-6
    noise for its projections — near-identity through the residual path, and
    unlike zeros it quantizes safely). ``hidden_states[N]`` is then the dummy
    layer's *input* = the raw output of real layer N-1, exactly the H3
    contract; everything the dummy layer feeds (final norm, lm_head) is never
    read. The config is patched to N+1 layers, the final RMSNorm is
    materialized (weights = 1) so the checkpoint is complete, and
    ``tie_word_embeddings`` is set so no untrained lm_head is allocated.
    """
    ckpt = ComfyCheckpoint(src_path, dtype=torch.bfloat16)
    keys = ckpt.keys()
    num_layers = _max_block_index(keys, r"^model\.layers\.(\d+)\.") + 1

    writer = ShardWriter(out_dir, "model", max_shard_bytes)
    hidden_size = None
    layer0_shapes: dict[str, tuple] = {}
    for i, key in enumerate(keys):
        tensor = ckpt.tensor(key)
        if key == "model.embed_tokens.weight":
            hidden_size = tensor.shape[1]
        if key.startswith("model.layers.0."):
            layer0_shapes[key[len("model.layers.0."):]] = tuple(tensor.shape)
        writer.add(convert_text_encoder_key(key), tensor)
        if progress and i % 200 == 0:
            progress(f"text_encoder {i}/{len(keys)} source keys")

    if hidden_size is None:
        raise ValueError("model.embed_tokens.weight missing — wrong file?")

    # Dummy layer N (see docstring). Deterministic: fixed-seed generator.
    gen = torch.Generator().manual_seed(0)
    for suffix, shape in sorted(layer0_shapes.items()):
        if len(shape) == 1:
            if "norm" not in suffix:
                raise ValueError(
                    f"unexpected non-norm 1-D layer tensor {suffix!r} — cannot "
                    f"synthesize a safe dummy value")
            dummy = torch.ones(shape, dtype=torch.bfloat16)
        else:
            dummy = torch.randn(shape, generator=gen) * 1e-6
            dummy = dummy.to(torch.bfloat16)
        writer.add(f"model.language_model.layers.{num_layers}.{suffix}", dummy)

    if "model.norm.weight" not in keys:
        writer.add("model.language_model.norm.weight",
                   torch.ones(hidden_size, dtype=torch.bfloat16))
    shards = writer.finalize()
    logger.info("text_encoder: %d source keys in %d shard(s), %.2f GiB, "
                "%d real layers + 1 dummy",
                len(keys), shards, writer.total_bytes / 1024 ** 3, num_layers)

    config = json.loads(json.dumps(official_config))  # deep copy
    text_cfg = config.get("text_config", config)
    if num_layers >= int(text_cfg.get("num_hidden_layers", num_layers + 1)):
        raise ValueError(
            f"checkpoint has {num_layers} decoder layers, official config only "
            f"{text_cfg.get('num_hidden_layers')} — wrong file?")
    text_cfg["num_hidden_layers"] = num_layers + 1
    # No lm_head in the checkpoint and H3 never calls it: tie it to the
    # embeddings so from_pretrained does not allocate ~1.5 GB of random rows.
    config["tie_word_embeddings"] = True
    text_cfg["tie_word_embeddings"] = True
    return config


# ---------------------------------------------------------------------------
# Video VAE key conversion (Comfy file keeps the raw per-head interleave)
# ---------------------------------------------------------------------------

VIDEO_VAE_DROPPED_KEYS = ("decoder.mask_token",)


def _rename_video_vae_key(source_key: str) -> str:
    target = source_key
    if target.startswith("encoder.down."):
        level, rest = target[len("encoder.down."):].split(".", 1)
        rest = rest.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        rest = rest.replace("downsample.", "downsamplers.0.", 1)
        target = f"encoder.down_blocks.{level}.{rest}"
    target = target.replace("decoder.x_embedder.", "decoder.proj_in.")
    target = target.replace(".attn.to_out.", ".attn.to_out.0.")
    target = target.replace(".ff.w1.", ".ff.net.0.proj.")
    target = target.replace(".ff.w2.", ".ff.net.2.")
    return target


def convert_video_vae_key(
    source_key: str, tensor: torch.Tensor, num_heads: int, head_dim: int
) -> list[tuple[str, torch.Tensor]]:
    """One original-layout video-VAE key/tensor -> diffusers pair(s)."""
    if source_key in VIDEO_VAE_DROPPED_KEYS:
        return []

    if ".attn.to_qkv." in source_key:
        reordered = reorder_interleaved_qkv(tensor, num_heads, head_dim)
        q, k, v = split_fused_qkv(reordered, num_heads * head_dim)
        prefix, suffix = source_key.split(".attn.to_qkv.")
        return [(f"{prefix}.attn.to_q.{suffix}", q),
                (f"{prefix}.attn.to_k.{suffix}", k),
                (f"{prefix}.attn.to_v.{suffix}", v)]

    target = _rename_video_vae_key(source_key)
    if ".ff.w1." in source_key:
        gate, up = tensor.chunk(2, dim=0)
        return [(target, torch.cat([up, gate], dim=0).contiguous())]
    return [(target, tensor)]


def convert_video_vae_file(
    src_path: str,
    out_dir: str,
    vae_config: dict[str, Any],
    max_shard_bytes: int = 5 * 1024 ** 3,
) -> int:
    """Convert the Comfy video-VAE single-file (fp16, kept as-is — the engine
    loads components in bf16 anyway) into ``out_dir``. Returns tensors written."""
    ckpt = ComfyCheckpoint(src_path)
    num_heads = int(vae_config["decoder_num_attention_heads"])
    head_dim = int(vae_config["decoder_attention_head_dim"])

    writer = ShardWriter(out_dir, "diffusion_pytorch_model", max_shard_bytes)
    written = 0
    for key in ckpt.keys():
        for target_key, out in convert_video_vae_key(
                key, ckpt.tensor(key), num_heads, head_dim):
            writer.add(target_key, out)
            written += 1
    shards = writer.finalize()
    logger.info("vae: %d diffusers keys in %d shard(s), %.2f GiB",
                written, shards, writer.total_bytes / 1024 ** 3)
    return written


# ---------------------------------------------------------------------------
# Audio VAE: identity mapping + weight_norm resynthesis
# ---------------------------------------------------------------------------

def resynthesize_weight_norm(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused conv weight back into ``torch.nn.utils.weight_norm``'s
    ``(weight_g, weight_v)`` pair (dim=0 convention). Exact: with ``v = w`` and
    ``g = ||w||`` over the non-output dims, ``g * v / ||v|| == w``."""
    g = torch.norm_except_dim(weight, 2, 0)
    return g, weight


# Stored as tensors in the Comfy audio-VAE file, but carried in the diffusers
# model's CONFIG (latents_mean / latents_std land in audio_vae/config.json from
# the official skeleton) — dropped from the state dict.
AUDIO_VAE_DROPPED_KEYS = ("latents_mean", "latents_std")


def convert_audio_vae_file(src_path: str, out_dir: str, expected_keys: set[str]) -> int:
    """Convert the Comfy audio-VAE single-file (fp32, ComfyUI fuses the
    weight-norm parametrization) back into the ``weight_g`` / ``weight_v``
    layout ``AutoencoderKLMiniMaxH3Audio`` expects. ``expected_keys`` is the
    state-dict key set of a freshly built model (the caller instantiates it —
    this module stays diffusers-free). Returns tensors written."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    normed_bases = {k[:-len(".weight_g")] for k in expected_keys
                    if k.endswith(".weight_g")}

    out: dict[str, torch.Tensor] = {}
    with safe_open(src_path, framework="pt", device="cpu") as f:
        source_keys = set(f.keys())
        for key in sorted(source_keys):
            if key in AUDIO_VAE_DROPPED_KEYS:
                continue
            tensor = f.get_tensor(key)
            base = key[:-len(".weight")] if key.endswith(".weight") else None
            if base is not None and base in normed_bases:
                g, v = resynthesize_weight_norm(tensor)
                out[f"{base}.weight_g"] = g
                out[f"{base}.weight_v"] = v
            elif key in expected_keys:
                out[key] = tensor
            else:
                raise KeyError(
                    f"unexpected audio-VAE key {key!r} (not in the model's "
                    f"state dict and not a fused weight-norm weight)")

    missing = sorted(expected_keys - set(out))
    if missing:
        raise KeyError(
            f"{len(missing)} audio-VAE key(s) unaccounted for, e.g. {missing[:5]}")

    os.makedirs(out_dir, exist_ok=True)
    save_file(out, os.path.join(out_dir, "diffusion_pytorch_model.safetensors"),
              metadata={"format": "pt"})
    logger.info("audio_vae: %d keys (weight-norm resynthesized for %d modules)",
                len(out), len(normed_bases))
    return len(out)
