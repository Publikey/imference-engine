"""GPU-free tests for minimax_h3/comfy_convert.py — ConvRot dequantization
(validated against a reference reimplementation of the Comfy quantizer recipe),
key-layout conversion, and tiny end-to-end file conversions."""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from imference_engine.minimax_h3 import comfy_convert as cc  # noqa: E402


# ---------------------------------------------------------------------------
# ConvRot math
# ---------------------------------------------------------------------------

def _reference_quantize(w: torch.Tensor, gs: int):
    """The Comfy quantizer recipe (absmax variant): fp32 upcast, block-Hadamard
    rotation, per-output-channel absmax int8."""
    wr = cc.rotate_weight(w.to(torch.float32), gs)
    scale = (wr.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-30)
    q = (wr / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def test_hadamard_is_symmetric_orthogonal():
    for size in (4, 16, 64, 256):
        h = cc.build_hadamard(size)
        assert torch.equal(h, h.T), size
        assert torch.allclose(h @ h.T, torch.eye(size), atol=1e-5), size


def test_rotate_weight_is_involutive():
    w = torch.randn(6, 32)
    assert torch.allclose(cc.rotate_weight(cc.rotate_weight(w, 16), 16), w, atol=1e-5)


@pytest.mark.parametrize("size", [0, 2, 8, 32, 128, 100])
def test_hadamard_rejects_non_power_of_4(size):
    with pytest.raises(ValueError):
        cc.build_hadamard(size)


def test_rotate_weight_rejects_indivisible_in_features():
    with pytest.raises(ValueError):
        cc.rotate_weight(torch.randn(4, 30), 16)


def test_convrot_roundtrip_reconstructs_weight():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    q, scale = _reference_quantize(w, 16)
    deq = cc.dequantize_convrot(
        q, scale, {"format": "int8_tensorwise", "convrot": True,
                   "convrot_groupsize": 16},
        dtype=torch.float32)
    cos = torch.nn.functional.cosine_similarity(deq.flatten(), w.flatten(), dim=0)
    relerr = (deq - w).norm() / w.norm()
    assert cos > 0.999
    assert relerr < 0.05


def test_dequantize_rejects_unknown_format():
    with pytest.raises(ValueError, match="int4"):
        cc.dequantize_convrot(torch.zeros(4, 16, dtype=torch.int8),
                              torch.ones(4, 1),
                              {"format": "convrot_w4a4", "convrot_groupsize": 16})


def test_parse_comfy_quant_roundtrip():
    cfg = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}
    raw = torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)
    assert cc.parse_comfy_quant(raw) == cfg


def _quant_triplet(base: str, w: torch.Tensor, gs: int) -> dict:
    q, scale = _reference_quantize(w, gs)
    cfg = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": gs}
    return {
        f"{base}.weight": q,
        f"{base}.weight_scale": scale,
        f"{base}.comfy_quant": torch.tensor(
            list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8),
    }


def test_comfy_checkpoint_folds_quant_sidecars(tmp_path):
    torch.manual_seed(1)
    w = torch.randn(8, 32)
    tensors = {"plain.weight": torch.randn(4, 4, dtype=torch.bfloat16),
               **_quant_triplet("blocks.0.attn.out_proj", w, 16)}
    path = str(tmp_path / "ckpt.safetensors")
    save_file(tensors, path)

    ckpt = cc.ComfyCheckpoint(path)
    assert sorted(ckpt.keys()) == ["blocks.0.attn.out_proj.weight", "plain.weight"]
    assert ckpt.quantized_bases == {"blocks.0.attn.out_proj"}
    deq = ckpt.tensor("blocks.0.attn.out_proj.weight")
    assert deq.dtype == torch.bfloat16
    assert (deq.float() - w).norm() / w.norm() < 0.05
    # passthrough keeps its stored dtype
    assert ckpt.tensor("plain.weight").dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Transformer key conversion
# ---------------------------------------------------------------------------

HEADS, HEAD_DIM = 2, 4
INNER = HEADS * HEAD_DIM


def _convert(key, tensor):
    return cc.convert_transformer_key(key, tensor, HEADS, HEAD_DIM)


def test_transformer_renames():
    t = torch.zeros(1)
    cases = {
        "video_patch_proj.weight": "proj_in.weight",
        "audio_patch_proj.bias": "audio_proj_in.bias",
        "condition_proj.weight": "context_embedder.weight",
        "time_embedder.proj_in.weight": "time_embedder.linear_1.weight",
        "time_embedder.proj_out.bias": "time_embedder.linear_2.bias",
        "final_layer.norm.weight": "norm_out.norm.weight",
        "final_layer.adaln_proj.linear.weight": "norm_out.linear.weight",
        "final_layer.video_out.weight": "proj_out.weight",
        "final_layer.audio_out.bias": "audio_proj_out.bias",
        "blocks.3.norm1.weight": "transformer_blocks.3.norm1.weight",
        "blocks.3.attn.q_norm.weight": "transformer_blocks.3.attn.norm_q.weight",
        "blocks.3.attn.k_norm.weight": "transformer_blocks.3.attn.norm_k.weight",
        "blocks.3.attn.out_proj.weight": "transformer_blocks.3.attn.to_out.0.weight",
        "blocks.3.mlp.fc2.weight": "transformer_blocks.3.ff.net.2.weight",
        "blocks.3.adaln_proj.linear.bias": "transformer_blocks.3.adaln_proj.linear.bias",
        "token_refiner.blocks.1.mlp.fc2.weight":
            "token_refiner.refiner_blocks.1.ff.net.2.weight",
        "token_refiner.final_norm.weight": "token_refiner.final_norm.weight",
    }
    for src, expected in cases.items():
        out = _convert(src, t)
        assert [k for k, _ in out] == [expected], src


def test_transformer_drops_rope_inv_freq():
    assert _convert("rope.inv_freq", torch.zeros(16)) == []


def test_transformer_qkv_split_contiguous_thirds():
    q = torch.full((INNER, 8), 1.0)
    k = torch.full((INNER, 8), 2.0)
    v = torch.full((INNER, 8), 3.0)
    fused = torch.cat([q, k, v], dim=0)
    out = dict(_convert("blocks.0.attn.qkv_proj.weight", fused))
    assert torch.equal(out["transformer_blocks.0.attn.to_q.weight"], q)
    assert torch.equal(out["transformer_blocks.0.attn.to_k.weight"], k)
    assert torch.equal(out["transformer_blocks.0.attn.to_v.weight"], v)


def test_transformer_fc1_swaps_swiglu_halves():
    gate = torch.full((3, 2), 1.0)
    value = torch.full((3, 2), 2.0)
    out = _convert("blocks.0.mlp.fc1.weight", torch.cat([gate, value], dim=0))
    [(key, tensor)] = out
    assert key == "transformer_blocks.0.ff.net.0.proj.weight"
    assert torch.equal(tensor, torch.cat([value, gate], dim=0))


def test_reorder_interleaved_qkv_inverts_interleave():
    torch.manual_seed(2)
    q = torch.randn(INNER, 8)
    k = torch.randn(INNER, 8)
    v = torch.randn(INNER, 8)
    # interleave per head: [h0: q k v, h1: q k v]
    per_head = [
        torch.cat([t[h * HEAD_DIM:(h + 1) * HEAD_DIM] for t in (q, k, v)], dim=0)
        for h in range(HEADS)
    ]
    interleaved = torch.cat(per_head, dim=0)
    assert torch.equal(cc.reorder_interleaved_qkv(interleaved, HEADS, HEAD_DIM),
                       torch.cat([q, k, v], dim=0))


# ---------------------------------------------------------------------------
# Tiny end-to-end file conversions
# ---------------------------------------------------------------------------

TINY_TRANSFORMER_CONFIG = {
    "num_attention_heads": HEADS,
    "attention_head_dim": HEAD_DIM,
    "num_layers": 2,
}


def _tiny_transformer_tensors() -> dict:
    hidden = 16
    bf16, f32 = torch.bfloat16, torch.float32
    tensors = {
        "video_patch_proj.weight": torch.randn(hidden, 96, dtype=f32),
        "video_patch_proj.bias": torch.randn(hidden, dtype=f32),
        "time_embedder.proj_in.weight": torch.randn(hidden, 16, dtype=f32),
        "final_layer.video_out.weight": torch.randn(96, hidden, dtype=f32),
        "condition_proj.weight": torch.randn(hidden, 16, dtype=bf16),
        "rope.inv_freq": torch.randn(16, dtype=f32),
        "token_refiner.final_norm.weight": torch.randn(hidden, dtype=bf16),
    }
    for i in range(2):
        tensors[f"blocks.{i}.norm1.weight"] = torch.randn(hidden, dtype=bf16)
        tensors[f"blocks.{i}.attn.qkv_proj.weight"] = torch.randn(
            3 * INNER, hidden, dtype=bf16)
        tensors[f"blocks.{i}.mlp.fc1.weight"] = torch.randn(32, hidden, dtype=bf16)
        tensors[f"blocks.{i}.mlp.fc2.weight"] = torch.randn(hidden, 16, dtype=bf16)
    # one ConvRot-quantized layer, to exercise dequant inside the streamer
    quant_w = torch.randn(hidden, INNER)
    tensors.update(_quant_triplet("blocks.1.attn.out_proj", quant_w, 4))
    tensors["blocks.0.attn.out_proj.weight"] = torch.randn(hidden, INNER, dtype=bf16)
    return tensors


def test_convert_transformer_file_tiny(tmp_path):
    torch.manual_seed(3)
    src = str(tmp_path / "dit.safetensors")
    save_file(_tiny_transformer_tensors(), src)
    out_dir = str(tmp_path / "transformer")

    cc.convert_transformer_file(src, out_dir, TINY_TRANSFORMER_CONFIG)

    from safetensors import safe_open
    out_path = os.path.join(out_dir, "diffusion_pytorch_model.safetensors")
    with safe_open(out_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        assert "proj_in.weight" in keys
        assert "transformer_blocks.0.attn.to_q.weight" in keys
        assert "transformer_blocks.1.attn.to_out.0.weight" in keys
        assert "transformer_blocks.1.ff.net.0.proj.weight" in keys
        assert "rope.inv_freq" not in keys
        assert not any(k.endswith((".weight_scale", ".comfy_quant")) for k in keys)
        assert f.get_tensor("proj_in.weight").dtype == torch.float32
        assert f.get_tensor("transformer_blocks.1.attn.to_out.0.weight").dtype == torch.bfloat16


def test_convert_transformer_file_refuses_pruned(tmp_path):
    tensors = _tiny_transformer_tensors()
    tensors["adaln_t_table"] = torch.randn(1025, 8, dtype=torch.float32)
    src = str(tmp_path / "dit_pruned.safetensors")
    save_file(tensors, src)
    with pytest.raises(ValueError, match="pruned"):
        cc.convert_transformer_file(src, str(tmp_path / "out"), TINY_TRANSFORMER_CONFIG)


def test_convert_transformer_file_refuses_layer_mismatch(tmp_path):
    src = str(tmp_path / "dit.safetensors")
    save_file(_tiny_transformer_tensors(), src)
    config = {**TINY_TRANSFORMER_CONFIG, "num_layers": 50}
    with pytest.raises(ValueError, match="blocks"):
        cc.convert_transformer_file(src, str(tmp_path / "out"), config)


def test_convert_text_encoder_file_tiny(tmp_path):
    torch.manual_seed(4)
    bf16 = torch.bfloat16
    hidden = 8
    tensors = {
        "model.embed_tokens.weight": torch.randn(32, hidden, dtype=bf16),
        "model.layers.0.input_layernorm.weight": torch.randn(hidden, dtype=bf16),
        "model.layers.1.input_layernorm.weight": torch.randn(hidden, dtype=bf16),
        "visual.patch_embed.proj.weight": torch.randn(4, 3, 2, 2, 2, dtype=bf16),
    }
    tensors.update(_quant_triplet(
        "model.layers.0.mlp.down_proj", torch.randn(hidden, 16), 4))
    tensors.update(_quant_triplet(
        "model.layers.1.mlp.down_proj", torch.randn(hidden, 16), 4))
    src = str(tmp_path / "te.safetensors")
    save_file(tensors, src)
    out_dir = str(tmp_path / "text_encoder")

    official = {"tie_word_embeddings": False,
                "text_config": {"num_hidden_layers": 64}}
    patched = cc.convert_text_encoder_file(src, out_dir, official)

    # 2 real layers + 1 dummy so hidden_states[2] stays the raw layer-1 output
    assert patched["text_config"]["num_hidden_layers"] == 3
    assert patched["tie_word_embeddings"] is True
    assert patched["text_config"]["tie_word_embeddings"] is True
    assert official["text_config"]["num_hidden_layers"] == 64  # input untouched

    from safetensors import safe_open
    with safe_open(os.path.join(out_dir, "model.safetensors"),
                   framework="pt", device="cpu") as f:
        keys = set(f.keys())
        assert "model.language_model.embed_tokens.weight" in keys
        assert "model.language_model.layers.1.mlp.down_proj.weight" in keys
        assert "model.visual.patch_embed.proj.weight" in keys
        norm = f.get_tensor("model.language_model.norm.weight")
        assert torch.equal(norm, torch.ones(hidden, dtype=bf16))
        # dummy layer 2 mirrors layer 0's tensor set: ones norms, ~0 projections
        dummy_norm = f.get_tensor("model.language_model.layers.2.input_layernorm.weight")
        assert torch.equal(dummy_norm, torch.ones(hidden, dtype=bf16))
        dummy_proj = f.get_tensor("model.language_model.layers.2.mlp.down_proj.weight")
        assert dummy_proj.shape == (hidden, 16)
        assert dummy_proj.float().abs().max() < 1e-4 and dummy_proj.abs().sum() > 0


def test_convert_text_encoder_rejects_unknown_root():
    with pytest.raises(ValueError):
        cc.convert_text_encoder_key("lm_head.weight")


# ---------------------------------------------------------------------------
# Video VAE
# ---------------------------------------------------------------------------

def test_video_vae_renames_and_transforms():
    t = torch.zeros(1)
    heads, head_dim = 2, 4
    cases = {
        "encoder.down.2.block.1.norm1.weight":
            "encoder.down_blocks.2.resnets.1.norm1.weight",
        "encoder.down.0.block.1.nin_shortcut.weight":
            "encoder.down_blocks.0.resnets.1.conv_shortcut.weight",
        "encoder.down.3.downsample.conv.bias":
            "encoder.down_blocks.3.downsamplers.0.conv.bias",
        "decoder.x_embedder.weight": "decoder.proj_in.weight",
        "decoder.transformer_blocks.5.attn.to_out.weight":
            "decoder.transformer_blocks.5.attn.to_out.0.weight",
        "decoder.transformer_blocks.5.ff.w2.weight":
            "decoder.transformer_blocks.5.ff.net.2.weight",
        "quant_conv.weight": "quant_conv.weight",
    }
    for src, expected in cases.items():
        out = cc.convert_video_vae_key(src, t, heads, head_dim)
        assert [k for k, _ in out] == [expected], src

    assert cc.convert_video_vae_key("decoder.mask_token", t, heads, head_dim) == []


def test_video_vae_qkv_deinterleaves_then_splits():
    torch.manual_seed(5)
    heads, head_dim = 2, 4
    inner = heads * head_dim
    q, k, v = (torch.randn(inner, 8) for _ in range(3))
    per_head = [
        torch.cat([t[h * head_dim:(h + 1) * head_dim] for t in (q, k, v)], dim=0)
        for h in range(heads)
    ]
    interleaved = torch.cat(per_head, dim=0)
    out = dict(cc.convert_video_vae_key(
        "decoder.transformer_blocks.0.attn.to_qkv.weight", interleaved,
        heads, head_dim))
    assert torch.equal(out["decoder.transformer_blocks.0.attn.to_q.weight"], q)
    assert torch.equal(out["decoder.transformer_blocks.0.attn.to_k.weight"], k)
    assert torch.equal(out["decoder.transformer_blocks.0.attn.to_v.weight"], v)


def test_video_vae_ff_w1_swaps_halves():
    gate = torch.full((3, 2), 1.0)
    up = torch.full((3, 2), 2.0)
    out = cc.convert_video_vae_key(
        "decoder.transformer_blocks.0.ff.w1.weight",
        torch.cat([gate, up], dim=0), 2, 4)
    [(key, tensor)] = out
    assert key == "decoder.transformer_blocks.0.ff.net.0.proj.weight"
    assert torch.equal(tensor, torch.cat([up, gate], dim=0))


# ---------------------------------------------------------------------------
# Audio VAE weight-norm resynthesis
# ---------------------------------------------------------------------------

def test_resynthesize_weight_norm_is_exact():
    torch.manual_seed(6)
    w = torch.randn(4, 3, 5)
    g, v = cc.resynthesize_weight_norm(w)
    reconstructed = g * v / torch.norm_except_dim(v, 2, 0)
    assert torch.allclose(reconstructed, w, atol=1e-6)
    assert g.shape == (4, 1, 1)


def test_convert_audio_vae_file_tiny(tmp_path):
    torch.manual_seed(7)
    fused = torch.randn(4, 3, 5)
    tensors = {
        "decoder.conv_pre.weight": fused,
        "decoder.conv_pre.bias": torch.randn(4),
        "pre_block.norm1.weight": torch.randn(8),
        # config-carried stats the Comfy file also stores as tensors — dropped
        "latents_mean": torch.randn(16),
        "latents_std": torch.randn(16),
    }
    src = str(tmp_path / "audio.safetensors")
    save_file(tensors, src)
    out_dir = str(tmp_path / "audio_vae")

    expected = {"decoder.conv_pre.weight_g", "decoder.conv_pre.weight_v",
                "decoder.conv_pre.bias", "pre_block.norm1.weight"}
    n = cc.convert_audio_vae_file(src, out_dir, expected)
    assert n == 4

    from safetensors import safe_open
    with safe_open(os.path.join(out_dir, "diffusion_pytorch_model.safetensors"),
                   framework="pt", device="cpu") as f:
        assert set(f.keys()) == expected
        g = f.get_tensor("decoder.conv_pre.weight_g")
        v = f.get_tensor("decoder.conv_pre.weight_v")
        assert torch.allclose(g * v / torch.norm_except_dim(v, 2, 0), fused, atol=1e-6)


def test_convert_audio_vae_file_rejects_unexpected_key(tmp_path):
    save_file({"stray.weight": torch.randn(2, 2)},
              str(tmp_path / "audio.safetensors"))
    with pytest.raises(KeyError, match="stray"):
        cc.convert_audio_vae_file(str(tmp_path / "audio.safetensors"),
                                  str(tmp_path / "out"), {"known.weight"})


def test_convert_audio_vae_file_reports_missing_keys(tmp_path):
    save_file({"known.weight": torch.randn(2, 2)},
              str(tmp_path / "audio.safetensors"))
    with pytest.raises(KeyError, match="unaccounted"):
        cc.convert_audio_vae_file(str(tmp_path / "audio.safetensors"),
                                  str(tmp_path / "out"),
                                  {"known.weight", "absent.weight"})


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------

def test_shard_writer_single_shard_no_index(tmp_path):
    w = cc.ShardWriter(str(tmp_path), "model")
    w.add("a", torch.randn(4, 4))
    assert w.finalize() == 1
    assert os.path.isfile(tmp_path / "model.safetensors")
    assert not os.path.isfile(tmp_path / "model.safetensors.index.json")


def test_shard_writer_multi_shard_index(tmp_path):
    w = cc.ShardWriter(str(tmp_path), "model", max_shard_bytes=64)
    for i in range(4):
        w.add(f"t{i}", torch.randn(4, 4))  # 64 bytes each -> one per shard
    shards = w.finalize()
    assert shards == 4
    index_path = tmp_path / "model.safetensors.index.json"
    assert index_path.is_file()
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    assert set(index["weight_map"]) == {"t0", "t1", "t2", "t3"}
    for shard_name in index["weight_map"].values():
        assert (tmp_path / shard_name).is_file()
    assert index["metadata"]["total_size"] == 4 * 64
