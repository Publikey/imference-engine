"""GPU-free tests for the Krea 2 single-file normalization (krea2/convert.py):
ComfyUI prefix strip, native-layout detection, scaled-fp8 dequantization, the
native→diffusers key remap (with modulation-table reshapes), and the
mixed-layout collision guard. Uses tiny synthetic tensors — no weights, no GPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from imference_engine.krea2.convert import (  # noqa: E402
    convert_krea2_native_to_diffusers,
    dequantize_int8_convrot,
    dequantize_scaled_fp8,
    has_fp8_weights,
    has_int8_convrot,
    is_native_krea2_format,
    prepare_krea2_state_dict,
    strip_comfyui_prefix,
)

H = 8  # tiny stand-in hidden size


# ---------------------------------------------------------------- prefix strip

def test_strips_model_diffusion_model_prefix():
    sd = {"model.diffusion_model.blocks.0.attn.wq.weight": torch.zeros(2, 2)}
    out = strip_comfyui_prefix(sd)
    assert list(out) == ["blocks.0.attn.wq.weight"]


def test_strips_bare_diffusion_model_prefix():
    sd = {"diffusion_model.txtfusion.projector.weight": torch.zeros(2, 2)}
    out = strip_comfyui_prefix(sd)
    assert list(out) == ["txtfusion.projector.weight"]


def test_no_prefix_is_untouched():
    sd = {"blocks.0.attn.wq.weight": torch.zeros(2, 2)}
    assert strip_comfyui_prefix(sd) is sd


# ------------------------------------------------------------------ detection

def test_native_format_detected_by_blocks_txtfusion_first_and_mod_lin():
    for key in ("blocks.0.attn.wq.weight", "txtfusion.projector.weight",
                "first.weight", "blocks.3.mod.lin"):
        assert is_native_krea2_format({key: None}), key


def test_diffusers_format_is_not_native():
    assert not is_native_krea2_format(
        {"transformer_blocks.0.attn.to_q.weight": None, "img_in.weight": None}
    )


# ---------------------------------------------------------------- fp8 dequant

def test_scaled_fp8_dequant_is_weight_times_scale():
    w_true = torch.tensor([[0.5, -1.25], [2.0, 0.0]])
    scale = torch.tensor(0.125)
    w_fp8 = (w_true / scale).to(torch.float8_e4m3fn)
    sd = {"blocks.0.attn.wq.weight": w_fp8,
          "blocks.0.attn.wq.weight_scale": scale}

    out = dequantize_scaled_fp8(sd, torch.bfloat16)

    assert "blocks.0.attn.wq.weight_scale" not in out
    got = out["blocks.0.attn.wq.weight"]
    assert got.dtype == torch.bfloat16
    # 0.5/-1.25/2.0/0.0 over 0.125 are exact in e4m3, so the roundtrip is exact.
    assert torch.equal(got.float(), w_true)


def test_dequant_is_a_noop_without_scale_keys():
    sd = {"blocks.0.attn.wq.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}
    assert dequantize_scaled_fp8(sd, torch.bfloat16) is sd


def test_has_fp8_weights_detects_scales_and_raw_fp8():
    assert has_fp8_weights({"x.weight_scale": torch.tensor(1.0)})
    assert has_fp8_weights({"x.weight": torch.zeros(2, dtype=torch.float8_e4m3fn)})
    assert not has_fp8_weights({"x.weight": torch.zeros(2, dtype=torch.bfloat16)})


def test_prepare_upcasts_plain_fp8_without_scales():
    """Raw-fp8 community quants (no weight_scale) must not leave float8 params."""
    sd = {"blocks.0.attn.wq.weight": torch.zeros(2, 2, dtype=torch.float8_e4m3fn)}
    out, was_fp8 = prepare_krea2_state_dict(sd, torch.bfloat16)
    assert was_fp8
    assert out["transformer_blocks.0.attn.to_q.weight"].dtype == torch.bfloat16


def test_prepare_casts_stray_fp32_params_to_compute_dtype():
    """Some finetune tooling saves fp32 biases next to bf16 weights; with
    load_state_dict(assign=True) they would stay fp32 and F.linear dies
    mid-inference with 'self and mat2 must have the same dtype'. Every
    floating tensor must come out in the compute dtype."""
    sd = {
        "tmlp.0.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
        "tmlp.0.bias": torch.zeros(4, dtype=torch.float32),
        "blocks.0.prenorm.scale": torch.zeros(4, dtype=torch.float32),
    }
    out, _ = prepare_krea2_state_dict(sd, torch.bfloat16)
    assert out["time_embed.linear_1.bias"].dtype == torch.bfloat16
    assert out["transformer_blocks.0.norm1.weight"].dtype == torch.bfloat16
    assert out["time_embed.linear_1.weight"].dtype == torch.bfloat16


# ------------------------------------------------------------------ key remap

def _native_sd():
    return {
        # per-block attention / mlp / norms / modulation
        "blocks.0.attn.wq.weight": torch.zeros(H, H),
        "blocks.0.attn.wk.weight": torch.zeros(H, H),
        "blocks.0.attn.wv.weight": torch.zeros(H, H),
        "blocks.0.attn.wo.weight": torch.zeros(H, H),
        "blocks.0.attn.gate.weight": torch.zeros(H, H),
        "blocks.0.attn.qknorm.qnorm.scale": torch.zeros(H),
        "blocks.0.attn.qknorm.knorm.scale": torch.zeros(H),
        "blocks.0.mlp.gate.weight": torch.zeros(H, H),
        "blocks.0.mlp.up.weight": torch.zeros(H, H),
        "blocks.0.mlp.down.weight": torch.zeros(H, H),
        "blocks.0.prenorm.scale": torch.zeros(H),
        "blocks.0.postnorm.scale": torch.zeros(H),
        "blocks.0.mod.lin": torch.arange(6 * H, dtype=torch.float32),
        # text fusion + projections
        "txtfusion.projector.weight": torch.zeros(1, 12),
        "txtfusion.layerwise_blocks.0.attn.wq.weight": torch.zeros(H, H),
        "txtmlp.0.scale": torch.zeros(H),
        "txtmlp.1.weight": torch.zeros(H, H),
        "txtmlp.3.weight": torch.zeros(H, H),
        # patchify-in, time embedding, final layer
        "first.weight": torch.zeros(H, 64),
        "tmlp.0.weight": torch.zeros(H, H),
        "tmlp.2.weight": torch.zeros(H, H),
        "tproj.1.weight": torch.zeros(6 * H, H),
        "last.linear.weight": torch.zeros(64, H),
        "last.linear.bias": torch.zeros(64),
        "last.norm.scale": torch.zeros(H),
        "last.modulation.lin": torch.arange(2 * H, dtype=torch.float32),
        # native-only projections, dropped
        "last.down.weight": torch.zeros(H, H),
        "last.up.weight": torch.zeros(H, H),
    }


def test_native_remap_produces_diffusers_keys():
    out = convert_krea2_native_to_diffusers(_native_sd())
    expected = {
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.to_k.weight",
        "transformer_blocks.0.attn.to_v.weight",
        "transformer_blocks.0.attn.to_out.0.weight",
        "transformer_blocks.0.attn.to_gate.weight",
        "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.0.attn.norm_k.weight",
        "transformer_blocks.0.ff.gate.weight",
        "transformer_blocks.0.ff.up.weight",
        "transformer_blocks.0.ff.down.weight",
        "transformer_blocks.0.norm1.weight",
        "transformer_blocks.0.norm2.weight",
        "transformer_blocks.0.scale_shift_table",
        "text_fusion.projector.weight",
        "text_fusion.layerwise_blocks.0.attn.to_q.weight",
        "txt_in.norm.weight",
        "txt_in.linear_1.weight",
        "txt_in.linear_2.weight",
        "img_in.weight",
        "time_embed.linear_1.weight",
        "time_embed.linear_2.weight",
        "time_mod_proj.weight",
        "final_layer.linear.weight",
        "final_layer.linear.bias",
        "final_layer.norm.weight",
        "final_layer.scale_shift_table",
    }
    assert set(out) == expected


def test_modulation_tables_are_reshaped():
    out = convert_krea2_native_to_diffusers(_native_sd())
    assert out["transformer_blocks.0.scale_shift_table"].shape == (6, H)
    assert out["final_layer.scale_shift_table"].shape == (2, H)
    # reshape preserves row-major order
    assert torch.equal(
        out["transformer_blocks.0.scale_shift_table"].flatten(),
        torch.arange(6 * H, dtype=torch.float32),
    )


def test_native_only_final_projections_are_dropped():
    out = convert_krea2_native_to_diffusers(_native_sd())
    assert not any("last." in k or ".up." in k or ".down.weight" in k
                   for k in out if k.startswith("final_layer"))


def test_mixed_layout_collision_is_rejected():
    sd = {
        "blocks.0.attn.wq.weight": torch.zeros(2, 2),
        "transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2),
    }
    with pytest.raises(RuntimeError, match="mix native and diffusers"):
        convert_krea2_native_to_diffusers(sd)


def test_prepare_full_pipeline_scaled_fp8_native_file():
    """End-to-end: prefixed + scaled-fp8 + native keys -> diffusers bf16 dict."""
    scale = torch.tensor(0.5)
    sd = {
        "model.diffusion_model.blocks.0.attn.wq.weight":
            (torch.ones(2, 2) / scale).to(torch.float8_e4m3fn),
        "model.diffusion_model.blocks.0.attn.wq.weight_scale": scale,
        "model.diffusion_model.blocks.0.prenorm.scale": torch.zeros(2),
        "model.diffusion_model.blocks.0.attn.wq.comfy_quant": torch.zeros(1),
    }
    out, was_fp8 = prepare_krea2_state_dict(sd, torch.bfloat16)
    assert was_fp8
    assert set(out) == {
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.norm1.weight",
    }
    w = out["transformer_blocks.0.attn.to_q.weight"]
    assert w.dtype == torch.bfloat16
    assert torch.equal(w.float(), torch.ones(2, 2))


# --------------------------------------------------------- int8 ConvRot dequant

def _comfy_quant(fmt="int8_tensorwise", convrot=True, groupsize=4):
    """A ``<base>.comfy_quant`` uint8 tensor as ComfyUI serializes it."""
    import json

    raw = json.dumps(
        {"format": fmt, "convrot": convrot, "convrot_groupsize": groupsize}
    ).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _convrot_quantize(w_true, groupsize=4):
    """Quantize like ComfyUI's ConvRot: block-Hadamard rotate, per-output-channel
    absmax scale, round to int8. (rotate_weight is involutive — the same op the
    dequant applies.)"""
    from imference_engine.minimax_h3.comfy_convert import rotate_weight

    w_rot = rotate_weight(w_true.float(), groupsize)
    scale = w_rot.abs().amax(dim=1, keepdim=True) / 127.0
    w_int8 = torch.round(w_rot / scale).to(torch.int8)
    return w_int8, scale


def test_int8_convrot_roundtrip_within_quantization_error():
    torch.manual_seed(7)
    w_true = torch.randn(3, 8)
    w_int8, scale = _convrot_quantize(w_true)
    sd = {
        "blocks.0.attn.wq.weight": w_int8,
        "blocks.0.attn.wq.weight_scale": scale,
        "blocks.0.attn.wq.comfy_quant": _comfy_quant(),
    }
    out = dequantize_int8_convrot(sd, torch.float32)
    assert "blocks.0.attn.wq.weight_scale" not in out
    assert "blocks.0.attn.wq.comfy_quant" not in out
    got = out["blocks.0.attn.wq.weight"]
    assert got.dtype == torch.float32
    # Error bound: <= 0.5 int8 step per element in the rotated domain; the
    # rotation is orthonormal so it cannot amplify beyond the group mixing.
    assert torch.allclose(got, w_true, atol=float(scale.max()) * 2.0)


def test_has_int8_convrot_gates_on_weight_dtype():
    w_int8 = torch.zeros(2, 4, dtype=torch.int8)
    assert has_int8_convrot({"x.weight": w_int8, "x.comfy_quant": _comfy_quant()})
    assert has_int8_convrot({"x.weight": w_int8, "x.weight_scale": torch.ones(2, 1)})
    # scaled-fp8 layers may carry .comfy_quant tags too — NOT int8-ConvRot
    assert not has_int8_convrot({
        "x.weight": torch.zeros(2, 4, dtype=torch.float8_e4m3fn),
        "x.weight_scale": torch.tensor(1.0),
        "x.comfy_quant": _comfy_quant(fmt="float8_e4m3fn", convrot=False),
    })


def test_prepare_full_pipeline_int8_convrot_native_file():
    """End-to-end: prefixed + int8-ConvRot + native keys -> diffusers dict,
    quantized flag set (drives fp8-resident storage)."""
    torch.manual_seed(11)
    w_true = torch.randn(2, 8)
    w_int8, scale = _convrot_quantize(w_true)
    sd = {
        "model.diffusion_model.blocks.0.attn.wq.weight": w_int8,
        "model.diffusion_model.blocks.0.attn.wq.weight_scale": scale,
        "model.diffusion_model.blocks.0.attn.wq.comfy_quant": _comfy_quant(),
        "model.diffusion_model.blocks.0.prenorm.scale": torch.zeros(2),
    }
    out, was_quantized = prepare_krea2_state_dict(sd, torch.float32)
    assert was_quantized
    assert set(out) == {
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.norm1.weight",
    }
    got = out["transformer_blocks.0.attn.to_q.weight"]
    assert got.dtype == torch.float32
    assert torch.allclose(got, w_true, atol=float(scale.max()) * 2.0)


def test_prepare_rejects_int8_without_quant_config():
    """int8 weights following an unknown scheme must fail loudly, not load."""
    sd = {"blocks.0.attn.wq.weight": torch.zeros(2, 4, dtype=torch.int8)}
    with pytest.raises(RuntimeError, match="int8 tensor"):
        prepare_krea2_state_dict(sd, torch.float32)


def test_int8_with_header_metadata_config():
    """Third-party converters (ComfyUI Kitchen) put ONE _quantization_metadata
    JSON in the safetensors header instead of per-layer .comfy_quant tensors —
    the config must be honored from there (keys may lack the tensor prefix)."""
    import json

    torch.manual_seed(5)
    w_true = torch.randn(2, 8)
    w_int8, scale = _convrot_quantize(w_true)
    sd = {
        "model.diffusion_model.blocks.0.attn.wq.weight": w_int8,
        "model.diffusion_model.blocks.0.attn.wq.weight_scale": scale,
    }
    meta = {"_quantization_metadata": json.dumps({"layers": {
        "blocks.0.attn.wq": {"format": "int8_tensorwise", "convrot": True,
                             "convrot_groupsize": 4},
    }})}
    out, was_quantized = prepare_krea2_state_dict(
        sd, torch.float32, quant_metadata=meta)
    assert was_quantized
    got = out["transformer_blocks.0.attn.to_q.weight"]
    assert torch.allclose(got, w_true, atol=float(scale.max()) * 2.0)


def test_int8_with_scale_but_no_config_anywhere_raises():
    """The CyberRealistic regression: int8 + weight_scale with NO config must
    refuse — falling through to the scaled-fp8 path multiplies without
    un-rotating and renders pure noise."""
    sd = {
        "blocks.0.attn.wq.weight": torch.zeros(2, 4, dtype=torch.int8),
        "blocks.0.attn.wq.weight_scale": torch.ones(2, 1),
    }
    with pytest.raises(RuntimeError, match="no quantization config"):
        prepare_krea2_state_dict(sd, torch.float32)


def test_dequant_device_falls_back_to_cpu_and_stays_correct():
    """A broken dequant device (simulating an OOM/driver error) must fall back
    to the CPU path per-tensor, not fail the load or corrupt results."""
    w_true = torch.tensor([[0.5, -1.25], [2.0, 0.0]])
    scale = torch.tensor(0.125)
    sd = {"blocks.0.attn.wq.weight": (w_true / scale).to(torch.float8_e4m3fn),
          "blocks.0.attn.wq.weight_scale": scale}
    out = dequantize_scaled_fp8(sd, torch.bfloat16, device="notadevice")
    assert torch.equal(out["blocks.0.attn.wq.weight"].float(), w_true)

    torch.manual_seed(3)
    w2 = torch.randn(2, 8)
    w_int8, s2 = _convrot_quantize(w2)
    sd2 = {
        "blocks.0.attn.wq.weight": w_int8,
        "blocks.0.attn.wq.weight_scale": s2,
        "blocks.0.attn.wq.comfy_quant": _comfy_quant(),
    }
    out2 = dequantize_int8_convrot(sd2, torch.float32, device="notadevice")
    assert torch.allclose(out2["blocks.0.attn.wq.weight"], w2,
                          atol=float(s2.max()) * 2.0)


def test_int8_convrot_unsupported_format_raises():
    sd = {
        "blocks.0.attn.wq.weight": torch.zeros(2, 4, dtype=torch.int8),
        "blocks.0.attn.wq.weight_scale": torch.ones(2, 1),
        "blocks.0.attn.wq.comfy_quant": _comfy_quant(fmt="int4_blockwise"),
    }
    with pytest.raises(RuntimeError, match="int4/nvfp4"):
        dequantize_int8_convrot(sd, torch.float32)
