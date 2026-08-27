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
    dequantize_scaled_fp8,
    has_fp8_weights,
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
