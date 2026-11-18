"""FLUX.2 klein single-file format detection (header only, no torch) and the
scaled-fp8 dequant (skipped when torch is not installed)."""
from __future__ import annotations

import json
import struct

import pytest

from imference_engine.flux2klein.convert import (
    QUANT_METADATA_KEY,
    detect_format,
    read_safetensors_header,
)


def _t(dtype, shape=(2, 2)):
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}


def test_plain_bf16_is_bf16():
    assert detect_format({"double_blocks.0.img_attn.qkv.weight": _t("BF16")}) == "bf16"


def test_scaled_fp8_is_detected_from_dtype_and_scale():
    header = {
        "double_blocks.0.img_attn.qkv.weight": _t("F8_E4M3"),
        "double_blocks.0.img_attn.qkv.weight_scale": _t("F32", ()),
        "double_blocks.0.img_attn.qkv.input_scale": _t("F32", ()),
        "double_blocks.0.img_attn.norm.key_norm.scale": _t("BF16", (2,)),
        "__metadata__": {QUANT_METADATA_KEY: '{"layers": {"x": {"format": "float8_e4m3fn"}}}'},
    }
    assert detect_format(header) == "fp8-scaled"


def test_plain_fp8_without_scales():
    assert detect_format({"a.weight": _t("F8_E4M3")}) == "fp8"


def test_nvfp4_is_detected_from_metadata_or_second_scale():
    by_meta = {"a.weight": _t("U8"),
               "__metadata__": {QUANT_METADATA_KEY: '{"layers": {"a": {"format": "nvfp4"}}}'}}
    by_keys = {"a.weight": _t("U8"), "a.weight_scale": _t("F8_E4M3"), "a.weight_scale_2": _t("F32", ())}
    assert detect_format(by_meta) == "nvfp4"
    assert detect_format(by_keys) == "nvfp4"


def test_read_safetensors_header_roundtrip(tmp_path):
    header = {"a.weight": _t("BF16"), "__metadata__": {"k": "v"}}
    raw = json.dumps(header).encode()
    p = tmp_path / "x.safetensors"
    p.write_bytes(struct.pack("<Q", len(raw)) + raw)
    assert read_safetensors_header(str(p)) == header


def test_dequantize_scaled_fp8():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8")
    from imference_engine.flux2klein.convert import dequantize_scaled_fp8

    w = torch.tensor([[1.0, -2.0], [0.5, 4.0]])
    sd = {
        "blk.proj.weight": w.to(torch.float8_e4m3fn),
        "blk.proj.weight_scale": torch.tensor(0.25),
        "blk.proj.input_scale": torch.tensor(0.3),
        "blk.norm.scale": torch.ones(2, dtype=torch.bfloat16),
    }
    out = dequantize_scaled_fp8(sd, torch.float32)
    assert set(out) == {"blk.proj.weight", "blk.norm.scale"}   # helper tensors dropped
    assert torch.allclose(out["blk.proj.weight"], w * 0.25)
    assert out["blk.norm.scale"].dtype == torch.float32
