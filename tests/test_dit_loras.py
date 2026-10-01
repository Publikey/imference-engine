"""DiT LoRA plumbing without a GPU: the Anima kohya / lora_down-up converter
(anima/lora_convert.py) and the fp8 adapter upcast for the Krea 2 fp8-resident
transformer (managers/lora.py)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from imference_engine.anima.lora_convert import needs_conversion, to_comfy_lora_ab  # noqa: E402
from imference_engine.managers.lora import _upcast_float8_adapter  # noqa: E402


def test_kohya_anima_keys_map_to_comfy_lora_ab_with_alpha_baked():
    down, up = torch.ones(4, 8), torch.ones(16, 4)
    sd = {
        "lora_unet_blocks_0_cross_attn_k_proj.lora_down.weight": down,
        "lora_unet_blocks_0_cross_attn_k_proj.lora_up.weight": up,
        "lora_unet_blocks_0_cross_attn_k_proj.alpha": torch.tensor(2.0),
        "lora_unet_blocks_11_mlp_layer2.lora_down.weight": down,
        "lora_unet_blocks_11_mlp_layer2.lora_up.weight": up,
        "lora_unet_blocks_3_adaln_modulation_self_attn_1.lora_down.weight": down,
        "lora_unet_blocks_3_adaln_modulation_self_attn_1.lora_up.weight": up,
    }
    assert needs_conversion(sd)
    out = to_comfy_lora_ab(sd)
    assert set(out) == {
        "diffusion_model.blocks.0.cross_attn.k_proj.lora_A.weight",
        "diffusion_model.blocks.0.cross_attn.k_proj.lora_B.weight",
        "diffusion_model.blocks.11.mlp.layer2.lora_A.weight",
        "diffusion_model.blocks.11.mlp.layer2.lora_B.weight",
        "diffusion_model.blocks.3.adaln_modulation_self_attn.1.lora_A.weight",
        "diffusion_model.blocks.3.adaln_modulation_self_attn.1.lora_B.weight",
    }
    # alpha 2 / rank 4 = 0.5 baked into lora_B; no alpha = scale 1.
    assert torch.allclose(out["diffusion_model.blocks.0.cross_attn.k_proj.lora_B.weight"], up * 0.5)
    assert torch.equal(out["diffusion_model.blocks.11.mlp.layer2.lora_B.weight"], up)


def test_comfy_lora_down_up_is_renamed():
    sd = {
        "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight": torch.ones(2, 8),
        "diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight": torch.ones(8, 2),
    }
    assert needs_conversion(sd)
    assert set(to_comfy_lora_ab(sd)) == {
        "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight",
        "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight",
    }


def test_comfy_lora_ab_needs_no_conversion():
    assert not needs_conversion(["diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight"])


def test_unknown_layout_fails_loudly():
    with pytest.raises(ValueError, match="Unrecognized Anima LoRA layout"):
        to_comfy_lora_ab({"lora_unet_mystery_module.lora_down.weight": torch.ones(1, 1),
                          "lora_unet_mystery_module.lora_up.weight": torch.ones(1, 1)})


class _LoraLinear(torch.nn.Module):
    """Shape of a peft LoraLayer: lora_A / lora_B ModuleDicts keyed by adapter."""

    def __init__(self, dtype):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"t": torch.nn.Linear(8, 2, bias=False).to(dtype)})
        self.lora_B = torch.nn.ModuleDict({"t": torch.nn.Linear(2, 8, bias=False).to(dtype)})


class _Pipe:
    def __init__(self, module):
        self.components = {"transformer": module, "tokenizer": object()}


def test_fp8_adapter_weights_are_upcast_to_bf16():
    layer = _LoraLinear(torch.float8_e4m3fn)
    _upcast_float8_adapter(_Pipe(torch.nn.Sequential(layer)), "t")
    assert layer.lora_A["t"].weight.dtype == torch.bfloat16
    assert layer.lora_B["t"].weight.dtype == torch.bfloat16


def test_non_fp8_adapters_and_other_names_are_untouched():
    layer = _LoraLinear(torch.float16)
    _upcast_float8_adapter(_Pipe(torch.nn.Sequential(layer)), "t")
    assert layer.lora_A["t"].weight.dtype == torch.float16
    other = _LoraLinear(torch.float8_e4m3fn)
    _upcast_float8_adapter(_Pipe(torch.nn.Sequential(other)), "not-this-one")
    assert other.lora_A["t"].weight.dtype == torch.float8_e4m3fn
