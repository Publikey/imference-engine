"""LoRA family detection for the families that share a key layout: FLUX.1 vs
FLUX.2 klein, and Lumina-Image 2.0 vs Z-Image. Header-only, no torch. The key
layouts are the ones seen in public files (ComfyUI and diffusers exports)."""
from __future__ import annotations

import json
import struct

from imference_engine.flux2klein import Flux2KleinBackend
from imference_engine.lumina2 import Lumina2Backend
from imference_engine.managers.lora_inspect import inspect_lora, is_compatible


def _write(tmp_path, tensors, metadata=None, name="lora.safetensors"):
    header = {k: {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 0]}
              for k, shape in tensors.items()}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    p = tmp_path / name
    p.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return str(p)


def _klein_comfy(rank=16):
    return {
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": (rank, 3072),
        "diffusion_model.double_blocks.4.img_attn.proj.lora_A.weight": (rank, 3072),
        "diffusion_model.single_blocks.19.linear1.lora_A.weight": (rank, 3072),
        "diffusion_model.single_blocks.19.linear2.lora_A.weight": (rank, 12288),
    }


def test_klein_from_trainer_metadata(tmp_path):
    p = _write(tmp_path, _klein_comfy(), {"ss_base_model_version": "flux2_klein_4b"})
    info = inspect_lora(p)
    assert info.family == "flux2-klein" and info.reason == "trainer metadata"
    assert is_compatible(info.family, Flux2KleinBackend.engine)
    assert not is_compatible(info.family, "flux")


def test_klein_comfy_layout_without_metadata(tmp_path):
    info = inspect_lora(_write(tmp_path, _klein_comfy()))
    assert info.family == "flux2-klein" and info.reason == "key layout"


def test_klein_diffusers_layout_is_recognized_by_its_module_name(tmp_path):
    p = _write(tmp_path, {
        "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight": (128, 3072),
        "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight": (128, 3072),
    })
    assert inspect_lora(p).family == "flux2-klein"


def test_flux1_stays_flux(tmp_path):
    # more blocks than klein has
    deep = _write(tmp_path, {
        "lora_unet_double_blocks_18_img_attn_qkv.lora_down.weight": (16, 3072),
        "lora_unet_single_blocks_37_linear1.lora_down.weight": (16, 3072),
    }, name="deep.safetensors")
    assert inspect_lora(deep).family == "flux"
    # FLUX.1's single-block projection is 15360 wide (klein: 12288)
    wide = _write(tmp_path, {
        "diffusion_model.single_blocks.0.linear2.lora_A.weight": (16, 15360),
    }, name="wide.safetensors")
    assert inspect_lora(wide).family == "flux"
    # metadata wins
    meta = _write(tmp_path, _klein_comfy(), {"ss_base_model_version": "flux1"},
                  name="meta.safetensors")
    assert inspect_lora(meta).family == "flux"
    assert not is_compatible("flux", Flux2KleinBackend.engine)


def test_ambiguous_flux_layout_is_not_claimed_for_klein(tmp_path):
    """Few blocks and no single-block projection: could be either generation —
    keep the historical answer (FLUX.1) rather than guess klein."""
    p = _write(tmp_path, {"diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": (16, 3072)})
    assert inspect_lora(p).family == "flux"


def test_lumina2_is_told_from_zimage_by_width(tmp_path):
    lumina = _write(tmp_path, {
        "transformer.layers.20.attn.to_k.lora_A.weight": (4, 2304),
        "transformer.context_refiner.0.attn.to_k.lora_A.weight": (4, 2304),
    }, name="lumina.safetensors")
    info = inspect_lora(lumina)
    assert info.family == "lumina2"
    assert is_compatible(info.family, Lumina2Backend.engine)
    assert not is_compatible(info.family, "zimage")

    zimage = _write(tmp_path, {
        "diffusion_model.layers.3.attention.to_q.lora_A.weight": (16, 3840),
        "diffusion_model.noise_refiner.0.attention.to_q.lora_A.weight": (16, 3840),
    }, name="zimage.safetensors")
    assert inspect_lora(zimage).family == "zimage"

    # ComfyUI-style Lumina keys (attention / feed_forward), still 2304 wide
    comfy = _write(tmp_path, {
        "diffusion_model.layers.3.attention.qkv.lora_A.weight": (8, 2304),
    }, name="comfy.safetensors")
    assert inspect_lora(comfy).family == "lumina2"


def test_layers_layout_without_a_known_width_defaults_to_zimage(tmp_path):
    p = _write(tmp_path, {"diffusion_model.layers.3.attention.to_q.lora_B.weight": (3840, 16)})
    assert inspect_lora(p).family == "zimage"


def test_lumina_metadata(tmp_path):
    p = _write(tmp_path, {"x.lora_A.weight": (4, 4)}, {"ss_base_model_version": "lumina2"})
    assert inspect_lora(p).family == "lumina2"


def test_backends_declare_lora_support():
    assert Flux2KleinBackend.supports_loras is True
    assert Lumina2Backend.supports_loras is True
