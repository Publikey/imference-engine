"""Unit tests for the Wan sub-package — pure logic, no torch/GPU.

Mirrors the image-side approach (test_model_lru.py): the residency manager's
build/load are monkeypatched so LRU behaviour is testable without weights.
"""
import pytest

from imference_engine.wan import MemoryProfile, WanLora, WanVariant
from imference_engine.wan import manager as mgr_mod
from imference_engine.wan.manager import ResidencyManager


def _variant(name: str, mode: str = "t2v") -> WanVariant:
    return WanVariant(name=name, mode=mode, base_repo="base",
                      gguf_repo="gguf", lightning_baked=True)


def test_builtins_have_offline_safe_templates():
    # Every built-in must resolve GGUF names deterministically (no list_repo_files
    # at runtime) so it can run under HF_HUB_OFFLINE=1 from a mirrored cache.
    from imference_engine.wan.presets import BUILTIN_VARIANTS
    for name, v in BUILTIN_VARIANTS.items():
        assert v.gguf_high_template and v.gguf_low_template, name
        hi = v.gguf_high_template.format(quant="Q8_0")
        lo = v.gguf_low_template.format(quant="Q6_K")
        assert hi.endswith("-Q8_0.gguf") and lo.endswith("-Q6_K.gguf")


def test_memory_profile_mapping():
    assert MemoryProfile.GGUF_Q8.gguf_quant == "Q8_0"
    assert MemoryProfile.GGUF_Q6.gguf_quant == "Q6_K"
    assert MemoryProfile.GGUF_Q8.is_gguf
    assert not MemoryProfile.BF16.is_gguf
    assert MemoryProfile.BF16.gguf_quant is None


def test_memory_profile_for_vram():
    assert MemoryProfile.for_vram(80) == MemoryProfile.GGUF_Q8
    assert MemoryProfile.for_vram(24) == MemoryProfile.GGUF_Q8
    assert MemoryProfile.for_vram(16) == MemoryProfile.GGUF_Q6
    assert MemoryProfile.for_vram(12) == MemoryProfile.GGUF_Q4


def test_variant_rejects_baked_with_loras():
    with pytest.raises(ValueError):
        WanVariant(name="x", mode="t2v", base_repo="b", gguf_repo="g",
                   lightning_baked=True, loras=[WanLora("r", "h", "l")])


def test_variant_rejects_bad_mode():
    with pytest.raises(ValueError):
        WanVariant(name="x", mode="zzz", base_repo="b", gguf_repo="g")


def test_residency_lru(monkeypatch):
    built: list[str] = []
    evicted: list[str] = []
    monkeypatch.setattr(mgr_mod, "load_shared_components",
                        lambda base, cache_dir=None: object())

    def fake_build(variant, **kwargs):
        built.append(variant.name)
        return f"pipe:{variant.name}"

    monkeypatch.setattr(mgr_mod, "build_pipeline", fake_build)

    m = ResidencyManager(quant="Q8_0", device="cpu", enable_offload=True,
                         vae_tiling=True, max_resident=2,
                         on_evicted=evicted.append)
    a, b, c = _variant("a"), _variant("b"), _variant("c")

    assert m.get_or_load(a) == "pipe:a"
    assert m.get_or_load(b) == "pipe:b"
    assert m.resident == ["a", "b"]

    # re-access a -> becomes MRU
    m.get_or_load(a)
    assert m.resident == ["b", "a"]

    # load c -> evicts LRU (b)
    m.get_or_load(c)
    assert m.resident == ["a", "c"]
    assert evicted == ["b"]

    # cached pipe is not rebuilt
    m.get_or_load(a)
    assert built == ["a", "b", "c"]  # a built once


def test_residency_single_resident(monkeypatch):
    monkeypatch.setattr(mgr_mod, "load_shared_components",
                        lambda base, cache_dir=None: object())
    monkeypatch.setattr(mgr_mod, "build_pipeline",
                        lambda variant, **kw: f"pipe:{variant.name}")
    m = ResidencyManager(quant="Q8_0", device="cpu", enable_offload=True,
                         vae_tiling=True, max_resident=1)
    m.get_or_load(_variant("a"))
    m.get_or_load(_variant("b"))
    assert m.resident == ["b"]  # single-resident evicts immediately
