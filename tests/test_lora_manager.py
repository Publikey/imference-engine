"""GPU-free tests for the image LoRAManager (managers/lora.py) and its
Engine.generate wiring: parsing/aliases, URL resolution + cache pruning, the
apply/reuse/evict lifecycle on a fake pipe, the never-fuse + deactivate-in-
finally contract, and the supports_loras gate (SDXL / Z-Image / Krea 2 /
Anima on, everything else warn+ignore).
"""
from __future__ import annotations

import os
from collections import OrderedDict

import pytest

from imference_engine.managers.lora import LoRAManager, _cache_filename, _derive_adapter_name


# ---------------------------------------------------------------- parse

def test_parse_normalizes_and_accepts_aliases():
    out = LoRAManager.parse([
        {"source": "/a/style.safetensors", "weight": "0.8"},
        {"path": "/b/Char Name-v2.safetensors"},          # legacy worker key
        {"url": "https://cdn/x.safetensors", "adapter_name": "x", "weight": 0.5},
        {"weight": 1.0},                                   # no source -> dropped
        "not-a-dict",                                      # dropped
    ])
    assert [c["adapter_name"] for c in out] == ["style", "char_name_v2", "x"]
    assert [c["weight"] for c in out] == [0.8, 1.0, 0.5]
    assert out[1]["source"] == "/b/Char Name-v2.safetensors"


def test_derive_adapter_name_sanitizes():
    assert _derive_adapter_name("https://x/My LoRA (v2).safetensors?token=1") == "my_lora__v2"


def test_cache_filename_is_stable_and_collision_free():
    a = _cache_filename("https://cdn/a/style.safetensors")
    b = _cache_filename("https://cdn/b/style.safetensors")
    assert a != b                       # same basename, different URL
    assert a == _cache_filename("https://cdn/a/style.safetensors")  # stable
    assert a.endswith(".safetensors")


# ---------------------------------------------------------------- resolve

def test_resolve_local_file_passthrough(tmp_path):
    f = tmp_path / "l.safetensors"
    f.write_bytes(b"x")
    m = LoRAManager(cache_dir=str(tmp_path / "cache"))
    assert m.resolve(str(f)) == str(f)


def test_resolve_url_downloads_once_then_reuses(tmp_path, monkeypatch):
    m = LoRAManager(cache_dir=str(tmp_path))
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        with open(dest, "wb") as f:
            f.write(b"weights")

    monkeypatch.setattr(m, "_download", fake_download)
    url = "https://cdn.example/loras/style.safetensors"
    p1 = m.resolve(url)
    p2 = m.resolve(url)
    assert p1 == p2 and os.path.isfile(p1)
    assert calls == [url]  # second call was a cache hit


def test_resolve_rejects_unknown_source(tmp_path):
    m = LoRAManager(cache_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        m.resolve("not/a/real/thing")


def test_cache_prune_keeps_newest(tmp_path, monkeypatch):
    m = LoRAManager(cache_dir=str(tmp_path), max_cached_files=2)
    monkeypatch.setattr(m, "_download", lambda url, dest: open(dest, "wb").write(b"x"))
    for i in range(4):
        p = m.resolve(f"https://cdn/l{i}.safetensors")
        os.utime(p, (i + 1, i + 1))  # deterministic mtimes, oldest first
        m._prune_cache()
    left = sorted(os.listdir(tmp_path))
    assert len(left) == 2
    assert any("l3" in f for f in left)  # newest survived


# ---------------------------------------------------------------- apply / deactivate

class FakePipe:
    def __init__(self):
        self.loaded: list[tuple] = []
        self.deleted: list[str] = []
        self.active: tuple | None = None
        self.enabled = True

    def enable_lora(self):
        self.enabled = True

    def disable_lora(self):
        self.enabled = False

    def load_lora_weights(self, path, weight_name=None, adapter_name=None):
        self.loaded.append((path, weight_name, adapter_name))

    def delete_adapters(self, name):
        self.deleted.append(name)

    def set_adapters(self, names, adapter_weights=None):
        self.active = (list(names), list(adapter_weights or []))


def _mgr_with_files(tmp_path, n=6):
    files = []
    for i in range(n):
        f = tmp_path / f"l{i}.safetensors"
        f.write_bytes(b"x")
        files.append(str(f))
    return LoRAManager(cache_dir=str(tmp_path / "cache"), max_adapters=2), files


def test_apply_loads_offline_safe_form_and_activates(tmp_path):
    m, files = _mgr_with_files(tmp_path, 2)
    pipe = FakePipe()
    cfgs = LoRAManager.parse([
        {"source": files[0], "weight": 0.8},
        {"source": files[1], "weight": 0.5, "adapter_name": "b"},
    ])
    m.apply(pipe, cfgs)
    # (dir, weight_name) form — offline-safe under HF_HUB_OFFLINE=1
    assert pipe.loaded[0] == (os.path.dirname(files[0]), "l0.safetensors", "l0")
    assert pipe.active == (["l0", "b"], [0.8, 0.5])


def test_apply_reuses_cached_adapter_and_evicts_lru(tmp_path):
    m, files = _mgr_with_files(tmp_path, 3)  # max_adapters=2
    pipe = FakePipe()
    m.apply(pipe, LoRAManager.parse([{"source": files[0]}]))
    m.apply(pipe, LoRAManager.parse([{"source": files[1]}]))
    assert len(pipe.loaded) == 2 and pipe.deleted == []
    # l0 again: reuse, no reload
    m.apply(pipe, LoRAManager.parse([{"source": files[0]}]))
    assert len(pipe.loaded) == 2
    # a third adapter evicts the LRU (l1 — l0 was just refreshed)
    m.apply(pipe, LoRAManager.parse([{"source": files[2]}]))
    assert pipe.deleted == ["l1"]
    assert isinstance(getattr(pipe, "_imference_loras"), OrderedDict)


def test_deactivate_disables_layers_and_never_raises():
    pipe = FakePipe()
    LoRAManager.deactivate(pipe)
    assert pipe.enabled is False

    class Broken:
        def disable_lora(self):
            raise RuntimeError("nope")

    LoRAManager.deactivate(Broken())  # must not raise


def test_apply_after_deactivate_re_enables_the_layers(tmp_path):
    """Regression (GPU-observed): after the first request's deactivate, every
    later request rendered without its LoRA — set_adapters alone does not undo
    disable_lora."""
    m, files = _mgr_with_files(tmp_path, 1)
    pipe = FakePipe()
    cfgs = LoRAManager.parse([{"source": files[0]}])
    m.apply(pipe, cfgs)
    LoRAManager.deactivate(pipe)
    m.apply(pipe, cfgs)
    assert pipe.enabled is True
    assert pipe.active == (["l0"], [1.0])


# ---------------------------------------------------------------- backend gate

def test_supports_loras_flags():
    from imference_engine.anima.backend import AnimaBackend
    from imference_engine.chroma.backend import ChromaBackend
    from imference_engine.flux.backend import FluxBackend
    from imference_engine.krea2.backend import Krea2Backend
    from imference_engine.pipelines.sd15 import SD15Backend
    from imference_engine.pipelines.sdxl import SDXLBackend
    from imference_engine.qwenimage.backend import QwenImageBackend
    from imference_engine.zimage.backend import ZImageBackend

    for be in (SDXLBackend, ZImageBackend, Krea2Backend, AnimaBackend):
        assert be.supports_loras is True, be.__name__
    for be in (SD15Backend, FluxBackend, ChromaBackend, QwenImageBackend):
        assert be.supports_loras is False, be.__name__


def test_generate_ignores_loras_on_unsupported_backend(caplog):
    """A LoRA request on a non-supporting backend warns and renders normally."""
    import logging

    from tests.test_generate_precedence import RecordingBackend, _engine_with

    be = RecordingBackend()  # supports_loras defaults to False
    engine = _engine_with(be)
    with caplog.at_level(logging.WARNING):
        result = engine.generate(model="m", prompt="cat", seed=1,
                                 loras=[{"source": "/x.safetensors"}])
    assert result.ok or result.media  # rendered (fake backend), not errored out
    assert any("does not support LoRAs" in r.message for r in caplog.records)


def test_generate_applies_and_deactivates_on_supported_backend(tmp_path):
    """supports_loras=True routes through the manager: apply before encode,
    deactivate after — and a failed load returns an error result, not a crash."""
    from tests.test_generate_precedence import RecordingBackend, _engine_with

    calls = []

    class LoraBackend(RecordingBackend):
        supports_loras = True

    be = LoraBackend()
    engine = _engine_with(be)

    class SpyLoras:
        def parse(self, loras):
            return LoRAManager.parse(loras)

        def apply(self, pipe, cfgs, family=None):
            calls.append(("apply", [c["adapter_name"] for c in cfgs]))

        def deactivate(self, pipe):
            calls.append(("deactivate",))

    engine._loras = SpyLoras()
    f = tmp_path / "style.safetensors"
    f.write_bytes(b"x")
    result = engine.generate(model="m", prompt="cat", seed=1,
                             loras=[{"source": str(f), "weight": 0.7}])
    assert result.media
    assert calls == [("apply", ["style"]), ("deactivate",)]

    # Failed apply -> error result with the partial-success contract intact.
    class FailingLoras(SpyLoras):
        def apply(self, pipe, cfgs, family=None):
            raise FileNotFoundError("no such lora")

    engine._loras = FailingLoras()
    result = engine.generate(model="m", prompt="cat", seed=1, batch=2,
                             loras=[{"source": str(f)}])
    assert not result.ok
    assert result.media == [None, None]
    assert "Failed to load LoRA" in result.errors[0].error


# ---------------------------------------------------------------- family check / robustness

def _write_lora(path, tensors: dict, metadata: dict | None = None) -> str:
    """Header-only safetensors file: enough for inspect_lora (never reads data)."""
    import json
    import struct

    header = {k: {"dtype": "F16", "shape": shape, "data_offsets": [0, 0]}
              for k, shape in tensors.items()}
    if metadata is not None:
        header["__metadata__"] = metadata
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
    return str(path)


@pytest.mark.parametrize("tensors,metadata,family,reason", [
    ({"lora_unet_x.lora_down.weight": [4, 8]}, {"ss_base_model_version": "sdxl_base_v1-0"},
     "sdxl", "trainer metadata"),
    ({"x.lora_A.weight": [4, 8]}, {"modelspec.architecture": "stable-diffusion-xl-v1-base/lora"},
     "sdxl", "trainer metadata"),
    ({"lora_unet_x.lora_down.weight": [4, 8]}, {"ss_base_model_version": "sd_v1"},
     "sd15", "trainer metadata"),
    ({"lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": [4, 1280]}, None,
     "sdxl", "key layout"),
    ({"lora_unet_input_blocks_4_1_proj_in.lora_down.weight": [4, 640]}, None,
     "sdxl", "key layout"),
    ({"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": [4, 3072]}, None,
     "flux", "key layout"),
    ({"unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight": [4, 2048]},
     None, "sdxl", "cross-attention width"),
    ({"lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight":
      [4, 768]}, None, "sd15", "cross-attention width"),
    ({"diffusion_model.layers.0.attention.to_q.lora_A.weight": [4, 3840]}, None, "zimage", "key layout"),
    ({"lora_unet_layers_0_attention_to_q.lora_down.weight": [4, 3840]}, None, "zimage", "key layout"),
    ({"diffusion_model.noise_refiner.0.attention.to_k.lora_A.weight": [4, 3840]}, None, "zimage", "key layout"),
    ({"diffusion_model.blocks.0.attn.wq.lora_A.weight": [4, 3072]}, None, "krea2", "key layout"),
    ({"transformer.text_fusion.refiner_blocks.0.attn.to_q.lora_A.weight": [4, 3072]}, None, "krea2", "key layout"),
    ({"diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": [4, 2048]}, None, "anima", "key layout"),
    ({"diffusion_model.llm_adapter.blocks.0.self_attn.q_proj.lora_A.weight": [4, 1024]}, None, "anima", "key layout"),
    # Anima in diffusers format has attn2.to_k: must not read as SD2 by its width.
    ({"transformer.transformer_blocks.0.attn2.to_k.lora_A.weight": [4, 1024],
      "transformer.transformer_blocks.0.norm1.linear_1.lora_A.weight": [4, 2048]}, None, "anima", "key layout"),
    ({"some.unknown.module.lora_A.weight": [4, 3840]}, None, None, "unrecognized layout"),
])
def test_inspect_lora_detects_family(tmp_path, tensors, metadata, family, reason):
    from imference_engine.managers.lora_inspect import inspect_lora

    info = inspect_lora(_write_lora(tmp_path / "l.safetensors", tensors, metadata))
    assert (info.family, info.reason) == (family, reason)


def test_inspect_lora_rejects_non_safetensors(tmp_path):
    from imference_engine.managers.lora_inspect import inspect_lora

    f = tmp_path / "pickled.safetensors"
    f.write_bytes(bytes([0x80, 0x02]) + b"}q(X model")  # pickle, not safetensors
    with pytest.raises(ValueError):
        inspect_lora(str(f))


def test_is_compatible_lets_unknown_through():
    from imference_engine.managers.lora_inspect import is_compatible

    assert is_compatible("sdxl", "sdxl")
    assert is_compatible(None, "sdxl")
    assert not is_compatible("sd15", "sdxl")


def test_apply_refuses_lora_for_another_family(tmp_path):
    f = _write_lora(tmp_path / "sd15.safetensors", {
        "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight":
            [4, 768]})
    m = LoRAManager(cache_dir=str(tmp_path / "cache"))
    pipe = FakePipe()
    with pytest.raises(ValueError, match="targets sd15 .* model is sdxl"):
        m.apply(pipe, LoRAManager.parse([{"source": f}]), family="sdxl")
    assert pipe.loaded == []


def test_apply_accepts_matching_family(tmp_path):
    f = _write_lora(tmp_path / "xl.safetensors", {"lora_te2_x.lora_down.weight": [4, 1280]})
    m = LoRAManager(cache_dir=str(tmp_path / "cache"))
    pipe = FakePipe()
    m.apply(pipe, LoRAManager.parse([{"source": f}]), family="sdxl")
    assert pipe.active == (["xl"], [1.0])


def test_apply_reloads_when_adapter_name_points_to_another_file(tmp_path):
    m, files = _mgr_with_files(tmp_path, 2)
    pipe = FakePipe()
    m.apply(pipe, LoRAManager.parse([{"source": files[0], "adapter_name": "style"}]))
    m.apply(pipe, LoRAManager.parse([{"source": files[1], "adapter_name": "style"}]))
    assert pipe.deleted == ["style"]
    assert [entry[1] for entry in pipe.loaded] == ["l0.safetensors", "l1.safetensors"]


def test_failed_load_cleans_up_half_injected_adapter(tmp_path):
    m, files = _mgr_with_files(tmp_path, 1)

    class FailingPipe(FakePipe):
        def load_lora_weights(self, *a, **k):
            raise RuntimeError("size mismatch")

    pipe = FailingPipe()
    with pytest.raises(RuntimeError):
        m.apply(pipe, LoRAManager.parse([{"source": files[0]}]))
    assert pipe.deleted == ["l0"]
    assert "l0" not in getattr(pipe, "_imference_loras")


def test_img2img_applies_loras_on_the_resident_pipe_not_the_wrapper(tmp_path):
    """Regression: the adapter bookkeeping must land on the resident t2i pipe.
    On the per-request img2img wrapper, the next request would reload the
    adapter under a name already taken in the shared modules."""
    from PIL import Image

    from tests.test_generate_precedence import RecordingBackend, _engine_with

    seen = []

    class Wrapper:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, **kwargs):
            return self.inner(**kwargs)

    class Img2ImgBackend(RecordingBackend):
        supports_loras = True

        def make_img2img(self, t2i_pipe):
            return Wrapper(t2i_pipe)

    engine = _engine_with(Img2ImgBackend())
    resident, _ = engine._models.get_or_load("m")

    class SpyLoras:
        def parse(self, loras):
            return LoRAManager.parse(loras)

        def apply(self, pipe, cfgs, family=None):
            seen.append(("apply", pipe))

        def deactivate(self, pipe):
            seen.append(("deactivate", pipe))

    engine._loras = SpyLoras()
    f = tmp_path / "style.safetensors"
    f.write_bytes(b"x")
    result = engine.generate(model="m", prompt="cat", seed=1,
                             source_image=Image.new("RGB", (64, 64)),
                             loras=[{"source": str(f)}])
    assert result.media
    assert seen == [("apply", resident), ("deactivate", resident)]


def test_apply_refuses_more_loras_than_the_adapter_cache(tmp_path):
    m, files = _mgr_with_files(tmp_path, 3)  # max_adapters=2
    pipe = FakePipe()
    with pytest.raises(ValueError, match="at most 2"):
        m.apply(pipe, LoRAManager.parse([{"source": f} for f in files]))
    assert pipe.loaded == []
