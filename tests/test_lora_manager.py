"""GPU-free tests for the image LoRAManager (managers/lora.py) and its
Engine.generate wiring: parsing/aliases, URL resolution + cache pruning, the
apply/reuse/evict lifecycle on a fake pipe, the never-fuse + deactivate-in-
finally contract, and the supports_loras gate (SDXL on, everything else
warn+ignore).
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


def test_deactivate_clears_active_set_and_never_raises():
    pipe = FakePipe()
    pipe.active = (["x"], [1.0])
    LoRAManager.deactivate(pipe)
    assert pipe.active == ([], [])

    class Broken:
        def set_adapters(self, *_a, **_k):
            raise RuntimeError("nope")

        def disable_lora(self):
            raise RuntimeError("nope")

    LoRAManager.deactivate(Broken())  # must not raise


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

    assert SDXLBackend.supports_loras is True
    for be in (SD15Backend, ZImageBackend, FluxBackend, ChromaBackend,
               QwenImageBackend, AnimaBackend, Krea2Backend):
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

        def apply(self, pipe, cfgs):
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
        def apply(self, pipe, cfgs):
            raise FileNotFoundError("no such lora")

    engine._loras = FailingLoras()
    result = engine.generate(model="m", prompt="cat", seed=1, batch=2,
                             loras=[{"source": str(f)}])
    assert not result.ok
    assert result.media == [None, None]
    assert "Failed to load LoRA" in result.errors[0].error
