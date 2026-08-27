"""Unit tests for offload_mode="group" (ModelManager group offloading).

No GPU required: torch is used only for nn.Module fakes and torch.device
strings; the diffusers call is stubbed via the module-level _leaf_offload
indirection. Verifies mode selection, the wiring recipe (compute block-level,
encoders leaf-level, VAE resident), the CUDA-only guard, the fallback to model
offload, and the env plumbing.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import imference_engine.managers.model as model_mod  # noqa: E402
from imference_engine.managers.model import ModelManager, RegisteredModel  # noqa: E402
from imference_engine.pipelines.base import PipelineBackend  # noqa: E402
from imference_engine.runtime.device import Device  # noqa: E402


class FakeCompute(nn.Module):
    """Stand-in transformer/unet — records enable_group_offload kwargs."""

    def __init__(self):
        super().__init__()
        self.group_offload_kwargs = None

    def enable_group_offload(self, **kwargs):
        self.group_offload_kwargs = kwargs


class FakeEncoder(nn.Module):
    pass


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.moved_to = None

    def to(self, device):  # noqa: A003
        self.moved_to = device
        return self

    def enable_slicing(self):
        pass

    def enable_tiling(self):
        pass


class FakePipe:
    def __init__(self):
        self.transformer = FakeCompute()
        self.text_encoder = FakeEncoder()
        self.vae = FakeVAE()
        self.tokenizer = object()  # non-nn.Module — must be skipped
        self.components = {
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
            "tokenizer": self.tokenizer,
            "scheduler": None,
        }
        self.model_offload_called = False
        self.to_calls: list[str] = []

    def enable_model_cpu_offload(self, device=None):
        self.model_offload_called = True

    def to(self, device):  # noqa: A003
        self.to_calls.append(device)
        return self


class FakeBackend(PipelineBackend):
    engine = "fake"

    def __init__(self, compute_attr="transformer"):
        self._compute_attr = compute_attr

    def load_pipeline(self, *, local_path, base_model=None):
        return FakePipe()

    def get_compute_module(self, pipe):
        return getattr(pipe, self._compute_attr, None)

    def make_img2img(self, t2i_pipe):
        raise NotImplementedError

    def encode_prompts(self, pipe, prompt, negative_prompt):
        raise NotImplementedError

    def apply_scheduler(self, pipe, scheduler, **kwargs):
        pass

    def build_inference_kwargs(self, **kwargs):
        return {}

    def make_generator(self, seed, device):
        return None


@pytest.fixture
def leaf_calls(monkeypatch):
    """Stub the diffusers apply_group_offloading indirection; records calls."""
    calls: list[tuple] = []
    monkeypatch.setattr(
        model_mod, "_leaf_offload",
        lambda module, **kw: calls.append((module, kw)))
    return calls


def _manager(mode: str, kind: str = "cuda") -> tuple[ModelManager, FakeBackend]:
    be = FakeBackend()
    dev = Device(kind=kind, index=0 if kind == "cuda" else None)
    mgr = ModelManager({"fake": be}, dev, enable_offload=True, offload_mode=mode)
    mgr.register(RegisteredModel(name="A", backend="fake", weights_path="A"))
    return mgr, be


def test_group_mode_wires_compute_encoders_and_vae(leaf_calls):
    mgr, _ = _manager("group")
    pipe, _ = mgr.get_or_load("A")

    # compute module: block-level, one block per group, CUDA stream prefetch
    kw = pipe.transformer.group_offload_kwargs
    assert kw is not None
    assert kw["offload_type"] == "block_level"
    assert kw["num_blocks_per_group"] == 1
    assert kw["use_stream"] is True
    # UNPINNED host buffers by default — pinning kills the process under a
    # container memlock cap and poisons the CUDA context on low-RAM Windows.
    assert kw["low_cpu_mem_usage"] is True
    assert kw["onload_device"] == torch.device("cuda:0")
    assert kw["offload_device"] == torch.device("cpu")

    # text encoder leaf-offloaded; tokenizer/scheduler untouched
    assert len(leaf_calls) == 1
    module, leaf_kw = leaf_calls[0]
    assert module is pipe.text_encoder
    assert leaf_kw["offload_type"] == "leaf_level"

    # VAE resident on the device, whole pipe NOT .to()'d, no model offload
    assert pipe.vae.moved_to == "cuda:0"
    assert pipe.model_offload_called is False
    assert "cuda:0" not in pipe.to_calls


def test_group_mode_falls_back_without_cuda(leaf_calls):
    mgr, _ = _manager("group", kind="cpu")
    pipe, _ = mgr.get_or_load("A")
    assert pipe.model_offload_called is True
    assert pipe.transformer.group_offload_kwargs is None
    assert leaf_calls == []


def test_group_mode_falls_back_when_no_compute_module(leaf_calls):
    be = FakeBackend(compute_attr="missing")
    dev = Device(kind="cuda", index=0)
    mgr = ModelManager({"fake": be}, dev, enable_offload=True, offload_mode="group")
    mgr.register(RegisteredModel(name="A", backend="fake", weights_path="A"))
    pipe, _ = mgr.get_or_load("A")
    assert pipe.model_offload_called is True


def test_model_mode_is_unchanged(leaf_calls):
    mgr, _ = _manager("model")
    pipe, _ = mgr.get_or_load("A")
    assert pipe.model_offload_called is True
    assert pipe.transformer.group_offload_kwargs is None
    assert leaf_calls == []


def test_unknown_mode_degrades_to_model():
    be = FakeBackend()
    dev = Device(kind="cuda", index=0)
    mgr = ModelManager({"fake": be}, dev, enable_offload=True, offload_mode="banana")
    assert mgr._offload_mode == "model"


def test_runtime_config_reads_env(monkeypatch):
    from imference_engine.engine import RuntimeConfig
    monkeypatch.setenv("IMAGE_OFFLOAD_MODE", "group")
    assert RuntimeConfig.from_env().offload_mode == "group"
    monkeypatch.delenv("IMAGE_OFFLOAD_MODE")
    assert RuntimeConfig.from_env().offload_mode == "model"


def test_krea2_fp8_storage_env_overrides(monkeypatch):
    """fp8-resident composes with group offloading (GPU-validated), so the
    auto no longer special-cases the offload mode — only the env forces it."""
    from imference_engine.krea2 import Krea2Backend
    monkeypatch.setenv("IMAGE_OFFLOAD_MODE", "group")
    monkeypatch.setenv("KREA2_FP8_STORAGE", "0")
    assert Krea2Backend._resolve_fp8_storage(source_was_fp8=True) is False
    monkeypatch.setenv("KREA2_FP8_STORAGE", "1")
    assert Krea2Backend._resolve_fp8_storage(source_was_fp8=True) is True
    # auto with a non-fp8 source stays off regardless of CUDA
    monkeypatch.delenv("KREA2_FP8_STORAGE")
    assert Krea2Backend._resolve_fp8_storage(source_was_fp8=False) is False
