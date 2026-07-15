"""Tests for device resolution. Runs without torch installed."""
import sys
import builtins
import types

from imference_engine.runtime.device import Device, is_rocm, resolve_device


def _fake_torch(hip=None, cuda_available=True):
    """Minimal torch stand-in: cuda availability + version.hip marker."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    torch.version = types.SimpleNamespace(hip=hip)
    return torch


def test_device_torch_str_cuda():
    assert Device(kind="cuda", index=0).torch_str == "cuda:0"
    assert Device(kind="cuda", index=1).torch_str == "cuda:1"


def test_device_torch_str_other():
    assert Device(kind="mps").torch_str == "mps"
    assert Device(kind="cpu").torch_str == "cpu"


def test_resolve_falls_back_to_cpu_without_torch(monkeypatch):
    """If torch can't be imported, we should land on CPU silently."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return original_import(name, *args, **kwargs)

    # Pop any cached torch import so the patched __import__ gets the call
    sys.modules.pop("torch", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert resolve_device("auto") == Device(kind="cpu")
    assert resolve_device("cuda") == Device(kind="cpu")


def test_rocm_resolves_as_cuda(monkeypatch):
    """ROCm torch masquerades as CUDA: same kind/torch_str, backend='rocm'."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hip="6.4.1"))

    assert is_rocm()
    dev = resolve_device("auto")
    assert dev == Device(kind="cuda", index=0)
    assert dev.torch_str == "cuda:0"  # HIP keeps the cuda device string
    assert dev.backend == "rocm"
    assert resolve_device("cuda:1").backend == "rocm"


def test_nvidia_backend_stays_cuda(monkeypatch):
    """On a CUDA (non-HIP) build, backend matches kind."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hip=None))

    assert not is_rocm()
    dev = resolve_device("cuda")
    assert dev.kind == "cuda"
    assert dev.backend == "cuda"


def test_backend_without_torch(monkeypatch):
    """backend never crashes when torch is missing (catalog-only installs)."""
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return original_import(name, *args, **kwargs)

    sys.modules.pop("torch", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert not is_rocm()
    assert Device(kind="cuda").backend == "cuda"
    assert Device(kind="cpu").backend == "cpu"
