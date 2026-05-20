"""Tests for device resolution. Runs without torch installed."""
import sys
import builtins
from unittest.mock import patch

from imference_engine.runtime.device import Device, resolve_device


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
