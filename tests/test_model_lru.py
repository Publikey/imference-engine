"""Unit tests for the multi-tier ModelManager LRU.

No GPU required: we mock PipelineBackend.load_pipeline to return a FakePipe
whose `.to(device)` is a no-op that just records the transition. Counts
of load_pipeline calls tell us whether we hit the cache or paid the disk
read again.
"""
from __future__ import annotations

import pytest

from imference_engine.managers.model import ModelManager, RegisteredModel
from imference_engine.pipelines.base import PipelineBackend
from imference_engine.runtime.device import Device


class FakePipe:
    """Stand-in for a diffusers pipeline. Tracks device residency so
    tests can assert that GPU→CPU swaps don't lose references."""

    def __init__(self, name: str):
        self.name = name
        self.transitions: list[str] = []  # ordered list of device targets seen
        # Mimic the .vae attribute the manager touches for slicing/tiling.
        self.vae = _FakeVAE()

    def to(self, device: str):
        self.transitions.append(device)
        return self


class _FakeVAE:
    def enable_slicing(self):
        pass

    def disable_slicing(self):
        pass

    def enable_tiling(self):
        pass

    def disable_tiling(self):
        pass


class FakeBackend(PipelineBackend):
    """Minimal backend that hands out FakePipes and counts load_pipeline
    calls per model name — that's our oracle for "did we re-read disk?"."""

    engine = "fake"

    def __init__(self):
        self.load_calls: dict[str, int] = {}

    def load_pipeline(self, *, local_path, base_model=None):
        name = local_path  # tests pass the model name as weights_path
        self.load_calls[name] = self.load_calls.get(name, 0) + 1
        return FakePipe(name)

    # The rest of the ABC isn't exercised here.
    def make_img2img(self, t2i_pipe):
        raise NotImplementedError

    def get_compute_module(self, pipe):
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
def device():
    # Pretend we have CUDA so torch_str is "cuda:0" — the FakePipe doesn't
    # care, but it matches what a real run would look like.
    return Device(kind="cuda", index=0)


@pytest.fixture
def backends():
    return {"fake": FakeBackend()}


def _register(mgr: ModelManager, name: str):
    """Convenience: register `name` with weights_path == name (so the
    FakeBackend can identify it without filesystem dependencies)."""
    mgr.register(RegisteredModel(name=name, backend="fake", weights_path=name))


# ---------------------------------------------------------------------------
# Single-resident behavior (defaults — matches the desktop sidecar).
# ---------------------------------------------------------------------------


def test_default_single_resident_loads_once(device, backends):
    """First get_or_load reads from disk; second hits the GPU cache."""
    mgr = ModelManager(backends, device)
    _register(mgr, "A")

    pipe1, _ = mgr.get_or_load("A")
    pipe2, _ = mgr.get_or_load("A")

    assert pipe1 is pipe2
    assert backends["fake"].load_calls == {"A": 1}
    assert mgr.gpu_resident() == ["A"]
    assert mgr.cpu_resident() == []


def test_default_single_resident_swap_drops_old(device, backends):
    """With max_cpu=0, evicted-from-GPU pipes are dropped, not cached.
    A second visit to the original model triggers a fresh disk read."""
    mgr = ModelManager(backends, device, max_gpu_models=1, max_cpu_models=0)
    _register(mgr, "A")
    _register(mgr, "B")

    mgr.get_or_load("A")
    mgr.get_or_load("B")
    mgr.get_or_load("A")  # fresh disk read because CPU tier is disabled

    assert backends["fake"].load_calls == {"A": 2, "B": 1}
    assert mgr.gpu_resident() == ["A"]
    assert mgr.cpu_resident() == []


# ---------------------------------------------------------------------------
# Multi-tier behavior (worker config).
# ---------------------------------------------------------------------------


def test_cpu_repromotion_skips_disk_read(device, backends):
    """The point of the CPU tier: alternating A/B/A doesn't reload A
    when A fits in CPU RAM."""
    mgr = ModelManager(backends, device, max_gpu_models=1, max_cpu_models=2)
    _register(mgr, "A")
    _register(mgr, "B")

    pipe_a_first, _ = mgr.get_or_load("A")
    mgr.get_or_load("B")            # A demoted to CPU
    pipe_a_second, _ = mgr.get_or_load("A")  # A promoted from CPU

    assert backends["fake"].load_calls == {"A": 1, "B": 1}
    assert pipe_a_first is pipe_a_second  # same pipe object, ref preserved
    assert mgr.gpu_resident() == ["A"]
    assert mgr.cpu_resident() == ["B"]


def test_cascade_eviction_drops_oldest_cpu(device, backends):
    """max_gpu=1, max_cpu=1: A→B→C should park B on CPU and DROP A entirely."""
    mgr = ModelManager(backends, device, max_gpu_models=1, max_cpu_models=1)
    _register(mgr, "A")
    _register(mgr, "B")
    _register(mgr, "C")

    mgr.get_or_load("A")
    mgr.get_or_load("B")  # A → CPU
    mgr.get_or_load("C")  # B → CPU, A dropped (CPU was at cap)

    assert mgr.gpu_resident() == ["C"]
    assert mgr.cpu_resident() == ["B"]

    # Visiting A again pays a full disk read.
    mgr.get_or_load("A")
    assert backends["fake"].load_calls == {"A": 2, "B": 1, "C": 1}


def test_gpu_lru_uses_oldest_for_eviction(device, backends):
    """max_gpu=2: when full, the LEAST recently touched resident gets evicted."""
    mgr = ModelManager(backends, device, max_gpu_models=2, max_cpu_models=2)
    _register(mgr, "A")
    _register(mgr, "B")
    _register(mgr, "C")

    mgr.get_or_load("A")  # GPU: [A]
    mgr.get_or_load("B")  # GPU: [A, B]
    mgr.get_or_load("A")  # touch A → GPU LRU: [B, A]
    mgr.get_or_load("C")  # B is LRU now → B → CPU; GPU: [A, C]

    assert mgr.gpu_resident() == ["A", "C"]
    assert mgr.cpu_resident() == ["B"]


# ---------------------------------------------------------------------------
# Lifecycle hooks.
# ---------------------------------------------------------------------------


def test_on_loaded_fires_once_per_disk_read(device, backends):
    loaded: list[str] = []
    mgr = ModelManager(
        backends, device,
        max_gpu_models=1,
        max_cpu_models=2,
        on_loaded=loaded.append,
    )
    _register(mgr, "A")
    _register(mgr, "B")

    mgr.get_or_load("A")          # disk read → fires
    mgr.get_or_load("A")          # cache hit → no fire
    mgr.get_or_load("B")          # disk read → fires
    mgr.get_or_load("A")          # CPU re-promotion → no fire (no disk read)

    assert loaded == ["A", "B"]


def test_on_evicted_fires_when_pipe_leaves_memory(device, backends):
    evicted: list[str] = []
    mgr = ModelManager(
        backends, device,
        max_gpu_models=1,
        max_cpu_models=1,
        on_evicted=evicted.append,
    )
    _register(mgr, "A")
    _register(mgr, "B")
    _register(mgr, "C")

    mgr.get_or_load("A")  # nothing evicted yet
    mgr.get_or_load("B")  # A → CPU (still in memory, no eviction)
    assert evicted == []

    mgr.get_or_load("C")  # B → CPU, A dropped → fires
    assert evicted == ["A"]


def test_on_evicted_fires_on_max_cpu_zero(device, backends):
    """With CPU tier disabled, evicted-from-GPU pipes fire on_evicted
    immediately (since they're dropped right away, not parked)."""
    evicted: list[str] = []
    mgr = ModelManager(
        backends, device,
        max_gpu_models=1,
        max_cpu_models=0,
        on_evicted=evicted.append,
    )
    _register(mgr, "A")
    _register(mgr, "B")

    mgr.get_or_load("A")
    mgr.get_or_load("B")  # A dropped → fires
    assert evicted == ["A"]


def test_hooks_swallow_exceptions(device, backends):
    """A misbehaving hook shouldn't break the manager."""
    def bad_loaded(name):
        raise RuntimeError("boom")

    mgr = ModelManager(backends, device, on_loaded=bad_loaded)
    _register(mgr, "A")

    # Should NOT raise — exception is logged and swallowed.
    pipe, _ = mgr.get_or_load("A")
    assert pipe is not None


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_unregistered_model_raises_keyerror(device, backends):
    mgr = ModelManager(backends, device)
    with pytest.raises(KeyError):
        mgr.get_or_load("nonexistent")


def test_register_unknown_backend_raises(device, backends):
    mgr = ModelManager(backends, device)
    with pytest.raises(ValueError, match="Unknown backend"):
        mgr.register(RegisteredModel(name="X", backend="ghost", weights_path="X"))


def test_max_gpu_clamped_to_at_least_one(device, backends):
    """A misconfigured max_gpu_models=0 silently becomes 1 (the manager
    needs at least one GPU slot to be useful at all)."""
    mgr = ModelManager(backends, device, max_gpu_models=0, max_cpu_models=0)
    _register(mgr, "A")
    mgr.get_or_load("A")
    assert mgr.gpu_resident() == ["A"]
