"""Wiring tests for SDXLBackend constructor flags.

These don't actually load a pipeline (that requires a real .safetensors
and a CUDA GPU). They check that the flag plumbing from RuntimeConfig
→ Engine.load() → SDXLBackend() is correct so a refactor doesn't silently
disable a perf opt the user opted into.
"""
from __future__ import annotations

from imference_engine.pipelines.sdxl import SDXLBackend


def test_sdxl_backend_default_no_tiny_vae():
    """Default construction = full VAE path (fp32 pre-cast)."""
    backend = SDXLBackend()
    assert backend._use_tiny_vae is False


def test_sdxl_backend_explicit_tiny_vae():
    """Explicit opt-in routes load_pipeline through TAESDxl swap."""
    backend = SDXLBackend(use_tiny_vae=True)
    assert backend._use_tiny_vae is True


def test_tiny_vae_repo_is_madebyollin_sdxl():
    """Pin the TAESD repo as a class constant so a typo in a future
    refactor doesn't silently pull the SD1.5 variant."""
    assert SDXLBackend.TINY_VAE_REPO == "madebyollin/taesdxl"


def test_engine_runtime_config_propagates_tiny_vae():
    """Engine(RuntimeConfig(use_tiny_vae=True)).load() must construct the
    SDXLBackend with the flag set — otherwise the user's setting is lost."""
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(device="cpu", use_tiny_vae=True)).load()
    sdxl = engine._backends["sdxl"]
    assert sdxl._use_tiny_vae is True

    # And the default flow keeps it off.
    engine2 = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert engine2._backends["sdxl"]._use_tiny_vae is False


def test_engine_runtime_config_propagates_cpu_offload():
    """enable_cpu_offload flag flows from RuntimeConfig into the ModelManager
    so _swap_pipe_to_gpu takes the accelerate path instead of pipe.to(device)."""
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(device="cpu", enable_cpu_offload=True)).load()
    assert engine._models._enable_cpu_offload is True

    engine2 = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert engine2._models._enable_cpu_offload is False


def test_cpu_offload_forces_max_cpu_models_zero():
    """The CPU LRU tier would shadow accelerate's hook-based offloader.
    Engine.load() must veto a misconfiguration that requests both."""
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(
        device="cpu",
        enable_cpu_offload=True,
        max_cpu_models=4,  # user mis-set this; engine should silently zero it
    )).load()
    assert engine._models._max_cpu == 0
    assert engine._models._enable_cpu_offload is True
