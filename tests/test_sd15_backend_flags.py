"""No-torch assertions on the SD 1.5 backend.

Guards the offline config mirror (config/tokenizer only, never weights), the
single-encoder / 512-native family defaults, and that it is a DISTINCT backend
from SDXL (own pipeline classes, own tiny VAE).
"""
from __future__ import annotations

from imference_engine.pipelines.sd15 import SD15Backend


def test_engine_key():
    assert SD15Backend.engine == "sd15"


def test_engine_defaults_512_native():
    d = SD15Backend().engine_defaults()
    assert d.width == 512 and d.height == 512
    assert d.guidance_scale == 7.0
    assert d.scheduler == "EulerAncestralDiscreteScheduler"


def test_config_patterns_never_pull_weights():
    pats = SD15Backend.CONFIG_PATTERNS
    assert "model_index.json" in pats and "scheduler/*" in pats
    assert "tokenizer/*" in pats
    # config.json glob only — never the unet/vae/text_encoder safetensors.
    assert "*/config.json" in pats
    assert not any("safetensors" in p for p in pats)


def test_distinct_tiny_vae_from_sdxl():
    """SD 1.5 uses taesd, SDXL uses taesdxl — must not be confused."""
    from imference_engine.pipelines.sdxl import SDXLBackend
    assert SD15Backend.TINY_VAE_REPO == "madebyollin/taesd"
    assert SD15Backend.TINY_VAE_REPO != SDXLBackend.TINY_VAE_REPO


def test_config_repo_is_maintained_mirror():
    # runwayml/stable-diffusion-v1-5 was delisted; use the community re-upload.
    assert SD15Backend.CONFIG_REPO == "stable-diffusion-v1-5/stable-diffusion-v1-5"


def test_build_inference_kwargs_supports_clip_skip():
    be = SD15Backend()
    kw = be.build_inference_kwargs(
        width=512, height=512, num_steps=25, guidance_scale=7.0,
        clip_skip=2, chunk_size=1, generator=None)
    assert kw["clip_skip"] == 2
    # clip_skip omitted when None/0
    kw2 = be.build_inference_kwargs(
        width=512, height=512, num_steps=25, guidance_scale=7.0,
        clip_skip=None, chunk_size=1, generator=None)
    assert "clip_skip" not in kw2


def test_registered_as_a_backend():
    from imference_engine import Engine, RuntimeConfig
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends.get("sd15"), SD15Backend)
