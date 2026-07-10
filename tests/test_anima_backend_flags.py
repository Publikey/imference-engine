"""No-torch assertions on the Anima backend (Modular Diffusers).

Anima is the one backend not on the standard pipeline API, so the guarantees
differ: img2img is unsupported, and it carries no opinionated engine defaults.
"""
from __future__ import annotations

import pytest

from imference_engine.anima import AnimaBackend


def test_engine_key():
    assert AnimaBackend.engine == "anima"


def test_no_engine_defaults_opinion():
    """Anima's recommended steps/guidance aren't documented → no opinion; the
    request/GLOBAL_DEFAULTS drive."""
    d = AnimaBackend().engine_defaults()
    assert d.num_steps is None
    assert d.guidance_scale is None
    assert d.scheduler is None
    assert d.backend_options == {}


def test_img2img_unsupported():
    with pytest.raises(NotImplementedError, match="no documented img2img"):
        AnimaBackend().make_img2img(object())


def test_encode_prompts_omits_empty_negative():
    """Only pass negative_prompt when provided (the doc example omits it)."""
    be = AnimaBackend()
    assert be.encode_prompts(pipe=None, prompt="a girl", negative_prompt=None) == {
        "prompt": "a girl"}
    assert be.encode_prompts(pipe=None, prompt="a girl", negative_prompt="") == {
        "prompt": "a girl"}
    assert be.encode_prompts(pipe=None, prompt="a girl", negative_prompt="blurry") == {
        "prompt": "a girl", "negative_prompt": "blurry"}


def test_build_inference_kwargs_standard_set():
    kw = AnimaBackend().build_inference_kwargs(
        width=1024, height=1024, num_steps=28, guidance_scale=6.0,
        clip_skip=None, chunk_size=1, generator=None)
    assert kw["num_inference_steps"] == 28
    assert kw["num_images_per_prompt"] == 1
    assert kw["width"] == 1024 and kw["height"] == 1024
    assert "clip_skip" not in kw  # Anima has none
    # Anima's modular __call__ ignores guidance_scale (separate Guider block) —
    # forwarding it triggers a diffusers "Unexpected input" warning, so it must
    # NOT be in the kwargs.
    assert "guidance_scale" not in kw


def test_registered_as_a_backend():
    from imference_engine import Engine, RuntimeConfig
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends.get("anima"), AnimaBackend)
