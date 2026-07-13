"""Unit tests for the defaults precedence chain (catalog/defaults.py).

Covers the merge semantics in isolation (no Engine, no torch): `self` wins where
set, a `None` field never shadows a lower layer, and `backend_options` merges
key-wise. Also pins the opinionated engine-layer defaults on the real backends.
"""
from __future__ import annotations

from imference_engine.catalog.defaults import GLOBAL_DEFAULTS, GenerationDefaults


def test_merged_over_self_wins_where_set():
    base = GenerationDefaults(num_steps=28, guidance_scale=6.0, width=1024)
    top = GenerationDefaults(num_steps=8)  # only num_steps set
    out = top.merged_over(base)
    assert out.num_steps == 8          # top wins
    assert out.guidance_scale == 6.0   # base fills
    assert out.width == 1024           # base fills


def test_none_does_not_shadow_lower_layer():
    """A None field on the higher layer must NOT overwrite the base value —
    this is the whole point of Optional request params."""
    base = GenerationDefaults(guidance_scale=6.0)
    top = GenerationDefaults(guidance_scale=None)
    assert top.merged_over(base).guidance_scale == 6.0


def test_backend_options_merge_key_wise():
    base = GenerationDefaults(backend_options={"shift": 3.0, "keep": 1})
    top = GenerationDefaults(backend_options={"shift": 5.0, "extra": 2})
    merged = top.merged_over(base).backend_options
    assert merged == {"shift": 5.0, "keep": 1, "extra": 2}  # top overrides per key


def test_merged_over_does_not_mutate_inputs():
    base = GenerationDefaults(num_steps=28, backend_options={"a": 1})
    top = GenerationDefaults(num_steps=8, backend_options={"b": 2})
    top.merged_over(base)
    assert base.num_steps == 28 and base.backend_options == {"a": 1}
    assert top.num_steps == 8 and top.backend_options == {"b": 2}


def test_full_chain_request_beats_model_beats_engine():
    """engine < model < request, resolved exactly as Engine.generate does."""
    engine = GenerationDefaults(scheduler="EulerAncestralDiscreteScheduler")
    model = GenerationDefaults(num_steps=8, backend_options={"shift": 3.0})
    request = GenerationDefaults(num_steps=50)  # overrides the model's 8

    eff = request.merged_over(model.merged_over(engine)).merged_over(GLOBAL_DEFAULTS)

    assert eff.num_steps == 50                              # request wins
    assert eff.backend_options == {"shift": 3.0}            # model fills
    assert eff.scheduler == "EulerAncestralDiscreteScheduler"  # engine fills
    assert eff.guidance_scale == 6.0                        # GLOBAL_DEFAULTS fills
    assert eff.width == 1024 and eff.height == 1024         # GLOBAL_DEFAULTS fills
    assert eff.strength == 0.75                             # GLOBAL_DEFAULTS fills


def test_global_defaults_values():
    assert GLOBAL_DEFAULTS.num_steps == 28
    assert GLOBAL_DEFAULTS.guidance_scale == 6.0
    assert GLOBAL_DEFAULTS.width == 1024
    assert GLOBAL_DEFAULTS.height == 1024
    assert GLOBAL_DEFAULTS.strength == 0.75
    # These stay None on purpose — backends handle None downstream.
    assert GLOBAL_DEFAULTS.scheduler is None
    assert GLOBAL_DEFAULTS.clip_skip is None
    assert GLOBAL_DEFAULTS.negative_prompt is None


def test_sdxl_engine_defaults():
    from imference_engine.pipelines.sdxl import SDXLBackend
    d = SDXLBackend().engine_defaults()
    assert d.scheduler == "EulerAncestralDiscreteScheduler"
    # SDXL has no opinion on these — they fall through to GLOBAL_DEFAULTS.
    assert d.num_steps is None
    assert d.guidance_scale is None


def test_zimage_engine_defaults():
    from imference_engine.zimage.backend import ZImageBackend
    d = ZImageBackend().engine_defaults()
    assert d.guidance_scale == 1.0          # flow-matching family norm
    assert d.scheduler is None              # ignored by this backend
    assert "shift" not in d.backend_options  # shift is a per-model knob, not engine
