"""Tests for the tiny SD 1.5 engine. Runs without torch/diffusers installed.

The engine imports torch/diffusers lazily inside load()/generate(), so we can
exercise its guard rails and defaults on a bare dev machine.
"""
import pytest

from imference_engine.sd15 import DEFAULT_MODEL, SD15Engine
from imference_engine.sd15.engine import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_WIDTH,
)


def test_defaults():
    assert DEFAULT_MODEL == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert (DEFAULT_WIDTH, DEFAULT_HEIGHT) == (512, 512)
    assert DEFAULT_NUM_INFERENCE_STEPS == 25
    assert DEFAULT_GUIDANCE_SCALE == 7.5


def test_construct_does_not_import_torch():
    """Constructing the engine must not require the heavy runtime deps."""
    engine = SD15Engine()
    assert engine.model_id == DEFAULT_MODEL
    assert engine._pipe is None


def test_generate_before_load_raises():
    engine = SD15Engine()
    with pytest.raises(RuntimeError):
        engine.generate(prompt="anything")
