"""End-to-end SDXL inference smoke test.

Skipped unless IMFERENCE_TEST_SDXL_PATH points to a local .safetensors file
(absolute path). Requires a GPU and the [runtime] extras installed:

    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install -e ".[runtime,dev]"
    set IMFERENCE_TEST_SDXL_PATH=C:\\path\\to\\sdxl_base.safetensors
    python -m pytest tests/test_e2e_sdxl.py -v -s
"""
from __future__ import annotations
import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_SDXL_PATH")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_SDXL_PATH to a local .safetensors file to enable",
)


def test_sdxl_generates_an_image(tmp_path):
    from PIL.Image import Image
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(device="auto")).load()
    engine.register_model("test-sdxl", backend="sdxl", weights_path=MODEL_PATH)

    result = engine.generate(
        model="test-sdxl",
        prompt="a cat sitting on a wooden table, photorealistic",
        negative_prompt="lowres, blurry, deformed",
        width=768,
        height=768,
        num_steps=15,
        guidance_scale=6.0,
        seed=42,
        batch=1,
    )

    assert not result.errors, f"unexpected errors: {result.errors}"
    assert len(result.images) == 1
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (768, 768)
    assert result.seeds == [42]

    out_path = tmp_path / "sdxl_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved smoke-test image: {out_path}")


def test_sdxl_seed_increments_for_batch():
    """batch=N with explicit seed=S should produce seeds [S, S+1, ..., S+N-1]."""
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(device="auto")).load()
    engine.register_model("test-sdxl", backend="sdxl", weights_path=MODEL_PATH)

    result = engine.generate(
        model="test-sdxl",
        prompt="abstract painting",
        width=768,
        height=768,
        num_steps=10,
        guidance_scale=5.0,
        seed=100,
        batch=2,
    )
    assert result.seeds == [100, 101]
    assert len(result.images) == 2
    assert not result.errors
