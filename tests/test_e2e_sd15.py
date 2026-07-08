"""End-to-end SD 1.5 inference smoke test.

Skipped unless IMFERENCE_TEST_SD15_PATH points to a local .safetensors file.
Requires a GPU and the [sd15] extras. SD 1.5 is light (~2 GB fp16).

Run:
    set IMFERENCE_TEST_SD15_PATH=C:\\path\\to\\sd15_model.safetensors
    python -m pytest tests/test_e2e_sd15.py -v -s
"""
from __future__ import annotations
import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_SD15_PATH")
pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_SD15_PATH to a local .safetensors file to enable",
)


def test_sd15_generates_an_image(tmp_path):
    from PIL.Image import Image
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(device="auto")).load()
    engine.register_model("test-sd15", backend="sd15", weights_path=MODEL_PATH)

    result = engine.generate(
        model="test-sd15",
        prompt="a cat sitting on a wooden table, photorealistic",
        negative_prompt="lowres, blurry, deformed",
        width=512,
        height=512,
        num_steps=25,
        guidance_scale=7.0,
        seed=42,
        batch=1,
    )

    assert not result.errors, f"unexpected errors: {result.errors}"
    assert len(result.images) == 1
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (512, 512)
    assert result.seeds == [42]

    out_path = tmp_path / "sd15_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved SD 1.5 smoke-test image: {out_path}")
