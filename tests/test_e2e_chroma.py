"""End-to-end Chroma inference smoke test.

Skipped unless IMFERENCE_TEST_CHROMA_PATH points to a local .safetensors file.
Chroma is an 8.9B transformer — needs a big GPU or cpu-offload. Community
checkpoints are transformer-only: set IMFERENCE_TEST_CHROMA_BASE to a
diffusers-format base repo supplying T5-XXL + VAE (e.g. `lodestones/Chroma1-HD`).

Run:
    set IMFERENCE_TEST_CHROMA_PATH=C:\\path\\to\\chroma.safetensors
    set IMFERENCE_TEST_CHROMA_BASE=lodestones/Chroma1-HD
    python -m pytest tests/test_e2e_chroma.py -v -s
"""
from __future__ import annotations
import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_CHROMA_PATH")
BASE_MODEL = os.getenv("IMFERENCE_TEST_CHROMA_BASE") or None

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_CHROMA_PATH to a local .safetensors file to enable",
)


def test_chroma_generates_an_image(tmp_path):
    from PIL.Image import Image
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(
        device="auto", enable_offload=True)).load()
    engine.register_model(
        "test-chroma",
        backend="chroma",
        weights_path=MODEL_PATH,
        base_model=BASE_MODEL,
    )

    # Chroma is de-distilled → real CFG → negative prompt is honored.
    result = engine.generate(
        model="test-chroma",
        prompt="a cat sitting on a wooden table, photorealistic, 35mm",
        negative_prompt="lowres, blurry, deformed",
        width=1024,
        height=1024,
        num_steps=28,
        guidance_scale=4.0,
        seed=42,
        batch=1,
    )

    assert not result.errors, f"unexpected errors: {result.errors}"
    assert len(result.images) == 1
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (1024, 1024)
    assert result.seeds == [42]

    out_path = tmp_path / "chroma_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved Chroma smoke-test image: {out_path}")
