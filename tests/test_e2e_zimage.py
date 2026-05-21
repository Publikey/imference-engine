"""End-to-end Z-Image inference smoke test.

Skipped unless IMFERENCE_TEST_ZIMAGE_PATH points to a local .safetensors file.
Requires a GPU with enough VRAM (Z-Image base ~12 GB, Z-Image Turbo varies).

If your checkpoint is a transformer-only finetune that needs shared components
from a base HF repo, also set IMFERENCE_TEST_ZIMAGE_BASE — e.g.
`Tongyi-MAI/Z-Image-Turbo`.

Run:
    set IMFERENCE_TEST_ZIMAGE_PATH=C:\\path\\to\\z-image-turbo.safetensors
    set IMFERENCE_TEST_ZIMAGE_BASE=Tongyi-MAI/Z-Image-Turbo
    python -m pytest tests/test_e2e_zimage.py -v -s
"""
from __future__ import annotations
import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_ZIMAGE_PATH")
BASE_MODEL = os.getenv("IMFERENCE_TEST_ZIMAGE_BASE") or None

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_ZIMAGE_PATH to a local .safetensors file to enable",
)


def test_zimage_generates_an_image(tmp_path):
    from PIL.Image import Image
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(device="auto")).load()
    engine.register_model(
        "test-zimage",
        backend="zimage",
        weights_path=MODEL_PATH,
        base_model=BASE_MODEL,
    )

    # Turbo-friendly defaults: 8 steps, guidance 1, shift 3.
    result = engine.generate(
        model="test-zimage",
        prompt="a cat sitting on a wooden table, photorealistic",
        negative_prompt="lowres, blurry, deformed",
        width=768,
        height=768,
        num_steps=8,
        guidance_scale=1.0,
        seed=42,
        batch=1,
        backend_options={"shift": 3.0},
    )

    assert not result.errors, f"unexpected errors: {result.errors}"
    assert len(result.images) == 1
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (768, 768)
    assert result.seeds == [42]

    out_path = tmp_path / "zimage_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved Z-Image smoke-test image: {out_path}")
