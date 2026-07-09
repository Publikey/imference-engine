"""End-to-end Qwen-Image inference smoke test.

Skipped unless IMFERENCE_TEST_QWENIMAGE_PATH points to a local .safetensors file.
Qwen-Image is a 20B transformer — needs a big GPU or cpu-offload (+ ideally a
quantized build). Community checkpoints are transformer-only: set
IMFERENCE_TEST_QWENIMAGE_BASE to the diffusers-format base repo supplying the
Qwen2.5-VL encoder + VAE (e.g. `Qwen/Qwen-Image`).

Run:
    set IMFERENCE_TEST_QWENIMAGE_PATH=C:\\path\\to\\qwen-image.safetensors
    set IMFERENCE_TEST_QWENIMAGE_BASE=Qwen/Qwen-Image
    python -m pytest tests/test_e2e_qwenimage.py -v -s
"""
from __future__ import annotations
import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_QWENIMAGE_PATH")
BASE_MODEL = os.getenv("IMFERENCE_TEST_QWENIMAGE_BASE") or None

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_QWENIMAGE_PATH to a local .safetensors file to enable",
)


def test_qwenimage_generates_an_image(tmp_path):
    from PIL.Image import Image
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(
        device="auto", enable_offload=True)).load()
    engine.register_model(
        "test-qwenimage",
        backend="qwenimage",
        weights_path=MODEL_PATH,
        base_model=BASE_MODEL,
    )

    # Qwen-Image excels at text rendering; true CFG with a negative prompt.
    result = engine.generate(
        model="test-qwenimage",
        prompt='a bookstore window with a sign that reads "Imference Engine"',
        negative_prompt=" ",
        width=1024,
        height=1024,
        num_steps=40,
        guidance_scale=4.0,
        seed=42,
        batch=1,
    )

    assert not result.errors, f"unexpected errors: {result.errors}"
    assert len(result.images) == 1
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (1024, 1024)
    assert result.seeds == [42]

    out_path = tmp_path / "qwenimage_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved Qwen-Image smoke-test image: {out_path}")
