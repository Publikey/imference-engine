"""End-to-end ERNIE-Image-Turbo smoke test (text-to-image).

Skipped unless IMFERENCE_TEST_ERNIE_PATH is set: a local transformer file (an
unsloth `.gguf`, or the Comfy-Org `ernie-image-turbo.safetensors`), or `auto`
to let the backend pick and fetch the build for this machine.
IMFERENCE_TEST_ERNIE_BASE overrides the base repo (default
baidu/ERNIE-Image-Turbo — Apache-2.0, ungated).

Run:
    set IMFERENCE_TEST_ERNIE_PATH=auto
    python -m pytest tests/test_e2e_ernieimage.py -v -s
"""
from __future__ import annotations

import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_ERNIE_PATH")
BASE_MODEL = os.getenv("IMFERENCE_TEST_ERNIE_BASE") or None

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_ERNIE_PATH (a local file, or 'auto') to enable",
)


def test_ernie_image_generates(tmp_path):
    import torch
    from PIL.Image import Image

    from imference_engine import Engine, RuntimeConfig
    from imference_engine.ernieimage import recommend_build

    vram = None
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    build = recommend_build(vram)
    engine = Engine(runtime=RuntimeConfig(device="auto", **build.runtime_kwargs())).load()
    engine.register_model(
        "test-ernie", backend="ernie-image", weights_path=MODEL_PATH, base_model=BASE_MODEL)

    # Engine defaults carry the Turbo recipe (8 steps, guidance 1.0).
    result = engine.generate(
        model="test-ernie",
        prompt='a storefront with a neon sign that reads "OPEN 24 HOURS", rainy night',
        width=1024, height=1024, seed=42)
    assert not result.errors, f"unexpected errors: {result.errors}"
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (1024, 1024)
    assert result.seeds == [42]

    out_path = tmp_path / "ernie_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved ERNIE-Image smoke-test image: {out_path}")
