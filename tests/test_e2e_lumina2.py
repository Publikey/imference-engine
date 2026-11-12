"""End-to-end Lumina-Image 2.0 smoke test (text-to-image).

Skipped unless IMFERENCE_TEST_LUMINA2_PATH is set: a local transformer file
(transformer-only or all-in-one `.safetensors`, or a `.gguf`), or `auto` to let
the backend pick and fetch the build for this machine.
IMFERENCE_TEST_LUMINA2_BASE overrides the base repo (default
Alpha-VLLM/Lumina-Image-2.0 — Apache-2.0, ungated; Gemma encoder under the
Gemma Terms of Use).

Run:
    set IMFERENCE_TEST_LUMINA2_PATH=auto
    python -m pytest tests/test_e2e_lumina2.py -v -s
"""
from __future__ import annotations

import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_LUMINA2_PATH")
BASE_MODEL = os.getenv("IMFERENCE_TEST_LUMINA2_BASE") or None

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_LUMINA2_PATH (a local file, or 'auto') to enable",
)


def test_lumina2_generates(tmp_path):
    import torch
    from PIL.Image import Image

    from imference_engine import Engine, RuntimeConfig
    from imference_engine.lumina2 import recommend_build

    vram = None
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    build = recommend_build(vram)
    engine = Engine(runtime=RuntimeConfig(device="auto", **build.runtime_kwargs())).load()
    engine.register_model(
        "test-lumina", backend="lumina2", weights_path=MODEL_PATH, base_model=BASE_MODEL)

    # Engine defaults carry the recipe (30 steps, true CFG 4.0).
    result = engine.generate(
        model="test-lumina",
        prompt="a red fox in a snowy forest, golden hour, sharp focus",
        negative_prompt="blurry, low quality",
        width=1024, height=1024, seed=42)
    assert not result.errors, f"unexpected errors: {result.errors}"
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (1024, 1024)
    assert result.seeds == [42]

    out_path = tmp_path / "lumina2_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved Lumina-Image 2.0 smoke-test image: {out_path}")
