"""End-to-end LongCat-Image smoke test (text-to-image, then an edit).

Skipped unless IMFERENCE_TEST_LONGCAT=1. Both backends run with
``weights_path="auto"`` (LongCat-Image and LongCat-Image-Edit-Turbo, Apache-2.0,
ungated), fetched from IMAGE_MODEL_CDN or HuggingFace, with the storage build
recommended for this card.

Run:
    set IMFERENCE_TEST_LONGCAT=1
    python -m pytest tests/test_e2e_longcat.py -v -s
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("IMFERENCE_TEST_LONGCAT") != "1",
    reason="set IMFERENCE_TEST_LONGCAT=1 to enable (downloads ~42 GB on first run)",
)


def test_longcat_generates_then_edits(tmp_path):
    import torch
    from PIL.Image import Image

    from imference_engine import Engine, RuntimeConfig
    from imference_engine.longcat import recommend_build

    vram = None
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    build = recommend_build(vram)
    engine = Engine(runtime=RuntimeConfig(device="auto", **build.runtime_kwargs())).load()
    engine.register_model("lc", backend="longcat", weights_path="auto")
    engine.register_model("lc-edit", backend="longcat-edit", weights_path="auto")

    # 20 steps keep the smoke test short; the recipe is 50.
    result = engine.generate(
        model="lc", prompt="a red fox in a snowy forest, golden hour, sharp focus",
        width=1024, height=1024, num_steps=20, seed=42)
    assert not result.errors, f"unexpected errors: {result.errors}"
    assert isinstance(result.images[0], Image) and result.images[0].size == (1024, 1024)
    result.images[0].save(tmp_path / "longcat_t2i.png")

    # Edit-Turbo defaults (8 steps, guidance 1.0); output follows the input size.
    edit = engine.generate(
        model="lc-edit", prompt="make it night with a full moon, keep the fox unchanged",
        source_image=result.images[0], seed=42)
    assert not edit.errors, f"unexpected errors: {edit.errors}"
    assert edit.images[0].size == (1024, 1024)
    edit.images[0].save(tmp_path / "longcat_edit.png")
    print(f"\nSaved LongCat smoke-test images in {tmp_path}")
