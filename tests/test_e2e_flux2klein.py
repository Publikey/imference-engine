"""End-to-end FLUX.2 klein 4B smoke test (text-to-image + reference edit).

Skipped unless IMFERENCE_TEST_FLUX2KLEIN_PATH is set: a local transformer file
(`flux-2-klein-4b.safetensors`, the fp8 file, or an unsloth `.gguf`), or
`auto` to let the backend pick and fetch the build for this machine.
IMFERENCE_TEST_FLUX2KLEIN_BASE overrides the base repo
(default black-forest-labs/FLUX.2-klein-4B — Apache-2.0, ungated).

Run:
    set IMFERENCE_TEST_FLUX2KLEIN_PATH=auto
    python -m pytest tests/test_e2e_flux2klein.py -v -s
"""
from __future__ import annotations

import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_FLUX2KLEIN_PATH")
BASE_MODEL = os.getenv("IMFERENCE_TEST_FLUX2KLEIN_BASE") or None

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_FLUX2KLEIN_PATH (a local file, or 'auto') to enable",
)


def test_flux2klein_generates_and_edits(tmp_path):
    import torch
    from PIL.Image import Image

    from imference_engine import Engine, RuntimeConfig
    from imference_engine.flux2klein import recommend_build

    vram = None
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1 << 30)
    build = recommend_build(vram)
    engine = Engine(runtime=RuntimeConfig(device="auto", **build.runtime_kwargs())).load()
    engine.register_model(
        "test-klein", backend="flux2-klein", weights_path=MODEL_PATH, base_model=BASE_MODEL)

    # Engine defaults carry the distilled recipe (4 steps, guidance 1.0).
    result = engine.generate(
        model="test-klein",
        prompt="a red fox in a snowy forest, golden hour, sharp focus",
        width=1024, height=1024, seed=42)
    assert not result.errors, f"unexpected errors: {result.errors}"
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (1024, 1024)
    assert result.seeds == [42]
    result.images[0].save(tmp_path / "klein_t2i.png")

    # Reference edit: same pipe, output takes the reference's size.
    source = result.images[0].resize((1024, 768))
    edit = engine.generate(
        model="test-klein", prompt="make it night with a full moon, keep the fox",
        source_image=source, seed=42)
    assert not edit.errors, f"unexpected errors: {edit.errors}"
    assert edit.images[0].size == (1024, 768)
    edit.images[0].save(tmp_path / "klein_edit.png")
    print(f"\nSaved FLUX.2 klein smoke-test images in {tmp_path}")
