"""End-to-end Anima inference smoke test (Modular Diffusers).

Skipped unless IMFERENCE_TEST_ANIMA_REPO is set to a diffusers-format repo id or
local directory (e.g. `circlestone-labs/Anima-Base-v1.0-Diffusers`) — NOT a
single .safetensors. Requires a GPU and the [anima] extras.

This is the validation gate for the modular __call__ kwargs (see
imference_engine/anima/backend.py): if the pipeline rejects one of
num_inference_steps / guidance_scale / width / height / num_images_per_prompt /
negative_prompt, trim it in the backend and re-run.

Run:
    set IMFERENCE_TEST_ANIMA_REPO=circlestone-labs/Anima-Base-v1.0-Diffusers
    python -m pytest tests/test_e2e_anima.py -v -s
"""
from __future__ import annotations
import os

import pytest

ANIMA_REPO = os.getenv("IMFERENCE_TEST_ANIMA_REPO")
pytestmark = pytest.mark.skipif(
    not ANIMA_REPO,
    reason="set IMFERENCE_TEST_ANIMA_REPO to a diffusers-format repo/dir to enable",
)


def test_anima_generates_an_image(tmp_path):
    from PIL.Image import Image
    from imference_engine import Engine, RuntimeConfig

    engine = Engine(runtime=RuntimeConfig(device="auto")).load()
    engine.register_model("test-anima", backend="anima", weights_path=ANIMA_REPO)

    result = engine.generate(
        model="test-anima",
        prompt="masterpiece, best quality, 1girl, solo, city lights",
        width=1024,
        height=1024,
        num_steps=28,
        seed=42,
        batch=1,
    )

    assert not result.errors, f"unexpected errors: {result.errors}"
    assert len(result.images) == 1
    assert isinstance(result.images[0], Image)
    assert result.seeds == [42]

    out_path = tmp_path / "anima_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved Anima smoke-test image: {out_path}")
