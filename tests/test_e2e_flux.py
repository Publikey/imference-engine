"""End-to-end FLUX inference smoke test.

Skipped unless IMFERENCE_TEST_FLUX_PATH points to a local .safetensors file.
FLUX.1 is a 12B transformer — needs a big GPU or cpu-offload. Most community
checkpoints are transformer-only finetunes: set IMFERENCE_TEST_FLUX_BASE to the
base repo that supplies CLIP-L + T5-XXL + VAE (e.g. `black-forest-labs/FLUX.1-dev`,
a GATED repo — accept its license / mirror it first).

Run:
    set IMFERENCE_TEST_FLUX_PATH=C:\\path\\to\\flux-dev.safetensors
    set IMFERENCE_TEST_FLUX_BASE=black-forest-labs/FLUX.1-dev
    python -m pytest tests/test_e2e_flux.py -v -s
"""
from __future__ import annotations
import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_FLUX_PATH")
BASE_MODEL = os.getenv("IMFERENCE_TEST_FLUX_BASE") or None

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_FLUX_PATH to a local .safetensors file to enable",
)


def test_flux_generates_an_image(tmp_path):
    from PIL.Image import Image
    from imference_engine import Engine, RuntimeConfig

    # cpu-offload keeps peak VRAM near the transformer alone on consumer GPUs.
    engine = Engine(runtime=RuntimeConfig(
        device="auto", enable_offload=True)).load()
    engine.register_model(
        "test-flux",
        backend="flux",
        weights_path=MODEL_PATH,
        base_model=BASE_MODEL,
    )

    # FLUX.1-dev defaults: guidance ~3.5 (embedded), ~28 steps, no negative.
    result = engine.generate(
        model="test-flux",
        prompt="a cat sitting on a wooden table, photorealistic, 35mm",
        width=1024,
        height=1024,
        num_steps=28,
        guidance_scale=3.5,
        seed=42,
        batch=1,
    )

    assert not result.errors, f"unexpected errors: {result.errors}"
    assert len(result.images) == 1
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (1024, 1024)
    assert result.seeds == [42]

    out_path = tmp_path / "flux_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved FLUX smoke-test image: {out_path}")
