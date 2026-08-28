"""End-to-end Krea 2 (Turbo) inference smoke test.

Skipped unless IMFERENCE_TEST_KREA2_PATH points to a local .safetensors file —
a civitai/ComfyUI transformer-only checkpoint (native keys, fp8_scaled or bf16;
e.g. Comfy-Org/Krea-2 `diffusion_models/krea2_turbo_fp8_scaled.safetensors`).
IMFERENCE_TEST_KREA2_BASE must name the diffusers base repo supplying the
Qwen3-VL encoder + VAE + tokenizer + scheduler (`krea/Krea-2-Turbo` — GATED,
accept the Krea 2 Community License / mirror it first).

Run:
    set IMFERENCE_TEST_KREA2_PATH=C:\\path\\to\\krea2_turbo_fp8_scaled.safetensors
    set IMFERENCE_TEST_KREA2_BASE=krea/Krea-2-Turbo
    python -m pytest tests/test_e2e_krea2.py -v -s
"""
from __future__ import annotations
import os

import pytest

MODEL_PATH = os.getenv("IMFERENCE_TEST_KREA2_PATH")
BASE_MODEL = os.getenv("IMFERENCE_TEST_KREA2_BASE") or None

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set IMFERENCE_TEST_KREA2_PATH to a local .safetensors file to enable",
)


def test_krea2_generates_an_image(tmp_path):
    from PIL.Image import Image
    from imference_engine import Engine, RuntimeConfig

    # fp8-resident storage (auto for fp8 checkpoints) keeps the transformer
    # ~13 GB; cpu-offload caps peak VRAM near the largest module.
    engine = Engine(runtime=RuntimeConfig(
        device="auto", enable_offload=True)).load()
    engine.register_model(
        "test-krea2",
        backend="krea2",
        weights_path=MODEL_PATH,
        base_model=BASE_MODEL,
    )

    # Turbo recipe: 8 steps, guidance 0.0 (TDM-distilled, Krea CFG convention).
    result = engine.generate(
        model="test-krea2",
        prompt="a red fox in a snowy forest, golden hour, sharp focus",
        width=1024,
        height=1024,
        num_steps=8,
        guidance_scale=0.0,
        seed=42,
        batch=1,
    )

    assert not result.errors, f"unexpected errors: {result.errors}"
    assert len(result.images) == 1
    assert isinstance(result.images[0], Image)
    assert result.images[0].size == (1024, 1024)
    assert result.seeds == [42]

    out_path = tmp_path / "krea2_smoke.png"
    result.images[0].save(out_path)
    print(f"\nSaved Krea 2 smoke-test image: {out_path}")
