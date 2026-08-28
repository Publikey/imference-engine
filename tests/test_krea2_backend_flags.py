"""No-torch assertions on the Krea 2 backend: engine defaults, the offline
base-component mirror, registration through the Engine, and the t2i-only /
base_model-required contracts.
"""
from __future__ import annotations

import pytest

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.krea2 import Krea2Backend


def test_engine_key():
    assert Krea2Backend.engine == "krea2"


def test_engine_defaults_are_turbo_family_norm():
    d = Krea2Backend().engine_defaults()
    assert d.guidance_scale == 0.0   # TDM-distilled: guidance OFF (Krea convention)
    assert d.num_steps == 8          # Turbo recipe
    assert d.scheduler is None       # flow matching, name ignored
    assert d.backend_options == {}


def test_base_patterns_cover_shared_components_config_only_transformer():
    pats = Krea2Backend.BASE_PATTERNS
    assert "model_index.json" in pats and "scheduler/*" in pats
    # Qwen3-VL-4B encoder + tokenizer + Qwen-Image VAE = full weights
    assert "text_encoder/*" in pats and "tokenizer/*" in pats and "vae/*" in pats
    # transformer = config only (the ~13-26 GB weights come from the checkpoint)
    assert "transformer/config.json" in pats
    assert "transformer/*" not in pats


def test_load_pipeline_requires_base_model():
    with pytest.raises(ValueError, match="base_model"):
        Krea2Backend().load_pipeline(local_path="x.safetensors", base_model=None)


def test_img2img_is_unsupported():
    with pytest.raises(NotImplementedError, match="img2img"):
        Krea2Backend().make_img2img(t2i_pipe=None)


def test_encode_prompts_passes_negative_through():
    """Negative passes through (the pipeline itself ignores it at guidance<=0)."""
    be = Krea2Backend()
    assert be.encode_prompts(pipe=None, prompt="a fox", negative_prompt=None) == {
        "prompt": "a fox"
    }
    out = be.encode_prompts(pipe=None, prompt="a fox", negative_prompt="blurry")
    assert out == {"prompt": "a fox", "negative_prompt": "blurry"}


def test_build_inference_kwargs_t2i():
    be = Krea2Backend()
    kw = be.build_inference_kwargs(
        width=1024, height=768, num_steps=8, guidance_scale=0.0,
        clip_skip=None, chunk_size=2, generator=None)
    assert kw["num_inference_steps"] == 8
    assert kw["guidance_scale"] == 0.0
    assert kw["width"] == 1024 and kw["height"] == 768
    assert kw["num_images_per_prompt"] == 2
    assert "clip_skip" not in kw  # Krea 2 has none


def test_apply_scheduler_is_a_noop():
    be = Krea2Backend()
    be.apply_scheduler(pipe=None, scheduler="euler_a")  # must not raise
    be.apply_scheduler(pipe=None, scheduler=None, shift=3.0)  # shift ignored too


def test_fp8_storage_env_override(monkeypatch):
    be = Krea2Backend()
    monkeypatch.setenv("KREA2_FP8_STORAGE", "1")
    assert be._resolve_fp8_storage(source_was_quantized=False) is True
    monkeypatch.setenv("KREA2_FP8_STORAGE", "0")
    assert be._resolve_fp8_storage(source_was_quantized=True) is False
    monkeypatch.delenv("KREA2_FP8_STORAGE")
    # auto: needs a quantized source (fp8 or int8-ConvRot; CUDA availability
    # decides the rest at runtime)
    assert be._resolve_fp8_storage(source_was_quantized=False) is False


def test_registered_as_a_backend():
    from imference_engine import Engine, RuntimeConfig
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert "krea2" in engine._backends
    assert isinstance(engine._backends["krea2"], Krea2Backend)


def test_krea2_catalog_row_carries_checkpoint_defaults(tmp_path):
    """A Raw (undistilled) checkpoint expresses its recipe as MODEL defaults;
    the engine layer stays Turbo-oriented (8 steps, guidance 0.0)."""
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text(
        "models:\n"
        "  - name: krea2-raw\n"
        "    engine: krea2\n"
        "    weights: /cache/krea2_raw_bf16.safetensors\n"
        "    base_model: krea/Krea-2-Raw\n"
        "    defaults: {num_steps: 28, guidance_scale: 4.5}\n",
        encoding="utf-8",
    )
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    meta = engine._models.config_for("krea2-raw")
    assert meta.backend == "krea2"
    assert meta.base_model == "krea/Krea-2-Raw"
    assert meta.defaults.num_steps == 28
    assert meta.defaults.guidance_scale == 4.5
    assert isinstance(meta.defaults, GenerationDefaults)
