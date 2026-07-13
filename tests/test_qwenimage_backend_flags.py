"""No-torch assertions on the Qwen-Image backend.

Guards the offline base-component contract, the single-encoder shape, and the
true-CFG mapping (guidance_scale -> true_cfg_scale, negative default " ").
"""
from __future__ import annotations

from imference_engine.qwenimage import QwenImageBackend


def test_engine_key():
    assert QwenImageBackend.engine == "qwenimage"


def test_engine_defaults_true_cfg():
    d = QwenImageBackend().engine_defaults()
    assert d.guidance_scale == 4.0
    assert d.scheduler is None
    # num_steps intentionally left to GLOBAL_DEFAULTS (base wants ~50; set per model)
    assert d.num_steps is None


def test_base_patterns_single_encoder_no_clip():
    pats = QwenImageBackend.BASE_PATTERNS
    assert "text_encoder/*" in pats and "vae/*" in pats
    assert "transformer/config.json" in pats
    assert "text_encoder_2/*" not in pats and "tokenizer_2/*" not in pats


def test_base_patterns_never_pull_transformer_weights():
    pats = QwenImageBackend.BASE_PATTERNS
    assert "transformer/*" not in pats
    assert not any(p.startswith("transformer/") and "safetensors" in p for p in pats)


def test_encode_prompts_negative_defaults_to_space():
    """Upstream uses a single space as the default negative (real CFG)."""
    be = QwenImageBackend()
    assert be.encode_prompts(pipe=None, prompt="a cat", negative_prompt=None) == {
        "prompt": "a cat", "negative_prompt": " "}
    assert be.encode_prompts(pipe=None, prompt="a cat", negative_prompt="blurry") == {
        "prompt": "a cat", "negative_prompt": "blurry"}


def test_build_inference_kwargs_maps_guidance_to_true_cfg():
    be = QwenImageBackend()
    kw = be.build_inference_kwargs(
        width=1328, height=1328, num_steps=50, guidance_scale=4.0,
        clip_skip=None, chunk_size=2, generator=None)
    assert kw["true_cfg_scale"] == 4.0        # mapped, not passed as guidance_scale
    assert "guidance_scale" not in kw          # distilled arg left at default
    assert kw["num_images_per_prompt"] == 2
    assert kw["width"] == 1328


def test_registered_as_a_backend():
    from imference_engine import Engine, RuntimeConfig
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends.get("qwenimage"), QwenImageBackend)
