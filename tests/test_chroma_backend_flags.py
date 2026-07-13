"""No-torch assertions on the Chroma backend.

Guards the offline base-component contract, the single-encoder shape (no CLIP),
and the KEY FLUX contrast: Chroma is de-distilled → real CFG → negative prompt is
honored (FLUX ignores it).
"""
from __future__ import annotations

from imference_engine.chroma import ChromaBackend


def test_engine_key():
    assert ChromaBackend.engine == "chroma"


def test_engine_defaults_real_cfg():
    d = ChromaBackend().engine_defaults()
    assert d.guidance_scale == 2.0   # validated: higher oversaturates
    assert d.num_steps == 28
    assert d.scheduler is None


def test_base_patterns_single_t5_encoder_no_clip():
    pats = ChromaBackend.BASE_PATTERNS
    assert "text_encoder/*" in pats and "vae/*" in pats
    assert "transformer/config.json" in pats
    # Single encoder — no CLIP second encoder/tokenizer (that's FLUX).
    assert "text_encoder_2/*" not in pats
    assert "tokenizer_2/*" not in pats


def test_base_patterns_never_pull_transformer_weights():
    pats = ChromaBackend.BASE_PATTERNS
    assert "transformer/*" not in pats
    assert not any(p.startswith("transformer/") and "safetensors" in p for p in pats)


def test_encode_prompts_keeps_negative():
    """Real CFG → negative prompt IS passed through (unlike FLUX)."""
    be = ChromaBackend()
    out = be.encode_prompts(pipe=None, prompt="a cat", negative_prompt="blurry")
    assert out == {"prompt": "a cat", "negative_prompt": "blurry"}
    # empty/None negative normalizes to ""
    assert be.encode_prompts(pipe=None, prompt="a cat", negative_prompt=None) == {
        "prompt": "a cat", "negative_prompt": ""}


def test_registered_as_a_backend():
    from imference_engine import Engine, RuntimeConfig
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends.get("chroma"), ChromaBackend)
