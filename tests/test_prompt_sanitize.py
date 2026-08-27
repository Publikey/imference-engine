"""Lone-surrogate prompt hygiene (core/text.py).

Broken copy-pastes leave lone UTF-16 surrogates in prompt strings; the Rust
fast tokenizer then fails at the PyO3 boundary with the opaque
"TextEncodeInput must be Union[...]" TypeError (hit in production via the
desktop sidecar). The engines sanitize at their generate() boundary.
"""
from __future__ import annotations

import logging

from imference_engine.core.text import sanitize_prompt_text


def test_valid_text_is_returned_unchanged():
    s = "a fox in the snow — 一位年轻女性, ünïcödé, 🦊"
    assert sanitize_prompt_text(s) is s  # fast path: same object, no copy


def test_none_passes_through():
    assert sanitize_prompt_text(None) is None


def test_lone_surrogates_are_dropped(caplog):
    broken = "hello \ud83d world \udc4d!"  # lone high + lone low surrogate
    with caplog.at_level(logging.WARNING):
        out = sanitize_prompt_text(broken, field="prompt")
    assert out == "hello  world !"
    out.encode("utf-8")  # must be tokenizer-safe now
    assert any("invalid Unicode" in r.message for r in caplog.records)


def test_valid_astral_pairs_survive():
    """A REAL emoji (a valid surrogate pair in the source literal) is fine and
    must not be touched — only lone halves are debris."""
    s = "night city 🌃 rain"
    assert sanitize_prompt_text(s) == s


def test_engine_generate_sanitizes_before_backends():
    """The image Engine strips the debris before any backend sees the prompt —
    guarded here at the API boundary via a registered-but-unloadable model:
    the error we get must be the load failure, never the tokenizer TypeError,
    and the sanitize must not raise on the way in."""
    from imference_engine import Engine, RuntimeConfig

    eng = Engine(runtime=RuntimeConfig(device="cpu")).load()
    eng.register_model("m", backend="sdxl", weights_path="/nonexistent.safetensors")
    try:
        eng.generate(model="m", prompt="broken \ud83d prompt", batch=1)
    except Exception as e:
        assert "TextEncodeInput" not in str(e)
