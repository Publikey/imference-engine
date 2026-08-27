"""Prompt text hygiene shared by the image and video engines.

One job: strip LONE UTF-16 SURROGATES from user-supplied text. They are the
debris of a copy-paste that split an emoji/astral codepoint in half; JavaScript
strings and JSON escapes carry them happily, Python reconstructs them into the
str — and the Rust fast tokenizer then fails to convert the string at the PyO3
boundary with the maximally confusing

    TypeError: TextEncodeInput must be Union[TextInputSequence,
                                             Tuple[InputSequence, InputSequence]]

(observed in production with a civitai-style pasted prompt). Every engine
tokenizes prompts with a fast tokenizer, so the fix lives here, once, at the
engine boundary — callers (desktop sidecar, worker payloads) don't need to
know.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def sanitize_prompt_text(text: Optional[str], *, field: str = "prompt") -> Optional[str]:
    """Return ``text`` with any lone surrogates removed. ``None`` passes
    through. Valid text is returned unchanged (fast path: one encode probe).
    Logs a warning when characters had to be dropped — the caller's text was
    already corrupt before it reached us, but silently altering input deserves
    a trace."""
    if text is None:
        return None
    try:
        text.encode("utf-8")
        return text  # valid Unicode — the overwhelmingly common case
    except UnicodeEncodeError:
        cleaned = text.encode("utf-8", errors="ignore").decode("utf-8")
        logger.warning(
            "%s contained %d invalid Unicode character(s) (lone surrogates from "
            "a broken copy-paste) — dropped so the tokenizer can run",
            field, len(text) - len(cleaned),
        )
        return cleaned
