"""Tool implementations, keyed by tool id. Imported lazily: a Pillow utility
must not pay for onnxruntime, and the cloud CPU image has no torch at all."""
from __future__ import annotations

import importlib
from typing import Callable

from imference_engine.tools.errors import UNSUPPORTED_RUNTIME, ToolError

# tool id -> "module:function"
_REGISTRY = {
    "resize": "image_utils:resize",
    "crop": "image_utils:crop",
    "convert": "image_utils:convert",
    "strip-metadata": "image_utils:strip_metadata",
    "video-trim": "video_utils:trim",
    "video-concat": "video_utils:concat",
    "video-convert": "video_utils:convert",
    "upscale": "upscale:run",
    "remove-background": "background:run",
    "erase-object": "erase:run",
    "segment": "segment:run",
    "describe": "describe:run",
    "interpolate": "interpolate:run",
    "restore-faces": "faces:run",
    "vary": "generation:vary",
    "inpaint": "generation:inpaint",
    "outpaint": "generation:outpaint",
    "instruct-edit": "generation:instruct_edit",
}


def get_impl(tool_id: str) -> Callable:
    ref = _REGISTRY.get(tool_id)
    if ref is None:
        raise ToolError(UNSUPPORTED_RUNTIME, f"tool {tool_id!r} has no implementation in this engine version")
    mod, _, fn = ref.partition(":")
    try:
        module = importlib.import_module(f"imference_engine.tools.impl.{mod}")
    except ImportError as e:
        raise ToolError(UNSUPPORTED_RUNTIME, f"tool {tool_id!r} needs a missing dependency: {e.name or e}. "
                        "Install imference-engine[tools]") from e
    return getattr(module, fn)


def implemented() -> set:
    return set(_REGISTRY)
