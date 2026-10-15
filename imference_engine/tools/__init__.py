"""Editing tools: a catalog of small, single-purpose image/video tools
(upscale, remove background, eraser, utilities, ...) that run the same code
locally (Desktop sidecar) and in the cloud (runqy ``tools-*`` workers).

    from imference_engine.tools import catalog, run_tool, ToolError

    res = run_tool("upscale", {"image": "in.png"}, {"scale": 2}, output_dir="out")
    print(res.outputs[0].path)

See ``docs/tools.md`` for the manifest format and the wire contracts.
"""
from imference_engine.tools import catalog
from imference_engine.tools.errors import ToolError
from imference_engine.tools.runner import ToolOutput, ToolResult, run_tool
from imference_engine.tools.schema import ToolManifest

__all__ = ["catalog", "run_tool", "ToolError", "ToolResult", "ToolOutput", "ToolManifest", "release", "loaded"]


def release() -> int:
    """Free every model a tool keeps warm (ONNX sessions, Florence-2)."""
    from imference_engine.tools.impl import onnx_util
    n = onnx_util.release()
    try:
        from imference_engine.tools.impl import describe
        describe.release()
    except Exception:
        pass
    return n


def loaded() -> list:
    """What the tools keep warm (ONNX sessions, Florence-2), for a host's
    status view: [{"name", "bytes", "runtime"}]."""
    from imference_engine.tools.impl import onnx_util
    out = onnx_util.loaded_detail()
    try:
        from imference_engine.tools.impl import describe
        out += describe.loaded_detail()
    except Exception:  # noqa: BLE001 — no transformers: nothing warm there
        pass
    return out
