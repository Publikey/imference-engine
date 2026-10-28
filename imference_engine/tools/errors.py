"""Typed tool errors — the one failure vocabulary shared by every surface.

Desktop, the cloud worker, the API and the MCP server all key off ``code``:
Desktop offers a cloud retry on ``oom`` / ``unsupported_runtime``, the API maps
``invalid_input`` / ``too_large`` to a 4xx and refunds credits on the rest.
"""
from __future__ import annotations

from typing import Optional

INVALID_INPUT = "invalid_input"
TOO_LARGE = "too_large"
UNSUPPORTED_RUNTIME = "unsupported_runtime"
WEIGHTS_MISSING = "weights_missing"
OOM = "oom"
TIMEOUT = "timeout"
INTERNAL = "internal"

CODES = (INVALID_INPUT, TOO_LARGE, UNSUPPORTED_RUNTIME, WEIGHTS_MISSING, OOM, TIMEOUT, INTERNAL)

# Codes where re-running the same request in the cloud can succeed.
CLOUD_RETRYABLE = (OOM, UNSUPPORTED_RUNTIME, WEIGHTS_MISSING)

# Codes caused by the request itself: no point retrying anywhere, no charge.
CLIENT_ERRORS = (INVALID_INPUT, TOO_LARGE)


class ToolError(Exception):
    """A tool failure with a stable ``code`` (one of ``CODES``)."""

    def __init__(self, code: str, message: str, *, field: Optional[str] = None) -> None:
        if code not in CODES:
            raise ValueError(f"unknown tool error code {code!r}")
        super().__init__(message)
        self.code = code
        self.field = field

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": str(self)}
        if self.field:
            d["field"] = self.field
        return d

    @property
    def cloud_retryable(self) -> bool:
        return self.code in CLOUD_RETRYABLE


def classify(exc: BaseException) -> ToolError:
    """Map an arbitrary exception from a runtime to a ``ToolError``.

    Out-of-memory comes in many spellings (torch CUDA OOM, onnxruntime's
    BFCArena / "Failed to allocate memory", plain MemoryError); all of them
    become ``oom`` so Desktop can offer the cloud instead of a stack trace.
    """
    if isinstance(exc, ToolError):
        return exc
    if isinstance(exc, MemoryError):
        return ToolError(OOM, "out of memory")
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    # cuBLAS / cuDNN handles created on a full GPU fail with *_STATUS_ALLOC_FAILED
    # (onnxruntime: "Exception during initialization ... CUBLAS_STATUS_ALLOC_FAILED").
    if ("out of memory" in low or "outofmemory" in low or "failed to allocate" in low
            or "bfcarena" in low or "_status_alloc_failed" in low or "cudaerrormemoryallocation" in low):
        return ToolError(OOM, text[:500])
    if isinstance(exc, TimeoutError):
        return ToolError(TIMEOUT, text[:500])
    return ToolError(INTERNAL, text[:500])
