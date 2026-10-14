"""ONNX Runtime sessions for the light models.

One cached ``InferenceSession`` per (weight file, device). Sessions stay warm
across calls during an editing session and are dropped by ``release()`` —
Desktop calls it when the editor closes so a 200 MB eraser doesn't squat RAM
next to an 8 GB generation model.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from imference_engine.tools.errors import UNSUPPORTED_RUNTIME, ToolError

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_SESSIONS: dict = {}

# Preference order for device="auto". CoreML is opt-in (device="mps"): its
# partitioner rejects the dynamic shapes of several of these graphs.
_AUTO_ORDER = ("CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider",
               "DmlExecutionProvider")
_BY_DEVICE = {
    "cuda": ("CUDAExecutionProvider",),
    "rocm": ("ROCMExecutionProvider", "MIGraphXExecutionProvider"),
    "directml": ("DmlExecutionProvider",),
    "mps": ("CoreMLExecutionProvider",),
    "cpu": (),
}


def _ort():
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ToolError(UNSUPPORTED_RUNTIME, "onnxruntime is not installed (pip install imference-engine[tools])") from e
    return ort


def providers_for(device: str = "auto") -> list:
    ort = _ort()
    available = set(ort.get_available_providers())
    dev = (device or "auto").split(":")[0].lower()
    if dev == "auto":
        wanted = [p for p in _AUTO_ORDER if p in available]
    else:
        if dev not in _BY_DEVICE:
            raise ToolError(UNSUPPORTED_RUNTIME, f"unknown device {device!r}")
        wanted = [p for p in _BY_DEVICE[dev] if p in available]
        if _BY_DEVICE[dev] and not wanted:
            raise ToolError(UNSUPPORTED_RUNTIME, f"device {device!r} is not available to onnxruntime "
                            f"(have: {sorted(available)})")
    return wanted + ["CPUExecutionProvider"]


_GPU_PROVIDERS = ("CUDAExecutionProvider", "TensorrtExecutionProvider")
_DLLS_PREPARED = False


def _prepare_cuda_dlls(ort) -> None:
    """Make cuBLAS / cuDNN loadable for the CUDA provider. onnxruntime-gpu does
    not ship them: importing a CUDA torch first loads its copies (ORT then
    reuses them), and ``preload_dlls`` (ORT >= 1.21) picks up the nvidia-* pip
    wheels otherwise. Without this the CUDA provider silently falls back to CPU."""
    global _DLLS_PREPARED
    if _DLLS_PREPARED:
        return
    _DLLS_PREPARED = True
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 — no torch: rely on preload_dlls / PATH
        pass
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            preload()
        except Exception as e:  # noqa: BLE001
            logger.debug("onnxruntime.preload_dlls failed: %s", e)


def session(path, device: str = "auto"):
    """Cached InferenceSession for ``path`` on ``device``. Returns (session, label)."""
    ort = _ort()
    path = str(Path(path))
    provs = providers_for(device)
    if any(p in _GPU_PROVIDERS for p in provs):
        _prepare_cuda_dlls(ort)
    key = (path, tuple(provs))
    with _LOCK:
        hit = _SESSIONS.get(key)
        if hit is not None:
            return hit
        opts = ort.SessionOptions()
        threads = os.environ.get("ORT_INTRA_OP_THREADS")
        if threads and threads.isdigit():
            opts.intra_op_num_threads = int(threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess = ort.InferenceSession(path, sess_options=opts, providers=provs)
        except Exception as e:
            msg = str(e)
            if "INVALID_PROTOBUF" in msg or "Protobuf parsing failed" in msg or "No graph" in msg:
                raise ToolError("weights_missing", f"corrupt weight file {os.path.basename(path)}: delete it to re-download") from e
            raise
        if provs[0] != "CPUExecutionProvider" and sess.get_providers()[0] == "CPUExecutionProvider":
            logger.warning("%s: %s could not be created, running on CPU (onnxruntime %s; the GPU build "
                           "must match the installed CUDA/cuDNN)", os.path.basename(path), provs[0],
                           getattr(ort, "__version__", "?"))
        label = "onnx:" + sess.get_providers()[0]
        _SESSIONS[key] = (sess, label)
        logger.info("Loaded %s on %s", os.path.basename(path), sess.get_providers())
        return sess, label


def release() -> int:
    """Drop every cached session. Returns how many were released."""
    with _LOCK:
        n = len(_SESSIONS)
        _SESSIONS.clear()
    import gc
    gc.collect()
    return n


def loaded() -> list:
    with _LOCK:
        return [os.path.basename(k[0]) for k in _SESSIONS]


def input_hw(sess, default: int):
    """Static (H, W) of the first input, or (default, default) when dynamic."""
    shape = sess.get_inputs()[0].shape
    try:
        h, w = int(shape[2]), int(shape[3])
        return h, w
    except (TypeError, ValueError, IndexError):
        return default, default
