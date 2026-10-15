"""Describe image -> prompt: Florence-2-base through transformers (native
``Florence2ForConditionalGeneration``, transformers >= 4.56).

The weight is a tarball of the ``florence-community/Florence-2-base`` repo
(mirrored like every tool weight, so no HuggingFace call at runtime); it is
extracted once next to the tarball.
"""
from __future__ import annotations

import logging
import tarfile
import threading
from pathlib import Path

from imference_engine.tools import media
from imference_engine.tools.errors import UNSUPPORTED_RUNTIME, ToolError
from imference_engine.tools.progress import LOADING, RUNNING
from imference_engine.tools.runner import ToolOutput

logger = logging.getLogger(__name__)

TASKS = {"short": "<CAPTION>", "detailed": "<DETAILED_CAPTION>", "more_detailed": "<MORE_DETAILED_CAPTION>"}
_LOCK = threading.Lock()
_MODEL: dict = {}


def extract(tar_path: Path) -> Path:
    dest = tar_path.with_suffix("")
    marker = dest / ".extracted"
    if marker.is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:  # Python < 3.12 without the filter argument
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    raise ToolError("weights_missing", f"unsafe path in {tar_path.name}: {m.name}")
            tf.extractall(dest)
    # Accept a tarball with a single top-level directory.
    entries = [p for p in dest.iterdir() if p.name != ".extracted"]
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else dest
    marker.write_text(str(root))
    return dest


def _model_dir(tar_path: Path) -> Path:
    d = extract(tar_path)
    root = Path((d / ".extracted").read_text().strip() or d)
    return root


def _device(pref: str):
    import torch

    from imference_engine.runtime.device import resolve_device
    dev = resolve_device(pref if pref in ("auto", "cpu", "cuda", "mps") else "auto")
    dtype = torch.float16 if dev.kind == "cuda" else torch.float32
    return dev.torch_str, dtype


def load(tar_path: Path, device: str):
    try:
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
    except ImportError as e:
        raise ToolError(UNSUPPORTED_RUNTIME, "describe needs torch + transformers>=4.56 "
                        "(install imference-engine[tools-torch])") from e
    dev, dtype = _device(device)
    key = (str(tar_path), dev)
    with _LOCK:
        if key not in _MODEL:
            root = _model_dir(tar_path)
            proc = AutoProcessor.from_pretrained(root, local_files_only=True)
            model = Florence2ForConditionalGeneration.from_pretrained(root, dtype=dtype, local_files_only=True).to(dev).eval()
            _MODEL.clear()
            _MODEL[key] = (proc, model, dev, dtype)
        return _MODEL[key]


def loaded_detail() -> list:
    """Florence-2 when warm: [{"name", "bytes" (parameters), "runtime"}]."""
    with _LOCK:
        items = list(_MODEL.values())
    out = []
    for _proc, model, dev, _dtype in items:
        try:
            size = sum(p.numel() * p.element_size() for p in model.parameters())
        except Exception:  # noqa: BLE001
            size = 0
        out.append({"name": "Florence-2", "bytes": int(size), "runtime": f"torch:{dev}"})
    return out


def release() -> None:
    with _LOCK:
        _MODEL.clear()


def run(ctx, inputs, params):
    import torch

    img = media.load_image(inputs["image"], ctx.manifest.inputs["image"], mode="RGB")
    ctx.progress(LOADING, 0.0)
    proc, model, dev, dtype = load(Path(ctx.weights["model"]), ctx.device)
    ctx.runtime = f"torch:{dev}"
    task = TASKS[params["detail"]]
    ctx.progress(RUNNING, 0.1)
    enc = proc(text=task, images=img, return_tensors="pt")
    enc = {k: (v.to(dev, dtype) if v.is_floating_point() else v.to(dev)) for k, v in enc.items()}
    with torch.inference_mode():
        ids = model.generate(**enc, max_new_tokens=256, num_beams=3, do_sample=False)
    raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(raw, task=task, image_size=img.size)
    text = parsed.get(task, raw) if isinstance(parsed, dict) else str(parsed)
    text = " ".join(str(text).replace("<pad>", "").replace("</s>", "").replace("<s>", "").split())
    ctx.progress(RUNNING, 1.0)
    return [ToolOutput(name="text", type="text", format="txt", text=text)]
