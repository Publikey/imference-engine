"""``run_tool`` — the one pure entry point every surface calls.

    from imference_engine.tools import run_tool
    res = run_tool("remove-background", {"image": "in.png"}, {"output": "png-alpha"},
                   output_dir="out/", on_progress=lambda stage, f: ...)
    res.outputs[0].path   # out/remove-background-<id>.png

The same call runs in the Desktop sidecar (local paths) and in the cloud
worker (files downloaded from signed URLs into a temp dir). It validates params
against the manifest, fetches weights, runs the implementation, writes the
outputs and returns a ``ToolResult``; every failure is a ``ToolError`` with a
stable code.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from imference_engine.tools import catalog as _catalog
from imference_engine.tools import weights as _weights
from imference_engine.tools.errors import OOM, UNSUPPORTED_RUNTIME, ToolError, classify
from imference_engine.tools.progress import ENCODING, Progress, ProgressFn

logger = logging.getLogger(__name__)


@dataclass
class ToolOutput:
    name: str
    type: str                       # image | mask | video | text | json
    format: str = ""                # png / jpeg / webp / mp4 / gif / txt / json
    path: Optional[Path] = None     # set once written (always for video)
    image: Any = None               # PIL image (image / mask outputs, before writing)
    text: Optional[str] = None      # text outputs
    data: Any = None                # json outputs
    width: Optional[int] = None
    height: Optional[int] = None
    save_options: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type, "format": self.format}
        if self.path is not None:
            d["path"] = str(self.path)
        if self.text is not None:
            d["text"] = self.text
        if self.data is not None:
            d["data"] = self.data
        if self.width:
            d["width"], d["height"] = self.width, self.height
        return d


def image_output(image, fmt: str, name: str = "image") -> ToolOutput:
    """An AI image tool's result in its ``format`` param: lossy WebP (the
    default, see media.WEBP_OUTPUT) or lossless PNG."""
    from imference_engine.tools import media
    return ToolOutput(name=name, type="image", format=fmt, image=image,
                      save_options=dict(media.WEBP_OUTPUT) if fmt == "webp" else {})


@dataclass
class ToolResult:
    tool_id: str
    tool_version: str
    outputs: list
    params: dict
    duration_ms: int
    runtime: str = ""               # e.g. "onnx:CUDAExecutionProvider", "pillow", "ffmpeg"
    stages: list = field(default_factory=list)

    def output(self, name: Optional[str] = None) -> ToolOutput:
        if name is None:
            return self.outputs[0]
        for o in self.outputs:
            if o.name == name:
                return o
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {"tool_id": self.tool_id, "tool_version": self.tool_version,
                "outputs": [o.to_dict() for o in self.outputs], "params": self.params,
                "duration_ms": self.duration_ms, "runtime": self.runtime, "stages": self.stages}


@dataclass
class RunContext:
    """What an implementation gets besides inputs/params."""

    manifest: Any
    weights: dict                   # name -> Path
    progress: Progress
    work_dir: Path
    device: str = "auto"            # auto | cpu | cuda | mps | directml | rocm
    engine: Any = None              # imference_engine.Engine, for needs_engine tools
    timeout: Optional[float] = None
    runtime: str = ""               # implementations fill it for logs
    deadline: Optional[float] = None

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            from imference_engine.tools.errors import TIMEOUT
            raise ToolError(TIMEOUT, f"tool {self.manifest.id} exceeded {self.timeout:.0f}s")


def run_tool(
    tool_id: str,
    inputs: dict,
    params: Optional[dict] = None,
    *,
    output_dir: Optional[str] = None,
    on_progress: Optional[ProgressFn] = None,
    device: str = "auto",
    engine: Any = None,
    timeout: Optional[float] = None,
    cache_dir: Optional[str] = None,
    cdn: Optional[str] = None,
    catalog: Optional[_catalog.ToolCatalog] = None,
    output_basename: Optional[str] = None,
) -> ToolResult:
    """Validate, fetch weights, run ``tool_id`` and write its outputs.

    ``inputs`` maps input names to a local path / bytes / PIL image (a list for
    ``multiple`` inputs). ``output_dir`` set -> outputs are written there and
    ``ToolOutput.path`` is filled; unset -> image outputs stay in memory
    (``.image``) and video outputs go to a temp dir.
    """
    import tempfile

    t0 = time.monotonic()
    cat = catalog or _catalog.builtin()
    manifest = cat.get(tool_id)
    manifest.validate_input_names(inputs)
    clean = manifest.validate_params(params)

    if manifest.runtimes.needs_engine and engine is None:
        raise ToolError(UNSUPPORTED_RUNTIME, f"tool {tool_id!r} runs a generation model: "
                        "load a model (Engine) first or run it in the cloud")

    from imference_engine.tools import media
    from imference_engine.tools.impl import get_impl
    impl = get_impl(manifest.id)
    media.precheck(manifest, inputs)   # too_large before any weight download

    progress = Progress(on_progress)
    work = Path(tempfile.mkdtemp(prefix=f"imtool-{tool_id}-"))
    try:
        w = _weights.ensure(manifest.weights_for(clean), cache_dir=cache_dir, cdn=cdn, on_progress=progress)
        ctx = RunContext(manifest=manifest, weights=w, progress=progress, work_dir=work, device=device,
                         engine=engine, timeout=timeout,
                         deadline=(time.monotonic() + timeout) if timeout else None)
        try:
            outputs = _run_impl(impl, ctx, inputs, clean, tool_id)
        except ToolError as err:
            # Out of GPU memory on a light tool: the warm sessions of the other
            # tools hold it (a cloud worker keeps them all). Free them and
            # retry once; generation tools share the engine's model, untouched.
            if err.code != OOM or manifest.runtimes.needs_engine or str(device).startswith("cpu"):
                raise
            freed = _free_tool_memory(err)
            logger.warning("tool %s ran out of memory: released %d warm session(s), retrying once", tool_id, freed)
            try:
                outputs = _run_impl(impl, ctx, inputs, clean, tool_id)
            except ToolError as again:
                # Too big even alone (e.g. upscale tile=0 on 4 MP): the failed
                # attempt's session keeps what it reserved, which would starve
                # every next request on this worker. Give it back first.
                if again.code == OOM:
                    _free_tool_memory(again)
                raise

        progress(ENCODING, 0.0)
        out_dir = Path(output_dir) if output_dir else None
        base = output_basename or f"{tool_id}-{uuid.uuid4().hex[:8]}"
        for i, o in enumerate(outputs):
            suffix = "" if len(outputs) == 1 else f"-{o.name}"
            _finalize(o, out_dir, f"{base}{suffix}", work)
            progress(ENCODING, (i + 1) / len(outputs))
        return ToolResult(tool_id=manifest.id, tool_version=manifest.version, outputs=outputs, params=clean,
                          duration_ms=int((time.monotonic() - t0) * 1000), runtime=ctx.runtime,
                          stages=list(progress.timeline))
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def _run_impl(impl, ctx, inputs, clean, tool_id):
    """The implementation, any runtime failure typed (oom, internal, ...)."""
    try:
        return impl(ctx, inputs, clean)
    except ToolError:
        raise
    except Exception as e:
        err = classify(e)
        logger.exception("tool %s failed: %s", tool_id, err)
        raise err from e


def _free_tool_memory(err: Optional[BaseException] = None) -> int:
    """Drop the warm tool sessions (ONNX, Florence-2) and torch's cache.

    ``err`` is the OOM being handled: its traceback chain (already logged by
    ``_run_impl``) keeps the failed attempt's frames — and the session in
    their locals — alive, so release() alone would not give that memory back.
    Clearing the frames' locals works whoever else still holds the traceback.
    """
    import traceback
    seen = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        if err.__traceback__ is not None:
            traceback.clear_frames(err.__traceback__)   # skips the frames still running
        err = err.__cause__ or err.__context__
    from imference_engine import tools
    n = tools.release()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — no torch, or no CUDA
        pass
    return n


def _finalize(o: ToolOutput, out_dir: Optional[Path], stem: str, work: Path) -> None:
    from imference_engine.tools import media

    if o.image is not None:
        o.width, o.height = o.image.size
        if not o.format:
            o.format = "png"
        if o.format == "webp" and max(o.image.size) > media.WEBP_MAX_SIDE:
            o.format = "png"
        if out_dir is not None:
            o.path = media.save_image(o.image, out_dir / (stem + media.ext_for(o.format)), o.format, **o.save_options)
    elif o.type == "video":
        # Implementations write video into the work dir (deleted after the
        # call): move it to output_dir, or to a stable temp dir.
        import shutil
        import tempfile
        dest_dir = out_dir or Path(tempfile.mkdtemp(prefix="imtool-out-"))
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / (stem + media.ext_for(o.format))
        shutil.move(str(o.path), dest)
        o.path = dest
    elif o.type in ("text", "json") and out_dir is not None:
        import json
        out_dir.mkdir(parents=True, exist_ok=True)
        if o.type == "text":
            o.format = o.format or "txt"
            o.path = out_dir / f"{stem}.txt"
            o.path.write_text(o.text or "", encoding="utf-8")
        else:
            o.format = "json"
            o.path = out_dir / f"{stem}.json"
            o.path.write_text(json.dumps(o.data), encoding="utf-8")
