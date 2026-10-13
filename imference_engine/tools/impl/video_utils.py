"""ffmpeg utilities: trim, concat, convert (MP4 / WebM / GIF)."""
from __future__ import annotations

from pathlib import Path

from imference_engine.tools import media
from imference_engine.tools.errors import INVALID_INPUT, ToolError
from imference_engine.tools.progress import RUNNING
from imference_engine.tools.runner import ToolOutput

# Widely playable H.264: yuv420p needs even dimensions.
_H264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart"]
_EVEN = "scale=trunc(iw/2)*2:trunc(ih/2)*2"


def _path(v) -> str:
    if not isinstance(v, (str, Path)):
        raise ToolError(INVALID_INPUT, "video inputs must be file paths")
    return str(v)


def trim(ctx, inputs, params):
    ctx.runtime = "ffmpeg"
    src = _path(inputs["video"])
    info = media.check_video(src, ctx.manifest.inputs["video"])
    start, end = params["start"], params.get("end")
    if end is None or end > info["duration"] > 0:
        end = info["duration"] or None
    if end is not None and end <= start:
        raise ToolError(INVALID_INPUT, f"end ({end}) must be after start ({start})", field="end")
    out = ctx.work_dir / "trim.mp4"
    args = ["-ss", f"{start:.3f}", "-i", src]
    if end is not None:
        args += ["-t", f"{end - start:.3f}"]
    if params["reencode"]:
        args += ["-vf", _EVEN, *_H264, "-c:a", "aac"]
    else:
        args += ["-c", "copy"]
    ctx.progress(RUNNING, 0.1)
    media.run_ffmpeg(args + [str(out)], timeout=ctx.timeout)
    ctx.progress(RUNNING, 1.0)
    return [ToolOutput(name="video", type="video", format="mp4", path=out)]


def concat(ctx, inputs, params):
    ctx.runtime = "ffmpeg"
    clips = [_path(v) for v in inputs["videos"]]
    spec = ctx.manifest.inputs["videos"]
    infos = [media.check_video(c, spec) for c in clips]
    w, h = infos[0]["width"] // 2 * 2, infos[0]["height"] // 2 * 2
    fps = params.get("fps") or round(infos[0]["fps"] or 24)
    # Normalize every clip (size, fps, sar) then the concat filter; audio is
    # dropped (generated clips are silent and mixed audio/no-audio breaks concat).
    chains, labels = [], []
    for i in range(len(clips)):
        chains.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                      f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[v{i}]")
        labels.append(f"[v{i}]")
    graph = ";".join(chains) + ";" + "".join(labels) + f"concat=n={len(clips)}:v=1:a=0[out]"
    args = []
    for c in clips:
        args += ["-i", c]
    out = ctx.work_dir / "concat.mp4"
    ctx.progress(RUNNING, 0.1)
    media.run_ffmpeg(args + ["-filter_complex", graph, "-map", "[out]", *_H264, "-an", str(out)], timeout=ctx.timeout)
    ctx.progress(RUNNING, 1.0)
    return [ToolOutput(name="video", type="video", format="mp4", path=out)]


def convert(ctx, inputs, params):
    ctx.runtime = "ffmpeg"
    src = _path(inputs["video"])
    info = media.check_video(src, ctx.manifest.inputs["video"])
    fmt = params["format"]
    fps = params.get("fps") or (15 if fmt == "gif" else None)
    width = params.get("width")
    filters = []
    if fps:
        filters.append(f"fps={fps}")
    if width:
        filters.append(f"scale={width}:-2:flags=lanczos")
    out = ctx.work_dir / f"convert{media.ext_for(fmt)}"
    ctx.progress(RUNNING, 0.1)
    if fmt == "gif":
        # Two-pass palette in one graph: far better colors than the default.
        chain = ",".join(filters) if filters else "null"
        vf = f"{chain},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        media.run_ffmpeg(["-i", src, "-filter_complex", vf, "-loop", "0", str(out)], timeout=ctx.timeout)
    elif fmt == "webm":
        vf = ",".join(filters + [_EVEN])
        media.run_ffmpeg(["-i", src, "-vf", vf, "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32",
                          "-row-mt", "1", "-c:a", "libopus", str(out)], timeout=ctx.timeout)
    else:
        vf = ",".join(filters + [_EVEN])
        media.run_ffmpeg(["-i", src, "-vf", vf, *_H264, "-c:a", "aac", str(out)], timeout=ctx.timeout)
    ctx.progress(RUNNING, 1.0)
    del info
    return [ToolOutput(name="video", type="video", format=fmt, path=out)]
