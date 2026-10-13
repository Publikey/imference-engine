"""Frame interpolation: RIFE v4.x (ONNX) between every pair of frames.

Frames are decoded with ffmpeg to raw RGB, padded to a multiple of 32, and for
each consecutive pair ``factor - 1`` intermediate frames are synthesized at
t = k/factor. The result is re-encoded at ``fps * factor``.

Graph contract: three inputs (img0, img1 ``[1, 3, H, W]`` in [0, 1]; timestep
as ``[1, 1, H, W]`` or ``[1]``), output ``[1, 3, H, W]``. Exports with a single
``[1, 6 or 7, H, W]`` input (frames stacked, + timestep plane) are handled too.
"""
from __future__ import annotations

import subprocess

from imference_engine.tools import media
from imference_engine.tools.errors import INVALID_INPUT, TOO_LARGE, ToolError
from imference_engine.tools.impl import onnx_util
from imference_engine.tools.progress import LOADING, RUNNING
from imference_engine.tools.runner import ToolOutput

MAX_FRAMES_OUT = 4000


def _feeds(sess, a, b, t):
    import numpy as np

    ins = sess.get_inputs()
    if len(ins) >= 3:
        ts_shape = ins[2].shape
        if len(ts_shape) == 4:
            ts = np.full((1, 1, a.shape[2], a.shape[3]), t, dtype=np.float32)
        else:
            ts = np.full([d if isinstance(d, int) and d > 0 else 1 for d in ts_shape] or [1], t, dtype=np.float32)
        return {ins[0].name: a, ins[1].name: b, ins[2].name: ts}
    ch = ins[0].shape[1]
    planes = [a, b]
    if ch == 7:
        planes.append(np.full((1, 1, a.shape[2], a.shape[3]), t, dtype=np.float32))
    return {ins[0].name: np.concatenate(planes, axis=1)}


def interpolate_pair(sess, a, b, t):
    return sess.run(None, _feeds(sess, a, b, t))[0]


def run(ctx, inputs, params):
    import numpy as np

    src = inputs["video"]
    if not isinstance(src, str):
        src = str(src)
    info = media.check_video(src, ctx.manifest.inputs["video"])
    W, H, fps = info["width"], info["height"], info["fps"] or 24.0
    factor = params["factor"]
    n_est = int(info["duration"] * fps) or 0
    if n_est * factor > MAX_FRAMES_OUT:
        raise ToolError(TOO_LARGE, f"~{n_est * factor} output frames (max {MAX_FRAMES_OUT}); trim the clip first")
    ctx.progress(LOADING, 0.0)
    sess, ctx.runtime = onnx_util.session(ctx.weights["model"], ctx.device)
    ff = media.ffmpeg_bin()
    dec = subprocess.Popen([ff, "-v", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE)
    out_path = ctx.work_dir / "interp.mp4"
    enc = subprocess.Popen([ff, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", f"{fps * factor:.6f}", "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
                            "-movflags", "+faststart", str(out_path)], stdin=subprocess.PIPE)
    ph, pw = (H + 31) // 32 * 32, (W + 31) // 32 * 32
    frame_bytes = W * H * 3

    def read():
        buf = dec.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            return None
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0
        p = np.zeros((ph, pw, 3), np.float32)
        p[:H, :W] = f
        return p.transpose(2, 0, 1)[None], buf

    def emit(raw: bytes):
        enc.stdin.write(raw)

    try:
        prev = read()
        if prev is None:
            raise ToolError(INVALID_INPUT, "the video has no frames")
        emit(prev[1])
        n = 1
        while True:
            ctx.check_deadline()
            cur = read()
            if cur is None:
                break
            for k in range(1, factor):
                mid = interpolate_pair(sess, prev[0], cur[0], k / factor)[0]
                mid = np.clip(mid[:, :H, :W].transpose(1, 2, 0), 0, 1)
                emit((mid * 255 + 0.5).astype(np.uint8).tobytes())
            emit(cur[1])
            prev = cur
            n += 1
            if n_est:
                ctx.progress(RUNNING, min(0.99, n / n_est))
    finally:
        try:
            enc.stdin.close()
        except OSError:
            pass
        enc.wait()
        dec.stdout.close()
        dec.wait()
    if enc.returncode != 0:
        raise ToolError("internal", "ffmpeg failed to encode the interpolated video")
    ctx.progress(RUNNING, 1.0)
    return [ToolOutput(name="video", type="video", format="mp4", path=out_path)]
