"""Real-ESRGAN upscaling (ONNX), tiled so memory stays flat on any size.

Graph contract: input ``[1, 3, H, W]`` float32 RGB in [0, 1], output
``[1, 3, sH, sW]`` in [0, 1] — the scale ``s`` is read from the first tile.
Tiles overlap by ``pad`` pixels and only their centre is kept, which removes
the seams of naive tiling. ``scale=2`` with an x4 model renders x4 then
Lanczos-downsamples (sharper than a native x2 model, same cost).

Whole-image (``tile=0``) or large tiles that run out of GPU memory fall back
to ``FALLBACK_TILE`` (same pixels, the seams are removed exactly): x4 of a
4 MP image whole needs more than 16 GB.
"""
from __future__ import annotations

import logging

from imference_engine.tools import media
from imference_engine.tools.errors import OOM, classify
from imference_engine.tools.impl import onnx_util
from imference_engine.tools.progress import LOADING, RUNNING
from imference_engine.tools.runner import image_output

logger = logging.getLogger(__name__)

PAD = 16
FALLBACK_TILE = 256


def _infer(sess, name, arr):
    return sess.run(None, {name: arr})[0]


def upscale_array(sess, rgb, tile: int, progress=None, check=None):
    """rgb: HxWx3 float32 [0,1] -> (sH x sW x 3 float32, s)."""
    import numpy as np

    h, w, _ = rgb.shape
    name = sess.get_inputs()[0].name
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    if tile <= 0 or (h <= tile and w <= tile):
        y = _infer(sess, name, x)[0]
        if progress:
            progress(RUNNING, 1.0)
        return np.clip(y.transpose(1, 2, 0), 0, 1), y.shape[1] // h
    scale = None
    out = None
    ys = list(range(0, h, tile))
    xs = list(range(0, w, tile))
    n, done = len(ys) * len(xs), 0
    for y0 in ys:
        for x0 in xs:
            if check:
                check()
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            py0, px0 = max(y0 - PAD, 0), max(x0 - PAD, 0)
            py1, px1 = min(y1 + PAD, h), min(x1 + PAD, w)
            t = _infer(sess, name, np.ascontiguousarray(x[:, :, py0:py1, px0:px1]))[0]
            if scale is None:
                scale = t.shape[1] // (py1 - py0)
                out = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
            s = scale
            oy, ox = (y0 - py0) * s, (x0 - px0) * s
            out[y0 * s:y1 * s, x0 * s:x1 * s] = t[:, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s].transpose(1, 2, 0)
            done += 1
            if progress:
                progress(RUNNING, done / n)
    return np.clip(out, 0, 1), scale


def run(ctx, inputs, params):
    import numpy as np
    from PIL import Image

    src = media.load_image(inputs["image"], ctx.manifest.inputs["image"])
    alpha = src.getchannel("A") if src.mode in ("RGBA", "LA") else None
    rgb = np.asarray(src.convert("RGB"), dtype=np.float32) / 255.0
    ctx.progress(LOADING, 0.0)
    sess, ctx.runtime = onnx_util.session(ctx.weights["model"], ctx.device)
    ctx.progress(RUNNING, 0.0)
    tile = params["tile"]
    up = None
    try:
        up, s = upscale_array(sess, rgb, tile, ctx.progress, ctx.check_deadline)
    except Exception as e:  # noqa: BLE001 — only OOM on a big tile falls back
        if 0 < tile <= FALLBACK_TILE or classify(e).code != OOM:
            raise
        logger.warning("upscale tile=%s out of memory on %dx%d: falling back to tile=%d",
                       tile, src.width, src.height, FALLBACK_TILE)
    if up is None:  # outside the except: the failed attempt's buffers are freed first
        up, s = upscale_array(sess, rgb, FALLBACK_TILE, ctx.progress, ctx.check_deadline)
    out = Image.fromarray((up * 255.0 + 0.5).astype(np.uint8), "RGB")
    want = params["scale"]
    target = (src.width * want, src.height * want)
    if out.size != target:
        out = out.resize(target, Image.LANCZOS)
    if alpha is not None:
        out.putalpha(alpha.resize(target, Image.LANCZOS))
    return [image_output(out, params["format"])]
