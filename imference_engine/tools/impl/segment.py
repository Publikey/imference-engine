"""Click-to-select: MobileSAM (ONNX encoder + SAM prompt decoder).

The encoder runs once per image (cached by image hash, so successive clicks on
the same image only re-run the ~10 ms decoder). Encoder exports come in two
layouts, both handled: ``[1, 3, 1024, 1024]`` normalized NCHW, or ``[H, W, 3]``
raw RGB with normalization inside the graph (samexporter). The decoder is the
standard segment-anything ONNX export (image_embeddings, point_coords,
point_labels, mask_input, has_mask_input, orig_im_size).
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict

from imference_engine.tools import media
from imference_engine.tools.errors import INVALID_INPUT, ToolError
from imference_engine.tools.impl import onnx_util
from imference_engine.tools.impl.erase import dilate
from imference_engine.tools.progress import LOADING, RUNNING
from imference_engine.tools.runner import ToolOutput

LONG = 1024
_MEAN = (123.675, 116.28, 103.53)
_STD = (58.395, 57.12, 57.375)
_EMB_CACHE: "OrderedDict" = OrderedDict()
_EMB_MAX = 4


def _prep(img):
    """Resize longest side to 1024, pad bottom/right. Returns (array HxWx3 float32 0-255 padded, scale)."""
    import numpy as np
    from PIL import Image

    w, h = img.size
    s = LONG / max(w, h)
    nw, nh = int(round(w * s)), int(round(h * s))
    arr = np.zeros((LONG, LONG, 3), dtype=np.float32)
    arr[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    return arr, s


def embed(enc, img):
    import numpy as np

    key = (id(enc), hashlib.sha1(img.tobytes()).hexdigest(), img.size)
    if key in _EMB_CACHE:
        _EMB_CACHE.move_to_end(key)
        return _EMB_CACHE[key]
    arr, s = _prep(img)
    inp = enc.get_inputs()[0]
    if len(inp.shape) == 3:  # raw HWC, graph normalizes
        x = arr
    else:
        x = ((arr - np.array(_MEAN, np.float32)) / np.array(_STD, np.float32)).transpose(2, 0, 1)[None]
    emb = enc.run(None, {inp.name: x.astype(np.float32)})[0]
    _EMB_CACHE[key] = (emb, s)
    while len(_EMB_CACHE) > _EMB_MAX:
        _EMB_CACHE.popitem(last=False)
    return emb, s


def decode(dec, emb, scale, points, size):
    import numpy as np
    from PIL import Image

    w, h = size
    coords = [[p["x"] * scale, p["y"] * scale] for p in points] + [[0.0, 0.0]]
    labels = [float(p["label"]) for p in points] + [-1.0]  # padding point (no box)
    feeds_all = {
        "image_embeddings": emb.astype(np.float32),
        "point_coords": np.array([coords], dtype=np.float32),
        "point_labels": np.array([labels], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros((1,), dtype=np.float32),
        "orig_im_size": np.array([h, w], dtype=np.float32),
    }
    names = [i.name for i in dec.get_inputs()]
    missing = [n for n in names if n not in feeds_all]
    if missing:
        raise ToolError("internal", f"unexpected SAM decoder inputs {missing}")
    outs = dec.run(None, {n: feeds_all[n] for n in names})
    masks = outs[0]
    if len(outs) > 1 and masks.shape[1] > 1:  # multimask: take the best IoU
        best = int(np.argmax(outs[1][0]))
        logits = masks[0, best]
    else:
        logits = masks[0, 0]
    m = Image.fromarray(((logits > 0) * 255).astype(np.uint8), "L")
    if m.size != (w, h):
        if m.size == (LONG, LONG) or m.size[0] == m.size[1]:
            # Padded square space: crop the valid area before resizing back.
            k = m.size[0] / LONG
            m = m.crop((0, 0, int(round(w * scale * k)), int(round(h * scale * k))))
        m = m.resize((w, h), Image.BILINEAR).point(lambda v: 255 if v >= 128 else 0)
    return m


def run(ctx, inputs, params):
    src = media.load_image(inputs["image"], ctx.manifest.inputs["image"], mode="RGB")
    pts = params["points"]
    if not pts:
        raise ToolError(INVALID_INPUT, "give at least one click", field="points")
    for p in pts:
        if not (0 <= p["x"] < src.width and 0 <= p["y"] < src.height):
            raise ToolError(INVALID_INPUT, f"click ({p['x']:.0f},{p['y']:.0f}) is outside the "
                            f"{src.width}x{src.height} image", field="points")
    ctx.progress(LOADING, 0.0)
    enc, ctx.runtime = onnx_util.session(ctx.weights["encoder"], ctx.device)
    dec, _ = onnx_util.session(ctx.weights["decoder"], "cpu")
    ctx.progress(RUNNING, 0.1)
    emb, s = embed(enc, src)
    ctx.progress(RUNNING, 0.8)
    mask = dilate(decode(dec, emb, s, pts, src.size), params["dilate"])
    ctx.progress(RUNNING, 1.0)
    return [ToolOutput(name="mask", type="mask", format="png", image=mask)]
