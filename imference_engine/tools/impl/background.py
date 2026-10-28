"""Background removal: BiRefNet-lite / ISNet / U2-Net-p (ONNX, rembg exports).

Each graph takes ``[1, 3, S, S]`` normalized RGB and returns a saliency map;
the per-model recipe (size, mean/std, sigmoid or min-max) is below. The map is
resized back to the source size and used as alpha — a soft matte by default,
or a hard cut with ``threshold``.
"""
from __future__ import annotations

from imference_engine.tools import media
from imference_engine.tools.impl import onnx_util
from imference_engine.tools.progress import LOADING, RUNNING
from imference_engine.tools.runner import ToolOutput, image_output

_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
RECIPES = {
    #               size  mean/std            post
    "birefnet-lite": (1024, _IMAGENET, "sigmoid"),
    "isnet": (1024, ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)), "minmax"),
    "u2netp": (320, _IMAGENET, "minmax"),
}


def predict_matte(sess, img, model: str):
    """PIL RGB -> PIL 'L' matte of the same size."""
    import numpy as np
    from PIL import Image

    size, (mean, std), post = RECIPES[model]
    sh, sw = onnx_util.input_hw(sess, size)
    x = np.asarray(img.convert("RGB").resize((sw, sh), Image.BICUBIC), dtype=np.float32) / 255.0
    x = (x - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    # BiRefNet exports list the finest prediction LAST; rembg's isnet/u2net FIRST.
    pred = outs[-1] if model == "birefnet-lite" else outs[0]
    pred = np.asarray(pred, dtype=np.float32).reshape(pred.shape[-2], pred.shape[-1])
    if post == "sigmoid":
        pred = 1.0 / (1.0 + np.exp(-pred))
    else:
        lo, hi = float(pred.min()), float(pred.max())
        pred = (pred - lo) / (hi - lo) if hi > lo else np.zeros_like(pred)
    m = Image.fromarray((np.clip(pred, 0, 1) * 255).astype(np.uint8), "L")
    return m.resize(img.size, Image.BICUBIC)


def run(ctx, inputs, params):
    from PIL import Image

    src = media.load_image(inputs["image"], ctx.manifest.inputs["image"], mode="RGB")
    ctx.progress(LOADING, 0.0)
    sess, ctx.runtime = onnx_util.session(ctx.weights["model"], ctx.device)
    ctx.progress(RUNNING, 0.1)
    matte = predict_matte(sess, src, params["model"])
    thr = params.get("threshold")
    if thr is not None:
        cut = int(round(thr * 255))
        matte = matte.point(lambda v: 255 if v >= cut else 0)
    ctx.progress(RUNNING, 1.0)
    mode = params["output"]
    if mode == "mask":
        return [ToolOutput(name="image", type="image", format="png", image=matte)]
    cut_out = src.convert("RGBA")
    cut_out.putalpha(matte)
    if mode == "color":
        bg = Image.new("RGBA", src.size, params["background"])
        bg.alpha_composite(cut_out)
        return [image_output(bg.convert("RGB"), params["format"])]
    return [image_output(cut_out, params["format"])]
