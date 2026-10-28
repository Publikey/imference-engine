"""Magic eraser: LaMa inpainting (ONNX, fixed 512x512 graph).

Only the region around the mask is processed: its bounding box (plus margin
for context) is cropped, squared, resized to 512, inpainted, resized back and
composited through a feathered mask — so a 4K photo with a small blemish keeps
full resolution everywhere else.

Graph contract (Carve/LaMa-ONNX): inputs ``image`` [1,3,512,512] in [0,1] and
``mask`` [1,1,512,512] in {0,1}; output [1,3,512,512] in [0,255] (some exports
in [0,1] — detected).
"""
from __future__ import annotations

from imference_engine.tools import media
from imference_engine.tools.errors import INVALID_INPUT, ToolError
from imference_engine.tools.impl import onnx_util
from imference_engine.tools.progress import LOADING, RUNNING
from imference_engine.tools.runner import image_output

SIZE = 512


def dilate(mask, px: int):
    from PIL import ImageFilter

    if px <= 0:
        return mask
    # MaxFilter needs an odd size; cap the kernel and iterate for big radii.
    out = mask
    left = px
    while left > 0:
        k = min(left, 10)
        out = out.filter(ImageFilter.MaxFilter(2 * k + 1))
        left -= k
    return out


def context_box(bbox, size, margin: float = 0.6, min_side: int = 256):
    """Square box around ``bbox`` grown by ``margin`` (fraction of its side),
    clamped to the image."""
    x0, y0, x1, y1 = bbox
    W, H = size
    side = max(x1 - x0, y1 - y0)
    side = int(max(min_side, side * (1 + 2 * margin)))
    side = min(side, max(W, H))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bx0 = int(round(cx - side / 2))
    by0 = int(round(cy - side / 2))
    bx0 = max(0, min(bx0, W - side)) if side <= W else 0
    by0 = max(0, min(by0, H - side)) if side <= H else 0
    return bx0, by0, min(W, bx0 + side), min(H, by0 + side)


def inpaint_region(sess, image, mask):
    """image: PIL RGB, mask: PIL L (same size, 255 = fill). Returns PIL RGB."""
    import numpy as np
    from PIL import Image

    w, h = image.size
    img = np.asarray(image.resize((SIZE, SIZE), Image.BICUBIC), dtype=np.float32) / 255.0
    m = np.asarray(mask.resize((SIZE, SIZE), Image.NEAREST), dtype=np.float32) / 255.0
    m = (m > 0.5).astype(np.float32)
    names = [i.name for i in sess.get_inputs()]
    feeds = {names[0]: img.transpose(2, 0, 1)[None], names[1]: m[None, None]}
    out = sess.run(None, feeds)[0][0].transpose(1, 2, 0)
    if out.max() <= 1.5:
        out = out * 255.0
    res = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    return res.resize((w, h), Image.BICUBIC)


def run(ctx, inputs, params):
    from PIL import ImageFilter

    src = media.load_image(inputs["image"], ctx.manifest.inputs["image"])
    alpha = src.getchannel("A") if src.mode == "RGBA" else None
    rgb = src.convert("RGB")
    mask = media.load_mask(inputs["mask"], ctx.manifest.inputs["mask"], rgb.size)
    mask = mask.point(lambda v: 255 if v >= 128 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ToolError(INVALID_INPUT, "the mask is empty: paint the area to erase", field="mask")
    mask = dilate(mask, params["dilate"])
    bbox = mask.getbbox()
    ctx.progress(LOADING, 0.0)
    sess, ctx.runtime = onnx_util.session(ctx.weights["model"], ctx.device)
    ctx.progress(RUNNING, 0.1)
    box = context_box(bbox, rgb.size)
    filled = inpaint_region(sess, rgb.crop(box), mask.crop(box))
    ctx.progress(RUNNING, 0.9)
    # Feathered composite: original pixels outside the mask stay bit-exact.
    soft = mask.crop(box).filter(ImageFilter.GaussianBlur(2))
    patch = rgb.crop(box)
    patch.paste(filled, (0, 0), soft)
    out = rgb.copy()
    out.paste(patch, box[:2])
    if alpha is not None:
        out.putalpha(alpha)
    ctx.progress(RUNNING, 1.0)
    return [image_output(out, params["format"])]
