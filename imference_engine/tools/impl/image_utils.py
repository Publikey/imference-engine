"""Pillow utilities: resize, crop, convert/compress, strip metadata. No AI,
no weights — CPU only, milliseconds."""
from __future__ import annotations

import io

from imference_engine.tools import media
from imference_engine.tools.errors import INVALID_INPUT, ToolError
from imference_engine.tools.progress import RUNNING
from imference_engine.tools.runner import ToolOutput


def _src(ctx, inputs):
    return media.load_image(inputs["image"], ctx.manifest.inputs["image"])


def _out(img, src, fmt_param, **save):
    fmt = media.image_format(src, fmt_param)
    if fmt == "jpeg":
        save.setdefault("quality", 92)
    return ToolOutput(name="image", type="image", format=fmt, image=img, save_options=save)


def resize(ctx, inputs, params):
    from PIL import Image, ImageOps

    ctx.runtime = "pillow"
    src = _src(ctx, inputs)
    ctx.progress(RUNNING, 0.0)
    w, h, scale, mode = params.get("width"), params.get("height"), params.get("scale"), params["mode"]
    sw, sh = src.size
    if not w and not h:
        if not scale:
            raise ToolError(INVALID_INPUT, "give width, height or scale", field="width")
        w, h = max(1, round(sw * scale)), max(1, round(sh * scale))
        mode = "stretch"
    elif not w:
        w = max(1, round(sw * h / sh))
        mode = "stretch"
    elif not h:
        h = max(1, round(sh * w / sw))
        mode = "stretch"
    if w * h > 64_000_000:
        raise ToolError(INVALID_INPUT, f"output {w}x{h} is over 64 MP", field="width")
    if mode == "stretch":
        out = src.resize((w, h), Image.LANCZOS)
    elif mode == "fit":
        out = ImageOps.contain(src, (w, h), Image.LANCZOS)
    else:  # fill
        out = ImageOps.fit(src, (w, h), Image.LANCZOS)
    ctx.progress(RUNNING, 1.0)
    return [_out(out, src, params["format"])]


def crop(ctx, inputs, params):
    ctx.runtime = "pillow"
    src = _src(ctx, inputs)
    sw, sh = src.size
    if params.get("aspect"):
        a, b = (int(v) for v in params["aspect"].split(":"))
        if a == 0 or b == 0:
            raise ToolError(INVALID_INPUT, "aspect parts must be > 0", field="aspect")
        if sw / sh > a / b:
            cw, ch = round(sh * a / b), sh
        else:
            cw, ch = sw, round(sw * b / a)
        x, y = (sw - cw) // 2, (sh - ch) // 2
    else:
        x, y = params["x"], params["y"]
        if x >= sw or y >= sh:
            raise ToolError(INVALID_INPUT, f"crop origin ({x},{y}) is outside the {sw}x{sh} image", field="x")
        cw = min(params.get("width") or (sw - x), sw - x)
        ch = min(params.get("height") or (sh - y), sh - y)
    out = src.crop((x, y, x + cw, y + ch))
    ctx.progress(RUNNING, 1.0)
    return [_out(out, src, params["format"])]


def _encoded_size(img, fmt, quality, lossless, background) -> int:
    buf = io.BytesIO()
    if fmt == "jpeg":
        media.flatten(img, background).save(buf, "JPEG", quality=quality, optimize=True)
    else:
        im = img if img.mode in ("RGB", "RGBA") else img.convert("RGBA")
        im.save(buf, "WEBP", quality=quality, lossless=lossless, method=4)
    return buf.tell()


def convert(ctx, inputs, params):
    ctx.runtime = "pillow"
    src = _src(ctx, inputs)
    fmt, quality = params["format"], params["quality"]
    lossless, bg, max_kb = params["lossless"], params["background"], params.get("max_kb")
    if max_kb and fmt in ("jpeg", "webp") and not lossless:
        # Binary search the highest quality that fits.
        lo, hi, best = 5, quality, None
        while lo <= hi:
            mid = (lo + hi) // 2
            ctx.progress(RUNNING, 0.5)
            if _encoded_size(src, fmt, mid, False, bg) <= max_kb * 1024:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        quality = best or 5
    ctx.progress(RUNNING, 1.0)
    return [ToolOutput(name="image", type="image", format=fmt, image=src,
                       save_options={"quality": quality, "lossless": lossless, "background": bg})]


def strip_metadata(ctx, inputs, params):
    from PIL import Image

    ctx.runtime = "pillow"
    src = _src(ctx, inputs)
    # Rebuild from raw pixels: no info dict, no EXIF/XMP/ICC/text chunks travel.
    clean = Image.frombytes(src.mode, src.size, src.tobytes()) if src.mode != "P" else src.convert("RGBA")
    ctx.progress(RUNNING, 1.0)
    return [_out(clean, src, params["format"], quality=95)]
