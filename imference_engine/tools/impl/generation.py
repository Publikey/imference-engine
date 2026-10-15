"""Generation tools: vary, inpaint, outpaint, instruct-edit — on top of
``Engine.generate``.

Backend-agnostic by construction: every image backend already does img2img,
so inpainting is "img2img on a pre-filled crop + feathered composite" and
outpainting is "extend the canvas, pre-fill, inpaint the new area". No
per-backend inpaint pipeline, so any model that accepts an image input can
inpaint; pixels outside the mask are kept bit-exact.

The caller passes a loaded ``Engine`` (``run_tool(..., engine=engine)``) and
``params["model"]`` names a registered model. When the name is unknown but
the engine holds exactly one model (Desktop's single-model sidecar), that one
is used.
"""
from __future__ import annotations

import random

from imference_engine.tools import media
from imference_engine.tools.errors import INVALID_INPUT, ToolError, classify
from imference_engine.tools.impl.erase import context_box, dilate
from imference_engine.tools.progress import LOADING, RUNNING
from imference_engine.tools.runner import ToolOutput

WORK_LONG_SIDE = 1024   # generation resolution for the processed region
OUTPAINT_STRENGTH = 0.85  # high enough to invent content, low enough to keep the layout
MULT = 64               # every backend accepts multiples of 64


def _model(engine, name: str) -> str:
    names = engine.registered_models() if hasattr(engine, "registered_models") else []
    if not names or name in names:
        return name
    if len(names) == 1:
        return names[0]
    raise ToolError(INVALID_INPUT, f"model {name!r} is not loaded (have {names})", field="model")


def _work_size(w: int, h: int, long_side: int = WORK_LONG_SIDE) -> tuple:
    s = long_side / max(w, h)
    return max(MULT, round(w * s / MULT) * MULT), max(MULT, round(h * s / MULT) * MULT)


def _generate(ctx, params, source, strength: float):
    engine = ctx.engine
    name = _model(engine, params["model"])
    if hasattr(engine, "model_backend") and engine.model_backend(name) in EDIT_BACKENDS:
        raise ToolError("unsupported_runtime", f"model {name!r} is an instruction-edit model: "
                        "use instruct-edit, or load an image generation model", field="model")
    seed = params.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    ctx.progress(RUNNING, 0.05)
    try:
        res = engine.generate(model=name, prompt=params.get("prompt") or "",
                              negative_prompt=params.get("negative_prompt") or None,
                              source_image=source, strength=strength, seed=seed, batch=1)
    except ToolError:
        raise
    except Exception as e:
        raise classify(e) from e
    img = res.images[0] if res.images else None
    if img is None:
        err = res.errors[0].error if res.errors else "generation returned no image"
        raise classify(RuntimeError(err))
    ctx.runtime = "engine"
    ctx.progress(RUNNING, 0.95)
    return img.convert("RGB"), seed


def prefill(img, mask):
    """Replace masked pixels with a smooth fill from their surroundings, so
    img2img does not just redraw the erased object.

    Push-pull in float: average the kept pixels down a 2x pyramid, then fill
    each level's holes from the coarser one on the way back up. Any hole size
    gets the colours of its border (a fixed-radius blur leaves the centre of a
    large mask with no weight — it came out black)."""
    from PIL import Image
    import numpy as np

    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    keep = (np.asarray(mask, dtype=np.float32) < 128).astype(np.float32)[..., None]
    if keep.min() > 0:
        return img.convert("RGB")

    def half(x):
        x = np.pad(x, ((0, x.shape[0] % 2), (0, x.shape[1] % 2), (0, 0)), mode="edge")
        return x.reshape(x.shape[0] // 2, 2, x.shape[1] // 2, 2, -1).mean(axis=(1, 3))

    def up(x, shape):
        return np.stack([np.asarray(Image.fromarray(x[..., c], "F").resize((shape[1], shape[0]), Image.BILINEAR))
                         for c in range(x.shape[2])], axis=-1)

    levels = [(a * keep, keep)]                       # premultiplied colour, weight in [0, 1]
    while min(levels[-1][1].shape[:2]) > 1:
        p, w = levels[-1]
        levels.append((half(p), half(w)))
    p, w = levels[-1]
    col = np.where(w > 0, p / np.maximum(w, 1e-6), 127.5)   # all masked: neutral grey
    for p, w in reversed(levels[:-1]):
        col = p + (1.0 - w) * up(col, p.shape)
    out = np.where(keep > 0.5, a, col)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")


def _inpaint_core(ctx, params, rgb, mask, strength: float, feather: int):
    from PIL import Image, ImageFilter

    bbox = mask.getbbox()
    if bbox is None:
        raise ToolError(INVALID_INPUT, "the mask is empty: paint the area to repaint", field="mask")
    box = context_box(bbox, rgb.size, margin=0.5, min_side=384)
    crop, mcrop = rgb.crop(box), mask.crop(box)
    ws = _work_size(*crop.size)
    src = prefill(crop, mcrop).resize(ws, Image.LANCZOS)
    ctx.progress(LOADING, 0.0)
    gen, seed = _generate(ctx, params, src, strength)
    gen = gen.resize(crop.size, Image.LANCZOS)
    soft = dilate(mcrop, max(1, feather // 2)).filter(ImageFilter.GaussianBlur(max(1, feather / 2)))
    patch = crop.copy()
    patch.paste(gen, (0, 0), soft)
    out = rgb.copy()
    out.paste(patch, box[:2])
    return out, seed


def vary(ctx, inputs, params):
    from PIL import Image

    src = media.load_image(inputs["image"], ctx.manifest.inputs["image"], mode="RGB")
    ws = _work_size(*src.size, long_side=min(WORK_LONG_SIDE, max(512, max(src.size))))
    gen, seed = _generate(ctx, params, src.resize(ws, Image.LANCZOS), float(params["strength"]))
    out = gen.resize(src.size, Image.LANCZOS)
    o = ToolOutput(name="image", type="image", format="png", image=out)
    o.data = {"seed": seed}
    return [o]


def inpaint(ctx, inputs, params):
    src = media.load_image(inputs["image"], ctx.manifest.inputs["image"])
    alpha = src.getchannel("A") if src.mode == "RGBA" else None
    rgb = src.convert("RGB")
    mask = media.load_mask(inputs["mask"], ctx.manifest.inputs["mask"], rgb.size).point(lambda v: 255 if v >= 128 else 0)
    out, seed = _inpaint_core(ctx, params, rgb, mask, params["strength"], params["feather"])
    if alpha is not None:
        out.putalpha(alpha)
    o = ToolOutput(name="image", type="image", format="png", image=out)
    o.data = {"seed": seed}
    return [o]


def outpaint(ctx, inputs, params):
    from PIL import Image, ImageDraw, ImageFilter

    src = media.load_image(inputs["image"], ctx.manifest.inputs["image"], mode="RGB")
    l, r, t, b = params["left"], params["right"], params["top"], params["bottom"]
    if not (l or r or t or b):
        raise ToolError(INVALID_INPUT, "set at least one of left/right/top/bottom", field="left")
    W, H = src.width + l + r, src.height + t + b
    if W * H > 16_000_000:
        raise ToolError(INVALID_INPUT, f"expanded canvas {W}x{H} is over 16 MP", field="left")
    # Seed the new area with the stretched edges, then blur: a color-correct
    # starting point the model can turn into content.
    canvas = src.resize((W, H), Image.BILINEAR).filter(ImageFilter.GaussianBlur(max(W, H) // 24))
    canvas.paste(src, (l, t))
    mask = Image.new("L", (W, H), 255)
    overlap = 8  # repaint a thin band of the original so the seam blends
    ImageDraw.Draw(mask).rectangle((l + (overlap if l else 0), t + (overlap if t else 0),
                                    l + src.width - 1 - (overlap if r else 0),
                                    t + src.height - 1 - (overlap if b else 0)), fill=0)
    # Process the whole canvas at work resolution (the model needs the
    # original as context to continue it), composite only the new area.
    ws = _work_size(W, H)
    ctx.progress(LOADING, 0.0)
    gen, seed = _generate(ctx, params, canvas.resize(ws, Image.LANCZOS), OUTPAINT_STRENGTH)
    gen = gen.resize((W, H), Image.LANCZOS)
    soft = mask.filter(ImageFilter.GaussianBlur(overlap / 2))
    out = canvas.copy()
    out.paste(gen, (0, 0), soft)
    o = ToolOutput(name="image", type="image", format="png", image=out)
    o.data = {"seed": seed}
    return [o]


EDIT_BACKENDS = ("qwenimage-edit",)


def instruct_edit(ctx, inputs, params):
    """Instruction editing on an edit model (Qwen-Image-Edit): the image (plus
    optional references) is the condition, the instruction is the prompt."""
    from PIL import Image

    engine = ctx.engine
    model = _model(engine, params["model"])
    backend = engine.model_backend(model) if hasattr(engine, "model_backend") else None
    if backend is not None and backend not in EDIT_BACKENDS:
        raise ToolError("unsupported_runtime", f"model {model!r} ({backend}) is not an instruction-edit model; "
                        "load an edit model such as Qwen-Image-Edit-2511", field="model")
    src = media.load_image(inputs["image"], ctx.manifest.inputs["image"], mode="RGB")
    refs = [media.load_image(r, ctx.manifest.inputs["references"], mode="RGB")
            for r in (inputs.get("references") or [])]
    seed = params.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    ctx.progress(LOADING, 0.0)
    ctx.progress(RUNNING, 0.05)
    try:
        res = engine.generate(model=model, prompt=params["instruction"],
                              negative_prompt=params.get("negative_prompt") or None,
                              source_image=[src, *refs] if refs else src, seed=seed, batch=1,
                              num_steps=params.get("steps"), guidance_scale=params.get("guidance"))
    except ToolError:
        raise
    except Exception as e:
        raise classify(e) from e
    img = res.images[0] if res.images else None
    if img is None:
        raise classify(RuntimeError(res.errors[0].error if res.errors else "edit returned no image"))
    ctx.runtime = "engine"
    img = img.convert("RGB")
    if params["keep_size"] and img.size != src.size:
        img = img.resize(src.size, Image.LANCZOS)
    ctx.progress(RUNNING, 1.0)
    o = ToolOutput(name="image", type="image", format="png", image=img)
    o.data = {"seed": seed}
    return [o]
