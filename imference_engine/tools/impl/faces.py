"""Face restoration: YuNet detection + GFPGAN v1.4 (ONNX), pasted back.

For each detected face (5 landmarks), the face is warped onto the FFHQ
512x512 template with a similarity transform, restored, warped back with the
inverse transform and blended through a feathered ellipse — the rest of the
image is untouched. Pure numpy + Pillow (no OpenCV needed); YuNet's decoding
is implemented here for its ONNX export (``face_detection_yunet_2023mar``).

Graph contracts: GFPGAN input ``[1, 3, 512, 512]`` RGB in [-1, 1], output the
same. YuNet input ``[1, 3, H, W]`` BGR 0-255 (H, W multiples of 32), outputs
cls/obj/bbox/kps per stride 8/16/32.
"""
from __future__ import annotations

from imference_engine.tools import media
from imference_engine.tools.impl import onnx_util
from imference_engine.tools.progress import LOADING, RUNNING
from imference_engine.tools.runner import ToolOutput

# FFHQ 5-point template (eyes, nose, mouth corners) for a 512 crop.
TEMPLATE = (
    (192.98138, 239.94708), (318.90277, 240.19366), (256.63416, 314.01935),
    (201.26117, 371.41043), (313.08905, 371.15118),
)
FACE = 512


def similarity(src_pts, dst_pts):
    """Least-squares similarity (scale+rotation+translation) mapping src->dst.
    Returns 2x3 matrix [[a, -b, tx], [b, a, ty]]."""
    import numpy as np

    s = np.asarray(src_pts, dtype=np.float64)
    d = np.asarray(dst_pts, dtype=np.float64)
    n = len(s)
    A = np.zeros((2 * n, 4))
    y = np.zeros(2 * n)
    A[0::2] = np.c_[s[:, 0], -s[:, 1], np.ones(n), np.zeros(n)]
    A[1::2] = np.c_[s[:, 1], s[:, 0], np.zeros(n), np.ones(n)]
    y[0::2], y[1::2] = d[:, 0], d[:, 1]
    a, b, tx, ty = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty]])


def invert(M):
    import numpy as np

    full = np.vstack([M, [0, 0, 1]])
    return np.linalg.inv(full)[:2]


def warp(img, M, size):
    """Warp PIL ``img`` by forward affine ``M`` (src->dst) into ``size``."""
    from PIL import Image

    inv = invert(M)  # PIL wants dst->src coefficients
    return img.transform(size, Image.AFFINE, tuple(inv.flatten()), resample=Image.BICUBIC)


def detect_yunet(sess, img, score_thr: float = 0.7, nms_thr: float = 0.3, top_k: int = 50):
    """-> list of (score, [x0,y0,x1,y1], [(x,y)*5]) in image pixels."""
    import numpy as np
    from PIL import Image

    W, H = img.size
    s = 640 / max(W, H) if max(W, H) > 640 else 1.0
    w, h = int(W * s), int(H * s)
    pw, ph = (w + 31) // 32 * 32, (h + 31) // 32 * 32
    canvas = np.zeros((ph, pw, 3), dtype=np.float32)
    canvas[:h, :w] = np.asarray(img.convert("RGB").resize((w, h), Image.BILINEAR), dtype=np.float32)[:, :, ::-1]
    inp = sess.get_inputs()[0]
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {inp.name: canvas.transpose(2, 0, 1)[None]})))
    dets = []
    for stride in (8, 16, 32):
        cls = outs.get(f"cls_{stride}")
        if cls is None:
            continue
        obj, bbox, kps = outs[f"obj_{stride}"], outs[f"bbox_{stride}"], outs[f"kps_{stride}"]
        cols, rows = pw // stride, ph // stride
        cls, obj = cls.reshape(-1), obj.reshape(-1)
        bbox, kps = bbox.reshape(-1, 4), kps.reshape(-1, 10)
        score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
        for i in np.where(score >= score_thr)[0]:
            r, c = divmod(int(i), cols)
            cx, cy = (c + bbox[i, 0]) * stride, (r + bbox[i, 1]) * stride
            bw, bh = np.exp(bbox[i, 2]) * stride, np.exp(bbox[i, 3]) * stride
            pts = [((c + kps[i, 2 * k]) * stride / s, (r + kps[i, 2 * k + 1]) * stride / s) for k in range(5)]
            dets.append((float(score[i]), [(cx - bw / 2) / s, (cy - bh / 2) / s, (cx + bw / 2) / s, (cy + bh / 2) / s], pts))
        del rows
    dets.sort(key=lambda d: -d[0])
    keep = []
    for d in dets:
        if all(_iou(d[1], k[1]) < nms_thr for k in keep):
            keep.append(d)
        if len(keep) >= top_k:
            break
    return keep


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def restore_face(sess, face):
    import numpy as np
    from PIL import Image

    x = np.asarray(face.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    y = sess.run(None, {sess.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
    y = (np.clip(y.transpose(1, 2, 0), -1, 1) + 1.0) * 127.5
    return Image.fromarray(y.astype(np.uint8), "RGB")


def _face_mask():
    from PIL import Image, ImageDraw, ImageFilter

    m = Image.new("L", (FACE, FACE), 0)
    ImageDraw.Draw(m).ellipse((48, 24, FACE - 48, FACE - 8), fill=255)
    return m.filter(ImageFilter.GaussianBlur(24))


def run(ctx, inputs, params):
    from PIL import Image

    src = media.load_image(inputs["image"], ctx.manifest.inputs["image"])
    alpha = src.getchannel("A") if src.mode == "RGBA" else None
    rgb = src.convert("RGB")
    ctx.progress(LOADING, 0.0)
    det, _ = onnx_util.session(ctx.weights["detector"], "cpu")
    gfp, ctx.runtime = onnx_util.session(ctx.weights["restorer"], ctx.device)
    ctx.progress(RUNNING, 0.05)
    faces = detect_yunet(det, rgb)[: params["max_faces"]]
    out = rgb.copy()
    base_mask = _face_mask()
    strength = params["strength"]
    for i, (_, _, pts) in enumerate(faces):
        ctx.check_deadline()
        M = similarity(pts, TEMPLATE)
        crop = warp(rgb, M, (FACE, FACE))
        fixed = restore_face(gfp, crop)
        if strength < 1:
            fixed = Image.blend(crop, fixed, strength)
        back = warp(fixed, invert(M), rgb.size)
        mask = warp(base_mask, invert(M), rgb.size)
        out.paste(back, (0, 0), mask)
        ctx.progress(RUNNING, 0.05 + 0.95 * (i + 1) / len(faces))
    ctx.progress(RUNNING, 1.0)
    if alpha is not None:
        out.putalpha(alpha)
    return [ToolOutput(name="image", type="image", format="png", image=out)]
