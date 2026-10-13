"""Input loading + output writing for tools (Pillow; ffprobe for video).

Inputs arrive as a local path, ``bytes`` or an already-open PIL image. Loading
enforces the manifest limits (``max_mb``, ``max_megapixels``) BEFORE decoding
the full raster — Pillow's lazy ``open`` reads only the header — so a 30k x 30k
PNG is a ``too_large`` error, not an OOM. EXIF orientation is applied once on
load so every tool sees the image the way a viewer shows it.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from imference_engine.tools.errors import INVALID_INPUT, TOO_LARGE, UNSUPPORTED_RUNTIME, ToolError
from imference_engine.tools.schema import InputSpec

# Pillow's decompression-bomb guard would raise a generic error far above our
# limits; our own max_megapixels check runs first and is the real gate.
_HARD_MAX_PIXELS = 400_000_000


def _size_mb(src: Any) -> Optional[float]:
    if isinstance(src, (bytes, bytearray)):
        return len(src) / 1e6
    if isinstance(src, (str, os.PathLike)):
        try:
            return os.path.getsize(src) / 1e6
        except OSError:
            return None
    return None


def load_image(src: Any, spec: Optional[InputSpec] = None, *, mode: Optional[str] = None):
    """Open ``src`` as a PIL image, enforcing ``spec`` limits."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    name = spec.name if spec else "image"
    if isinstance(src, Image.Image):
        img = src
    else:
        mb = _size_mb(src)
        if spec and spec.max_mb and mb is not None and mb > spec.max_mb:
            raise ToolError(TOO_LARGE, f"input {name!r} is {mb:.1f} MB (max {spec.max_mb} MB)", field=name)
        if isinstance(src, (str, os.PathLike)) and not os.path.isfile(src):
            raise ToolError(INVALID_INPUT, f"input {name!r}: file not found: {src}", field=name)
        Image.MAX_IMAGE_PIXELS = _HARD_MAX_PIXELS
        try:
            img = Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            raise ToolError(INVALID_INPUT, f"input {name!r} is not a readable image: {e}", field=name) from None
    w, h = img.size
    if spec and spec.max_megapixels and w * h > spec.max_megapixels * 1e6:
        raise ToolError(TOO_LARGE, f"input {name!r} is {w}x{h} ({w*h/1e6:.1f} MP, max {spec.max_megapixels} MP)",
                        field=name)
    src_format = getattr(img, "format", None)
    try:
        img.load()
        img = ImageOps.exif_transpose(img) or img
    except OSError as e:
        raise ToolError(INVALID_INPUT, f"input {name!r} is truncated or corrupt: {e}", field=name) from None
    if mode and img.mode != mode:
        img = img.convert(mode)
    # exif_transpose / convert return copies without .format; keep it so
    # format="same" can round-trip a JPEG as a JPEG.
    img.format = src_format
    return img


def load_mask(src: Any, spec: Optional[InputSpec], size: tuple):
    """Mask as an 'L' image of ``size`` (white = selected). Alpha-only masks
    (transparent everywhere but the painted strokes) are accepted too."""
    from PIL import Image

    m = load_image(src, spec)
    if m.mode in ("RGBA", "LA") and m.getextrema()[-1][0] < 255:
        # Painted strokes on a transparent layer: alpha is the mask.
        m = m.getchannel("A")
    else:
        m = m.convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return m


def image_format(img, requested: Optional[str] = None, *, default: str = "png") -> str:
    """Resolve an output format: explicit param > source format > default."""
    if requested and requested != "same":
        return requested
    fmt = (getattr(img, "format", None) or "").lower()
    if fmt in ("jpeg", "jpg", "mpo"):
        return "jpeg"
    if fmt in ("png", "webp"):
        return fmt
    return default


_EXT = {"png": ".png", "jpeg": ".jpg", "webp": ".webp", "gif": ".gif", "mp4": ".mp4", "webm": ".webm"}
_MIME = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif",
         "mp4": "video/mp4", "webm": "video/webm", "txt": "text/plain", "json": "application/json"}


def ext_for(fmt: str) -> str:
    return _EXT.get(fmt, "." + fmt)


def mime_for(fmt: str) -> str:
    return _MIME.get(fmt, "application/octet-stream")


def flatten(img, background: str = "#ffffff"):
    """RGBA/LA/P-with-transparency -> RGB over ``background``."""
    from PIL import Image

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, background)
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return img.convert("RGB") if img.mode != "RGB" else img


def save_image(img, path: Path, fmt: str, *, quality: int = 92, lossless: bool = False,
               background: str = "#ffffff", keep_metadata: bool = False) -> Path:
    """Write ``img`` as ``fmt``. Metadata is dropped unless ``keep_metadata``
    (tools produce new pixels; carrying the old prompt/EXIF along is a leak)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {}
    if fmt == "jpeg":
        img = flatten(img, background)
        kwargs = {"quality": quality, "optimize": True}
    elif fmt == "webp":
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in img.mode or "transparency" in img.info else "RGB")
        kwargs = {"quality": quality, "lossless": lossless, "method": 4}
    elif fmt == "png":
        if img.mode not in ("RGB", "RGBA", "L", "LA", "P", "I", "1"):
            img = img.convert("RGBA")
        kwargs = {"optimize": False, "compress_level": 6}
    else:
        raise ToolError(INVALID_INPUT, f"unsupported image format {fmt!r}")
    if keep_metadata:
        if "exif" in img.info:
            kwargs["exif"] = img.info["exif"]
    img.save(path, format=fmt.upper(), **kwargs)
    return path


# -------------------------------------------------------------------- video


def ffmpeg_bin() -> str:
    """ffmpeg path: $IMFERENCE_FFMPEG > PATH > imageio-ffmpeg's bundled binary."""
    env = os.environ.get("IMFERENCE_FFMPEG")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise ToolError(UNSUPPORTED_RUNTIME, "ffmpeg not found (install ffmpeg or imageio-ffmpeg, or set IMFERENCE_FFMPEG)")


def ffprobe_bin() -> Optional[str]:
    env = os.environ.get("IMFERENCE_FFPROBE")
    if env and os.path.isfile(env):
        return env
    return shutil.which("ffprobe")


def probe_video(path: str) -> dict:
    """{duration, width, height, fps, frames?} — via ffprobe, or by parsing
    ``ffmpeg -i`` stderr when only ffmpeg is available (imageio-ffmpeg ships no
    ffprobe)."""
    import re

    if not os.path.isfile(path):
        raise ToolError(INVALID_INPUT, f"video not found: {path}")
    probe = ffprobe_bin()
    if probe:
        r = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", path],
                           capture_output=True, text=True)
        if r.returncode == 0:
            try:
                d = json.loads(r.stdout)
                st = (d.get("streams") or [{}])[0]
                num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
                fps = float(num) / float(den or 1) if float(den or 1) else 0.0
                dur = float((d.get("format") or {}).get("duration") or 0)
                if st.get("width"):
                    return {"duration": dur, "width": int(st["width"]), "height": int(st["height"]), "fps": fps}
            except (ValueError, KeyError, ZeroDivisionError):
                pass
        raise ToolError(INVALID_INPUT, f"not a readable video: {os.path.basename(path)}")
    r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", path], capture_output=True, text=True)
    err = r.stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    v = re.search(r"Video: .*?(\d{2,5})x(\d{2,5})", err)
    f = re.search(r"(\d+(?:\.\d+)?) fps", err) or re.search(r"(\d+(?:\.\d+)?) tbr", err)
    if not v:
        raise ToolError(INVALID_INPUT, f"not a readable video: {os.path.basename(path)}")
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    return {"duration": dur, "width": int(v.group(1)), "height": int(v.group(2)),
            "fps": float(f.group(1)) if f else 0.0}


def check_video(path: str, spec: Optional[InputSpec]) -> dict:
    name = spec.name if spec else "video"
    mb = _size_mb(path)
    if spec and spec.max_mb and mb is not None and mb > spec.max_mb:
        raise ToolError(TOO_LARGE, f"input {name!r} is {mb:.1f} MB (max {spec.max_mb} MB)", field=name)
    info = probe_video(path)
    if spec and spec.max_seconds and info["duration"] > spec.max_seconds:
        raise ToolError(TOO_LARGE, f"input {name!r} is {info['duration']:.0f} s (max {spec.max_seconds:.0f} s)", field=name)
    return info


def run_ffmpeg(args: list, *, timeout: Optional[float] = None) -> None:
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        from imference_engine.tools.errors import TIMEOUT
        raise ToolError(TIMEOUT, "ffmpeg timed out") from None
    if r.returncode != 0:
        msg = (r.stderr or "").strip().splitlines()[-3:]
        raise ToolError(INVALID_INPUT if "Invalid data" in (r.stderr or "") else "internal",
                        "ffmpeg failed: " + " | ".join(msg))
