"""Tests for imference_engine.tools: catalog, validation, weights, runner and
every implementation (with tiny synthetic ONNX graphs instead of real weights)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from imference_engine.tools import catalog, run_tool  # noqa: E402
from imference_engine.tools import weights as tw  # noqa: E402
from imference_engine.tools.errors import ToolError, classify  # noqa: E402
from imference_engine.tools.schema import ManifestError, parse_manifest  # noqa: E402

HAS_ORT = True
try:
    import onnxruntime  # noqa: F401
    import onnx  # noqa: F401
except ImportError:
    HAS_ORT = False
needs_ort = pytest.mark.skipif(not HAS_ORT, reason="onnx/onnxruntime not installed")


def _ffmpeg():
    """The binary the tools themselves use (PATH or imageio-ffmpeg), or None."""
    from imference_engine.tools.media import ffmpeg_bin
    try:
        return ffmpeg_bin()
    except ToolError:
        return None


FFMPEG = _ffmpeg()
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")


# ----------------------------------------------------------------- fixtures

@pytest.fixture
def cache(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    monkeypatch.setenv("TOOLS_MODEL_CACHE", str(d))
    monkeypatch.delenv("TOOLS_MODEL_CDN", raising=False)
    monkeypatch.delenv("IMAGE_MODEL_CDN", raising=False)
    from imference_engine.tools.impl import onnx_util
    onnx_util.release()
    yield d
    onnx_util.release()


def put_weight(cache: Path, tool: str, name: str, builder, *args, params=None, **kw):
    m = catalog.get(tool)
    w = [w for w in m.weights_for(m.validate_params(params or _min_params(tool))) if w.name == name][0]
    out = builder(cache / w.path, *args, **kw)
    if w.sha256:
        # The synthetic graph stands in for the real file: stamp it as already
        # verified against the manifest hash (see weights._verified).
        (cache / (w.path + ".sha256")).write_text(w.sha256)
    return out


def _min_params(tool):
    return {"points": [{"x": 1, "y": 1, "label": 1}]} if tool == "segment" else {}


def img(tmp_path, size=(64, 48), color=(10, 200, 30), mode="RGB", name="in.png", fmt=None):
    p = tmp_path / name
    Image.new(mode, size, color).save(p, format=fmt)
    return p


# ------------------------------------------------------------------ catalog

def test_builtin_catalog_parses_and_is_consistent():
    tools = catalog.builtin().all(include_unavailable=True)
    ids = {t.id for t in tools}
    assert {"upscale", "remove-background", "erase-object", "resize", "vary", "inpaint"} <= ids
    for t in tools:
        # Every enum value of a param referenced by a weight's `when` has a weight.
        whens = [w.when for w in t.weights if w.when]
        for key in {k for wh in whens for k in wh}:
            covered = {wh[key] for wh in whens if key in wh}
            assert covered == set(t.params[key].values), (t.id, key)
        if t.available:
            from imference_engine.tools.impl import implemented
            assert t.id in implemented(), t.id
        assert t.mcp_name.replace("_", "").isalnum()


def test_unknown_tool():
    assert "instruct-edit" in catalog.builtin().ids()
    with pytest.raises(ToolError) as e:
        catalog.get("no-such-tool")
    assert e.value.code == "invalid_input"


def test_for_queue_applies_license_gate():
    light = {t.id for t in catalog.for_queue("tools-light")}
    assert "upscale" in light and "remove-background" in light
    assert "restore-faces" not in light       # license not verified -> local only
    cpu = {t.id for t in catalog.for_queue("tools-cpu")}
    assert {"resize", "crop", "convert", "strip-metadata", "video-trim"} <= cpu
    assert not light & cpu


def test_export_is_public_json():
    doc = catalog.export()
    assert doc["version"] == 1
    t = {x["id"]: x for x in doc["tools"]}["upscale"]
    assert "weights" not in t and t["weights_mb"] > 0 and t["cloud_available"] is True
    json.dumps(doc)


def test_manifest_errors():
    good = json.loads(json.dumps(catalog.get("resize").to_dict()))
    parse_manifest(good)
    for mutate in (
        lambda d: d.pop("license"),
        lambda d: d.update(id="Bad Id"),
        lambda d: d.update(category="audio"),
        lambda d: d["params"].update(x={"type": "enum"}),
        lambda d: d["runtimes"].update(cloud_queue="nope"),
        lambda d: d.update(extra=1),
        lambda d: d["params"]["mode"].update(default="nope"),
    ):
        d = json.loads(json.dumps(good))
        mutate(d)
        with pytest.raises(ManifestError):
            parse_manifest(d)


# --------------------------------------------------------------- validation

@pytest.mark.parametrize("tool,params,expect", [
    ("resize", {"width": "512"}, {"width": 512, "mode": "fit", "format": "same"}),
    ("resize", {"scale": 2}, {"scale": 2.0}),
    ("upscale", {"scale": "2"}, {"scale": 2, "model": "realesrgan-x4plus", "tile": 256}),
    ("convert", {"lossless": "true"}, {"lossless": True}),
    ("segment", {"points": [[3, 4]]}, {"points": [{"x": 3.0, "y": 4.0, "label": 1}]}),
    ("segment", {"points": '[{"x":1,"y":2,"label":0}]'}, {"points": [{"x": 1.0, "y": 2.0, "label": 0}]}),
])
def test_validate_params_ok(tool, params, expect):
    out = catalog.get(tool).validate_params(params)
    for k, v in expect.items():
        assert out[k] == v


@pytest.mark.parametrize("tool,params,field", [
    ("resize", {"widht": 3}, "widht"),
    ("resize", {"width": 0}, "width"),
    ("resize", {"width": 1.5}, "width"),
    ("resize", {"width": True}, "width"),
    ("resize", {"mode": "zoom"}, "mode"),
    ("convert", {"background": "red"}, "background"),
    ("convert", {"lossless": "maybe"}, "lossless"),
    ("segment", {}, "points"),
    ("segment", {"points": [{"x": 1, "y": 1, "label": 2}]}, "points"),
    ("vary", {}, "model"),
    ("crop", {"aspect": "16x9"}, "aspect"),
    ("resize", {"scale": float("nan")}, "scale"),
])
def test_validate_params_rejects(tool, params, field):
    with pytest.raises(ToolError) as e:
        catalog.get(tool).validate_params(params)
    assert e.value.code == "invalid_input" and e.value.field == field


def test_input_name_validation():
    m = catalog.get("video-concat")
    with pytest.raises(ToolError):
        m.validate_input_names({"videos": ["a.mp4"]})          # min_items 2
    with pytest.raises(ToolError):
        m.validate_input_names({"videos": "a.mp4"})            # not a list
    with pytest.raises(ToolError):
        catalog.get("resize").validate_input_names({"image": "a", "mask": "b"})
    with pytest.raises(ToolError):
        catalog.get("resize").validate_input_names({})


def test_classify_errors():
    assert classify(MemoryError()).code == "oom"
    assert classify(RuntimeError("CUDA out of memory. Tried to allocate")).code == "oom"
    assert classify(RuntimeError("BFCArena::AllocateRawInternal Failed to allocate memory")).code == "oom"
    assert classify(TimeoutError("x")).code == "timeout"
    assert classify(ValueError("boom")).code == "internal"
    e = ToolError("oom", "x")
    assert classify(e) is e and e.cloud_retryable
    with pytest.raises(ValueError):
        ToolError("nope", "x")


# ------------------------------------------------------------------ weights

def test_weights_missing_without_mirror(tmp_path, cache):
    with pytest.raises(ToolError) as e:
        run_tool("upscale", {"image": str(img(tmp_path))}, {})
    assert e.value.code == "weights_missing"


def test_weights_download_and_checksum(tmp_path, cache):
    from imference_engine.tools.schema import WeightSpec

    payload = b"weights-bytes"
    sha = hashlib.sha256(payload).hexdigest()
    calls = []

    def fake_dl(url, dest):
        calls.append(url)
        Path(dest).write_bytes(payload)

    w = WeightSpec(name="model", path="x/model.onnx", sha256=sha, size_mb=1)
    seen = []
    out = tw.ensure([w], cdn="https://cdn.example/tools", downloader=fake_dl, on_progress=lambda s, f: seen.append(s))
    assert out["model"].read_bytes() == payload
    assert calls == ["https://cdn.example/tools/x/model.onnx"]
    assert "downloading_weights" in seen
    tw.ensure([w], cdn="https://cdn.example/tools", downloader=fake_dl)
    assert len(calls) == 1                                      # cached, stamp reused

    bad = WeightSpec(name="model", path="x/bad.onnx", sha256="0" * 64)
    with pytest.raises(ToolError) as e:
        tw.ensure([bad], cdn="https://cdn.example/tools", downloader=fake_dl)
    assert e.value.code == "weights_missing"
    assert not (cache / "x/bad.onnx").exists()


def test_cdn_fallback_to_image_cdn(monkeypatch):
    monkeypatch.delenv("TOOLS_MODEL_CDN", raising=False)
    monkeypatch.setenv("IMAGE_MODEL_CDN", "https://cdn.example/")
    assert tw.cdn_root() == "https://cdn.example/tools"
    monkeypatch.setenv("TOOLS_MODEL_CDN", "https://t.example/x/")
    assert tw.cdn_root() == "https://t.example/x"


# ---------------------------------------------------------------- utilities

def test_resize_modes(tmp_path):
    p = img(tmp_path, (400, 200))
    r = run_tool("resize", {"image": str(p)}, {"width": 100, "height": 100})
    assert r.outputs[0].image.size == (100, 50)                 # fit
    r = run_tool("resize", {"image": str(p)}, {"width": 100, "height": 100, "mode": "fill"})
    assert r.outputs[0].image.size == (100, 100)
    r = run_tool("resize", {"image": str(p)}, {"width": 100})
    assert r.outputs[0].image.size == (100, 50)
    r = run_tool("resize", {"image": str(p)}, {"scale": 0.5}, output_dir=str(tmp_path / "o"))
    assert r.outputs[0].path.suffix == ".png" and Image.open(r.outputs[0].path).size == (200, 100)
    assert r.stages[-1] == "encoding" and r.runtime == "pillow"
    with pytest.raises(ToolError):
        run_tool("resize", {"image": str(p)}, {})


def test_jpeg_stays_jpeg_and_exif_orientation(tmp_path):
    p = tmp_path / "o.jpg"
    im = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    im.save(p, exif=exif.tobytes())
    r = run_tool("resize", {"image": str(p)}, {"scale": 1.0}, output_dir=str(tmp_path / "o"))
    o = r.outputs[0]
    assert o.format == "jpeg" and o.path.suffix == ".jpg"
    assert Image.open(o.path).size == (20, 40)                  # orientation applied


def test_crop(tmp_path):
    p = img(tmp_path, (300, 100))
    assert run_tool("crop", {"image": str(p)}, {"aspect": "1:1"}).outputs[0].image.size == (100, 100)
    assert run_tool("crop", {"image": str(p)}, {"x": 250, "y": 50}).outputs[0].image.size == (50, 50)
    with pytest.raises(ToolError):
        run_tool("crop", {"image": str(p)}, {"x": 400})


def test_convert_and_max_kb(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "noise.png"
    Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(p)
    r = run_tool("convert", {"image": str(p)}, {"format": "jpeg", "max_kb": 20}, output_dir=str(tmp_path / "o"))
    assert r.outputs[0].path.suffix == ".jpg"
    assert r.outputs[0].path.stat().st_size <= 20 * 1024
    rgba = img(tmp_path, (10, 10), (0, 0, 0, 0), mode="RGBA", name="t.png")
    r = run_tool("convert", {"image": str(rgba)}, {"format": "jpeg", "background": "#ff0000"}, output_dir=str(tmp_path / "o"))
    assert Image.open(r.outputs[0].path).getpixel((5, 5))[0] > 240


def test_strip_metadata(tmp_path):
    from PIL import PngImagePlugin

    p = tmp_path / "m.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a secret prompt")
    Image.new("RGB", (8, 8)).save(p, pnginfo=info)
    assert "parameters" in Image.open(p).info
    r = run_tool("strip-metadata", {"image": str(p)}, {}, output_dir=str(tmp_path / "o"))
    assert "parameters" not in Image.open(r.outputs[0].path).info


def test_input_limits(tmp_path):
    p = tmp_path / "big.png"
    Image.new("L", (2100, 2100)).save(p)                        # 4.4 MP > upscale's 4 MP
    with pytest.raises(ToolError) as e:
        run_tool("upscale", {"image": str(p)}, {}, cdn="https://unused")
    assert e.value.code == "too_large"
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ToolError) as e:
        run_tool("resize", {"image": str(bad)}, {"scale": 1})
    assert e.value.code == "invalid_input"
    with pytest.raises(ToolError) as e:
        run_tool("resize", {"image": str(tmp_path / "nope.png")}, {"scale": 1})
    assert e.value.code == "invalid_input"


def test_needs_engine_without_engine(tmp_path):
    p = img(tmp_path)
    with pytest.raises(ToolError) as e:
        run_tool("vary", {"image": str(p)}, {"model": "x"})
    assert e.value.code == "unsupported_runtime"


# -------------------------------------------------------------------- video

@pytest.fixture
def clip(tmp_path):
    if FFMPEG is None:
        pytest.skip("ffmpeg not installed")
    p = tmp_path / "clip.mp4"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=10",
                    "-t", "2", "-pix_fmt", "yuv420p", str(p)], check=True)
    return p


def test_video_tools(tmp_path, clip):
    from imference_engine.tools.media import probe_video

    out = tmp_path / "o"
    r = run_tool("video-trim", {"video": str(clip)}, {"start": 0.5, "end": 1.5}, output_dir=str(out))
    d = probe_video(str(r.outputs[0].path))["duration"]
    assert 0.8 <= d <= 1.25
    r = run_tool("video-concat", {"videos": [str(clip), str(clip)]}, {}, output_dir=str(out))
    assert 3.5 <= probe_video(str(r.outputs[0].path))["duration"] <= 4.5
    r = run_tool("video-convert", {"video": str(clip)}, {"format": "gif", "width": 48}, output_dir=str(out))
    assert r.outputs[0].path.suffix == ".gif" and Image.open(r.outputs[0].path).size[0] == 48
    with pytest.raises(ToolError):
        run_tool("video-trim", {"video": str(clip)}, {"start": 1.5, "end": 1.0})
    with pytest.raises(ToolError) as e:
        run_tool("video-trim", {"video": str(img(tmp_path))}, {})
    assert e.value.code == "invalid_input"


# --------------------------------------------------------------- ONNX tools

def _fx():
    sys.path.insert(0, str(Path(__file__).parent))
    import tools_onnx_fixtures
    return tools_onnx_fixtures


@needs_ort
def test_upscale(tmp_path, cache):
    fx = _fx()
    put_weight(cache, "upscale", "model", fx.upscale_x4)
    rng = np.random.default_rng(1)
    p = tmp_path / "n.png"
    Image.fromarray(rng.integers(0, 255, (70, 90, 3), dtype=np.uint8)).save(p)
    full = run_tool("upscale", {"image": str(p)}, {"tile": 0}).outputs[0].image
    tiled = run_tool("upscale", {"image": str(p)}, {"tile": 32}).outputs[0].image
    assert full.size == (360, 280)
    assert np.array_equal(np.asarray(full), np.asarray(tiled))  # seams removed exactly
    x2 = run_tool("upscale", {"image": str(p)}, {"scale": 2}).outputs[0].image
    assert x2.size == (180, 140)
    rgba = img(tmp_path, (20, 20), (1, 2, 3, 100), mode="RGBA", name="a.png")
    out = run_tool("upscale", {"image": str(rgba)}, {}).outputs[0].image
    assert out.mode == "RGBA" and out.getpixel((5, 5))[3] == 100


@needs_ort
@pytest.mark.parametrize("model,size,n_out", [("u2netp", 320, 2), ("isnet", 1024, 1), ("birefnet-lite", 1024, 3)])
def test_remove_background(tmp_path, cache, model, size, n_out):
    fx = _fx()
    put_weight(cache, "remove-background", "model", fx.saliency, size, n_outputs=n_out, params={"model": model})
    im = Image.new("RGB", (80, 60), (0, 0, 0))
    im.paste((255, 255, 255), (40, 0, 80, 60))                 # bright right half = "subject"
    p = tmp_path / "s.png"
    im.save(p)
    out = run_tool("remove-background", {"image": str(p)}, {"model": model}).outputs[0].image
    assert out.mode == "RGBA" and out.size == (80, 60)
    a = np.asarray(out.getchannel("A"))
    assert a[:, 70].mean() > a[:, 10].mean() + 100
    m = run_tool("remove-background", {"image": str(p)}, {"model": model, "output": "mask", "threshold": 0.5}).outputs[0].image
    assert m.mode == "L" and set(np.unique(np.asarray(m))) <= {0, 255}
    c = run_tool("remove-background", {"image": str(p)}, {"model": model, "output": "color", "background": "#00ff00"}).outputs[0].image
    assert c.mode == "RGB"


@needs_ort
def test_erase_keeps_outside_pixels(tmp_path, cache):
    fx = _fx()
    put_weight(cache, "erase-object", "model", fx.lama)
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    p, mp = tmp_path / "i.png", tmp_path / "m.png"
    Image.fromarray(arr).save(p)
    mask = Image.new("L", (400, 300), 0)
    mask.paste(255, (100, 100, 140, 130))
    mask.save(mp)
    out = np.asarray(run_tool("erase-object", {"image": str(p), "mask": str(mp)}, {"dilate": 4}).outputs[0].image)
    assert np.array_equal(out[:50], arr[:50])                   # far away: bit-exact
    assert out[110:120, 110:130].mean() > 200                   # fake LaMa paints white
    empty = tmp_path / "e.png"
    Image.new("L", (400, 300), 0).save(empty)
    with pytest.raises(ToolError) as e:
        run_tool("erase-object", {"image": str(p), "mask": str(empty)}, {})
    assert e.value.code == "invalid_input"


@needs_ort
def test_segment(tmp_path, cache):
    fx = _fx()
    put_weight(cache, "segment", "encoder", fx.sam_encoder)
    put_weight(cache, "segment", "decoder", fx.sam_decoder)
    p = img(tmp_path, (200, 100))
    r = run_tool("segment", {"image": str(p)}, {"points": [{"x": 20, "y": 50, "label": 1}]})
    m = np.asarray(r.outputs[0].image)
    assert r.outputs[0].type == "mask" and m.shape == (100, 200)
    assert m[:, :90].min() == 255 and m[:, 110:].max() == 0
    with pytest.raises(ToolError):
        run_tool("segment", {"image": str(p)}, {"points": [{"x": 500, "y": 5, "label": 1}]})


@needs_ort
def test_restore_faces_math_and_identity(tmp_path, cache):
    from imference_engine.tools.impl import faces

    # similarity() recovers a known transform; invert() round-trips.
    src = np.array(faces.TEMPLATE) * 0.5 + 10
    M = faces.similarity(src, faces.TEMPLATE)
    back = (np.c_[src, np.ones(5)] @ M.T)
    assert np.allclose(back, faces.TEMPLATE, atol=1e-6)
    Minv = faces.invert(M)
    assert np.allclose(np.c_[back, np.ones(5)] @ Minv.T, src, atol=1e-6)
    # identity "restorer" on a crop returns the crop
    fx = _fx()
    sess_path = fx.identity_face(tmp_path / "g.onnx")
    from imference_engine.tools.impl import onnx_util
    sess, _ = onnx_util.session(sess_path, "cpu")
    crop = Image.new("RGB", (512, 512), (120, 60, 30))
    out = np.asarray(faces.restore_face(sess, crop)).astype(int)
    assert np.abs(out - np.asarray(crop)).max() <= 1



@needs_ort
def test_yunet_static_input_is_fit_and_padded(tmp_path):
    """The official YuNet export takes a fixed 640x640 input: any image size
    must be scaled to fit, padded, and the detections mapped back."""
    from imference_engine.tools.impl import faces, onnx_util

    sess, _ = onnx_util.session(_fx().yunet_static(tmp_path / "y.onnx"), "cpu")
    for size, s in (((1280, 960), 0.5), ((300, 200), 1.0)):
        dets = faces.detect_yunet(sess, Image.new("RGB", size))
        assert len(dets) == 1
        x0, y0, x1, y1 = dets[0][1]
        assert np.allclose([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0], [80 / s, 48 / s, 32 / s])

@needs_ort
@needs_ffmpeg
def test_interpolate_doubles_frames(tmp_path, cache, clip):
    from imference_engine.tools.media import probe_video

    fx = _fx()
    put_weight(cache, "interpolate", "model", fx.rife_mean)
    r = run_tool("interpolate", {"video": str(clip)}, {"factor": 2}, output_dir=str(tmp_path / "o"))
    info = probe_video(str(r.outputs[0].path))
    assert abs(info["fps"] - 20) < 0.5
    assert 1.6 <= info["duration"] <= 2.2


# ---------------------------------------------------------- generation tools

class FakeEngine:
    """Engine stand-in: img2img returns the source shifted toward blue."""

    def __init__(self):
        self.calls = []

    def registered_models(self):
        return ["local"]

    def generate(self, **kw):
        from imference_engine.core.result import MediaResult
        self.calls.append(kw)
        src = kw["source_image"]
        out = Image.new("RGB", src.size, (0, 0, 255))
        return MediaResult(kind="image", media=[out], seeds=[kw["seed"]])


def test_vary_uses_strength_and_seed(tmp_path):
    eng = FakeEngine()
    p = img(tmp_path, (300, 200))
    r = run_tool("vary", {"image": str(p)}, {"model": "any", "strength": "strong", "seed": 7, "prompt": "cat"}, engine=eng)
    kw = eng.calls[0]
    assert kw["model"] == "local" and kw["strength"] == 0.65 and kw["seed"] == 7 and kw["prompt"] == "cat"
    assert kw["source_image"].size[0] % 64 == 0
    assert r.outputs[0].image.size == (300, 200) and r.outputs[0].data == {"seed": 7}


def test_inpaint_composites_only_mask(tmp_path):
    eng = FakeEngine()
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    p, mp = tmp_path / "i.png", tmp_path / "m.png"
    Image.fromarray(arr).save(p)
    mask = Image.new("L", (512, 512), 0)
    mask.paste(255, (200, 200, 260, 260))
    mask.save(mp)
    out = np.asarray(run_tool("inpaint", {"image": str(p), "mask": str(mp)},
                              {"model": "local", "prompt": "a flower", "feather": 8}, engine=eng).outputs[0].image)
    assert np.array_equal(out[:150], arr[:150])
    assert out[220:240, 220:240, 2].mean() > 240
    # prefill removed the original object from what the model saw
    assert eng.calls[0]["strength"] == 0.9


def test_outpaint_canvas(tmp_path):
    eng = FakeEngine()
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 255, (100, 120, 3), dtype=np.uint8)
    p = tmp_path / "i.png"
    Image.fromarray(arr).save(p)
    r = run_tool("outpaint", {"image": str(p)}, {"model": "local", "left": 40, "bottom": 20}, engine=eng)
    out = np.asarray(r.outputs[0].image)
    assert out.shape == (120, 160, 3)
    assert np.array_equal(out[10:80, 60:150], arr[10:80, 20:110])   # interior kept
    assert out[110, 10, 2] > 200                                    # new area from the model
    with pytest.raises(ToolError):
        run_tool("outpaint", {"image": str(p)}, {"model": "local"}, engine=eng)


def test_engine_error_is_typed(tmp_path):
    from imference_engine.core.result import GenerationError, MediaResult

    class OOMEngine(FakeEngine):
        def generate(self, **kw):
            return MediaResult(kind="image", media=[None], seeds=[1],
                               errors=[GenerationError("CUDA out of memory", 1, 0)])

    with pytest.raises(ToolError) as e:
        run_tool("vary", {"image": str(img(tmp_path))}, {"model": "local"}, engine=OOMEngine())
    assert e.value.code == "oom"


# ---------------------------------------------------------------------- CLI

def test_cli_list_and_run(tmp_path):
    from imference_engine.tools.__main__ import main

    p = img(tmp_path, (50, 50))
    assert main(["list"]) == 0
    assert main(["run", "resize", "-i", f"image={p}", "-p", "width=10", "-o", str(tmp_path / "o")]) == 0
    assert main(["run", "resize", "-i", f"image={p}", "-p", "nope=1"]) == 2


def test_probe_without_ffprobe(tmp_path, clip, monkeypatch):
    """imageio-ffmpeg ships ffmpeg only: probing falls back to `ffmpeg -i`."""
    from imference_engine.tools import media

    monkeypatch.setattr(media, "ffprobe_bin", lambda: None)
    info = media.probe_video(str(clip))
    assert (info["width"], info["height"]) == (96, 64)
    assert abs(info["fps"] - 10) < 0.5 and 1.8 <= info["duration"] <= 2.2
    with pytest.raises(ToolError):
        media.probe_video(str(img(tmp_path)))


class FakeEditEngine(FakeEngine):
    def __init__(self, backend="qwenimage-edit"):
        super().__init__()
        self.backend = backend

    def registered_models(self):
        return ["qwen-image-edit-2511"]

    def model_backend(self, name):
        return self.backend if name == "qwen-image-edit-2511" else None

    def generate(self, **kw):
        from imference_engine.core.result import MediaResult
        self.calls.append(kw)
        return MediaResult(kind="image", media=[Image.new("RGB", (1024, 768), (0, 255, 0))], seeds=[kw["seed"]])


def test_instruct_edit(tmp_path):
    eng = FakeEditEngine()
    p = img(tmp_path, (640, 480))
    ref = img(tmp_path, (300, 300), name="ref.png")
    r = run_tool("instruct-edit", {"image": str(p), "references": [str(ref)]},
                 {"model": "qwen-image-edit-2511", "instruction": "make it green", "steps": 8, "guidance": 1.0, "seed": 3},
                 engine=eng)
    kw = eng.calls[0]
    assert kw["prompt"] == "make it green" and kw["num_steps"] == 8 and kw["guidance_scale"] == 1.0
    assert isinstance(kw["source_image"], list) and len(kw["source_image"]) == 2
    assert r.outputs[0].image.size == (640, 480) and r.outputs[0].data == {"seed": 3}
    r = run_tool("instruct-edit", {"image": str(p)}, {"model": "qwen-image-edit-2511", "instruction": "x",
                                                     "keep_size": False}, engine=eng)
    assert r.outputs[0].image.size == (1024, 768)
    assert not isinstance(eng.calls[-1]["source_image"], list)
    assert eng.calls[-1]["num_steps"] is None                    # model recipe applies


def test_instruct_edit_rejects_non_edit_model_and_vary_rejects_edit_model(tmp_path):
    p = img(tmp_path)
    with pytest.raises(ToolError) as e:
        run_tool("instruct-edit", {"image": str(p)}, {"model": "qwen-image-edit-2511", "instruction": "x"},
                 engine=FakeEditEngine(backend="sdxl"))
    assert e.value.code == "unsupported_runtime"
    with pytest.raises(ToolError) as e:
        run_tool("vary", {"image": str(p)}, {"model": "qwen-image-edit-2511"}, engine=FakeEditEngine())
    assert e.value.code == "unsupported_runtime"
    with pytest.raises(ToolError) as e:
        run_tool("instruct-edit", {"image": str(p)}, {"model": "m"}, engine=FakeEditEngine())
    assert e.value.field == "instruction"
