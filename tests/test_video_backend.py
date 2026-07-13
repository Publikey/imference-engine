"""No-torch tests for the VideoBackend ABC + WanBackend (Phase 4b).

WanBackend wraps wan/loader.py; here we check the identity, the pipe(**call)
kwargs it builds, the loader delegation (monkeypatched), and teardown — the
contract a 2nd architecture (LTX) will implement.
"""
from __future__ import annotations

from imference_engine.video import VideoBackend, VideoBuildContext, WanBackend


def _ctx(**over):
    base = dict(quant="Q8_0", device="cuda:0", enable_offload=True, vae_tiling=True,
                cache_dir="/cache", cdn_base="https://cdn.x/wan", text_encoder_quant="int8")
    base.update(over)
    return VideoBuildContext(**base)


def test_wan_backend_identity():
    assert WanBackend.engine == "wan"
    assert isinstance(WanBackend(), VideoBackend)


def test_build_call_t2v_no_image():
    call = WanBackend().build_call(
        prompt="a fox", negative_prompt=None, width=832, height=480,
        num_frames=81, num_steps=4, guidance_scale=1.0, guidance_scale_2=None,
        image=None, generator="G")
    assert call["prompt"] == "a fox"
    assert call["negative_prompt"] == ""            # None -> ""
    assert call["width"] == 832 and call["height"] == 480
    assert call["num_frames"] == 81 and call["num_inference_steps"] == 4
    assert call["guidance_scale_2"] == 1.0          # falls back to guidance_scale
    assert call["output_type"] == "pil"
    assert "image" not in call


def test_build_call_dual_guidance_and_i2v():
    class FakeImg:
        def __init__(self, w, h):
            self.width, self.height = w, h
        def convert(self, mode):
            assert mode == "RGB"
            return self
        def resize(self, size):
            self.size = size
            return self

    # Square image, square budget -> aspect preserved, dims aligned to 16.
    img = FakeImg(1024, 1024)
    call = WanBackend().build_call(
        prompt="p", negative_prompt="bad", width=640, height=640,
        num_frames=49, num_steps=6, guidance_scale=3.0, guidance_scale_2=1.5,
        image=img, generator="G")
    assert call["guidance_scale"] == 3.0 and call["guidance_scale_2"] == 1.5
    assert call["negative_prompt"] == "bad"
    # square in a 640*640 area budget -> 640x640 (already 16-aligned), no squash
    assert call["image"] is img and img.size == (640, 640)
    assert call["width"] == 640 and call["height"] == 640

    # Landscape 16:9 image in the same area budget: aspect preserved, not forced
    # to 640x640; both dims multiples of 16; area ~ budget.
    wide = FakeImg(1920, 1080)
    c2 = WanBackend().build_call(
        prompt="p", negative_prompt=None, width=640, height=640,
        num_frames=49, num_steps=6, guidance_scale=1.0, guidance_scale_2=None,
        image=wide, generator="G")
    w, h = c2["width"], c2["height"]
    assert w % 16 == 0 and h % 16 == 0
    assert w > h  # landscape preserved
    assert c2["image"].size == (w, h)


def test_load_shared_and_build_delegate_to_loader(monkeypatch):
    from imference_engine.wan import loader as L
    seen = {}
    monkeypatch.setattr(L, "load_shared_components",
                        lambda repo, cache_dir=None, text_encoder_quant="none", cdn_base=None:
                        seen.update(shared_repo=repo, shared_cdn=cdn_base, teq=text_encoder_quant)
                        or "SHARED")
    monkeypatch.setattr(L, "build_pipeline",
                        lambda spec, **kw: seen.update(build_kw=kw, spec=spec) or "PIPE")

    be = WanBackend()
    ctx = _ctx()
    assert be.load_shared(ctx) == "SHARED"
    assert seen["shared_repo"] == WanBackend.SHARED_BASE_REPO
    assert seen["shared_cdn"] == "https://cdn.x/wan" and seen["teq"] == "int8"

    assert be.build("VAR", "SHARED", ctx) == "PIPE"
    kw = seen["build_kw"]
    assert kw["quant"] == "Q8_0" and kw["device"] == "cuda:0"
    assert kw["enable_offload"] is True and kw["vae_tiling"] is True
    assert kw["cache_dir"] == "/cache" and kw["cdn_base"] == "https://cdn.x/wan"
    assert seen["spec"] == "VAR"


def test_teardown_drops_moe_and_hooks():
    class FakePipe:
        def __init__(self):
            self.transformer = object()
            self.transformer_2 = object()
            self.hooks_removed = False
        def remove_all_hooks(self):
            self.hooks_removed = True
        def maybe_free_model_hooks(self):
            pass

    p = FakePipe()
    WanBackend().teardown(p)
    assert p.hooks_removed is True
    assert p.transformer is None and p.transformer_2 is None
