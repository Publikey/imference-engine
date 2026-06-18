"""img2img wiring tests for the generic Engine path (no torch / GPU).

Covers two things:
  1. Each backend's build_inference_kwargs switches to img2img shape when an
     image is passed (image + strength, no width/height).
  2. Engine.generate(source_image=...) wraps the resident t2i pipe via
     backend.make_img2img and threads image + strength down to the pipe call;
     the t2i path does neither.
"""
from __future__ import annotations

from imference_engine.pipelines.sdxl import SDXLBackend
from imference_engine.zimage import ZImageBackend


def _common():
    return dict(width=1024, height=768, num_steps=8, guidance_scale=5.0,
                clip_skip=None, chunk_size=2, generator=None)


def test_sdxl_build_kwargs_switches_to_img2img():
    be = SDXLBackend()
    t2i = be.build_inference_kwargs(**_common())
    assert t2i["width"] == 1024 and t2i["height"] == 768
    assert "image" not in t2i and "strength" not in t2i

    sentinel = object()
    i2i = be.build_inference_kwargs(image=sentinel, strength=0.4, **_common())
    assert i2i["image"] is sentinel and i2i["strength"] == 0.4
    # img2img takes its size from the source image — no width/height passed.
    assert "width" not in i2i and "height" not in i2i


def test_zimage_build_kwargs_switches_to_img2img():
    be = ZImageBackend()
    t2i = be.build_inference_kwargs(**_common())
    assert t2i["width"] == 1024 and "image" not in t2i

    sentinel = object()
    i2i = be.build_inference_kwargs(image=sentinel, strength=0.6, **_common())
    assert i2i["image"] is sentinel and i2i["strength"] == 0.6
    assert "width" not in i2i and "height" not in i2i


# --- Engine-level: make_img2img invoked + image/strength threaded -----------

from imference_engine import Engine, RuntimeConfig  # noqa: E402
from imference_engine.pipelines.base import PipelineBackend  # noqa: E402


class _FakePipe:
    def __init__(self, name="t2i"):
        self.name = name
        self.vae = None

    def to(self, device):  # ModelManager moves it around
        return self

    def __call__(self, **kwargs):
        self.seen = kwargs

        class _Out:
            images = [object()]
        return _Out()


class _RecordingBackend(PipelineBackend):
    """Fake backend that records whether make_img2img ran and what kwargs the
    pipe was called with — the oracle for the engine's img2img threading."""

    engine = "fake"

    def __init__(self):
        self.made_img2img = False
        self.t2i = _FakePipe("t2i")
        self.i2i = _FakePipe("i2i")

    def load_pipeline(self, *, local_path, base_model=None):
        return self.t2i

    def make_img2img(self, t2i_pipe):
        self.made_img2img = True
        return self.i2i

    def get_compute_module(self, pipe):
        return None

    def encode_prompts(self, pipe, prompt, negative_prompt):
        return {"prompt": prompt}

    def apply_scheduler(self, pipe, scheduler, **kwargs):
        pass

    def build_inference_kwargs(self, *, width, height, num_steps, guidance_scale,
                               clip_skip, chunk_size, generator,
                               image=None, strength=0.75):
        kw = {"num_inference_steps": num_steps, "num_images_per_prompt": chunk_size}
        if image is not None:
            kw["image"] = image
            kw["strength"] = strength
        else:
            kw["width"] = width
            kw["height"] = height
        return kw

    def make_generator(self, seed, device):
        return None


def _engine_with_fake():
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    fake = _RecordingBackend()
    engine._backends["fake"] = fake
    engine._models._backends["fake"] = fake
    engine.register_model("m", backend="fake", weights_path="m")
    return engine, fake


def test_engine_img2img_wraps_and_threads():
    engine, fake = _engine_with_fake()
    img = object()
    res = engine.generate(model="m", prompt="x", source_image=img, strength=0.35,
                          width=512, height=512, batch=1)
    assert res.errors == [] and res.images and res.images[0] is not None
    assert fake.made_img2img is True
    # The img2img pipe (not the t2i one) ran, with image + strength, no width/height.
    assert fake.i2i.seen["image"] is img
    assert fake.i2i.seen["strength"] == 0.35
    assert "width" not in fake.i2i.seen
    assert not hasattr(fake.t2i, "seen")  # t2i pipe never called


def test_engine_t2i_does_not_wrap():
    engine, fake = _engine_with_fake()
    res = engine.generate(model="m", prompt="x", width=512, height=512, batch=1)
    assert res.errors == [] and res.images and res.images[0] is not None
    assert fake.made_img2img is False
    assert fake.t2i.seen["width"] == 512
    assert "image" not in fake.t2i.seen
