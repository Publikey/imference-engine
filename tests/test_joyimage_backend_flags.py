"""No-torch assertions on the JoyAI-Image-Edit-Plus backend: defaults, reference
handling, the one-image-per-call cap through the engine's chunk loop, the build
table and registration."""
from __future__ import annotations

import pytest
from PIL import Image

from imference_engine.joyimage import AUTO, BUILDS, JoyImageEditBackend, recommend_build


def _img(color="white"):
    im = Image.new("RGB", (64, 64), color)
    for x in range(64):
        im.putpixel((x, x), (0, 0, 0))        # not a flat image
    return im


def test_engine_key_and_defaults():
    be = JoyImageEditBackend()
    assert be.engine == "joyimage-edit"
    assert be.instruction_edit is True and be.max_batch_per_call == 1
    d = be.engine_defaults()
    assert d.num_steps == 30 and d.guidance_scale == 4.0
    assert be.resolve_weights(AUTO) == "jdopensource/JoyAI-Image-Edit-Plus-Diffusers"


def test_references_one_to_five():
    be = JoyImageEditBackend()
    one = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=30, guidance_scale=4.0,
        clip_skip=None, chunk_size=1, generator=None, image=_img())
    assert len(one["images"]) == 1 and one["images"][0].mode == "RGB"
    assert "num_images_per_prompt" not in one       # the pipeline has no such argument
    assert "width" not in one and "strength" not in one
    three = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=30, guidance_scale=4.0,
        clip_skip=None, chunk_size=1, generator=None, image=[_img(), _img(), _img()])
    assert len(three["images"]) == 3
    with pytest.raises(ValueError, match="at most 5"):
        be.build_inference_kwargs(
            width=1024, height=1024, num_steps=30, guidance_scale=4.0,
            clip_skip=None, chunk_size=1, generator=None, image=[_img()] * 6)
    with pytest.raises(ValueError, match="at least one input image"):
        be.build_inference_kwargs(
            width=1024, height=1024, num_steps=30, guidance_scale=4.0,
            clip_skip=None, chunk_size=1, generator=None)


def test_default_backends_have_no_per_call_cap():
    from imference_engine.zimage import ZImageBackend
    assert ZImageBackend.max_batch_per_call is None


def test_engine_renders_a_batch_one_image_per_call():
    """Even when the batch sizer would allow more, a backend capped at one image
    per call gets one call per seed — and every seed still gets its image."""
    from imference_engine import Engine, RuntimeConfig

    calls = []

    class _Out:
        def __init__(self):
            self.images = [_img()]

    class _Pipe:
        def __call__(self, **kwargs):
            calls.append(kwargs["generator"])
            return _Out()

    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    backend = engine._backends["joyimage-edit"]
    # as if profiling had concluded 4 images fit at once
    engine._batch_sizer._cache[(backend.engine, True)] = 4
    result = engine._run_chunked(
        pipe=_Pipe(), backend=backend, prompt_kwargs={"prompt": "x"},
        width=1024, height=1024, num_steps=2, guidance_scale=4.0, clip_skip=None,
        seeds=[1, 2, 3], is_img2img=True, source_image=_img())
    assert result.ok and len(result.images) == 3 and all(i is not None for i in result.images)
    assert len(calls) == 3


def test_build_table():
    floors = [b.min_vram_gb for b in BUILDS]
    assert floors == sorted(floors, reverse=True) and BUILDS[-1].min_vram_gb == 0.0
    for b in BUILDS:
        assert b.offload in ("model", "group") and b.transformer == "fp8"   # never resident
    assert recommend_build(24.0).offload == "model"
    assert recommend_build(16.0).offload == "group"
    none = recommend_build(None)
    assert (none.transformer, none.offload) == ("bf16", "model")


def test_registered_as_a_backend(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text("models:\n  - name: joy\n    engine: joyimage-edit\n    weights: auto\n",
                 encoding="utf-8")
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends["joyimage-edit"], JoyImageEditBackend)
