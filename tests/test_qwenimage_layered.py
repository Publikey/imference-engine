"""Qwen-Image-Layered backend: prompt handling, call options, layer output
through the engine (``MediaResult.layers``) — no torch, fake pipe."""
from __future__ import annotations

import pytest
from PIL import Image

from imference_engine.qwenimage import QwenImageLayeredBackend
from imference_engine.qwenimage.layered_backend import compose_layers


def _layer(color, box, size=(64, 64)):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            im.putpixel((x, y), color)
    return im


def test_engine_key_and_defaults():
    be = QwenImageLayeredBackend()
    assert be.engine == "qwenimage-layered"
    assert be.instruction_edit is True
    d = be.engine_defaults()
    assert d.num_steps == 50 and d.guidance_scale == 4.0
    # no Lightning recipe for this checkpoint, GGUF or not
    assert be.weights_defaults("/w/qwen-image-layered-Q4_K_M.gguf") is None
    assert be.weights_defaults("auto") is None


def test_base_patterns_carry_the_processor_and_never_the_transformer_weights():
    for pats in (QwenImageLayeredBackend.BASE_PATTERNS, QwenImageLayeredBackend.LITE_BASE_PATTERNS):
        assert "processor/*" in pats and "vae/*" in pats
        assert "transformer/config.json" in pats and "transformer/*" not in pats
    assert "text_encoder/*" in QwenImageLayeredBackend.BASE_PATTERNS
    assert "text_encoder/*" not in QwenImageLayeredBackend.LITE_BASE_PATTERNS


def test_empty_prompt_lets_the_pipeline_caption_the_image():
    be = QwenImageLayeredBackend()
    assert be.encode_prompts(None, "", None) == {"negative_prompt": " "}
    assert be.encode_prompts(None, "  ", "blurry") == {"negative_prompt": "blurry"}
    assert be.encode_prompts(None, "a house on a hill", None) == {
        "prompt": "a house on a hill", "negative_prompt": " "}


def test_inference_kwargs_and_per_request_options():
    class _Pipe:
        scheduler = object()

    be, pipe = QwenImageLayeredBackend(), _Pipe()
    src = Image.new("RGB", (32, 32), "white")
    be.apply_scheduler(pipe, None, layers=3, resolution=1024)
    kw = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=50, guidance_scale=4.0,
        clip_skip=None, chunk_size=1, generator=None, image=src)
    assert kw["layers"] == 3 and kw["resolution"] == 1024
    assert kw["image"].mode == "RGBA"
    assert kw["true_cfg_scale"] == 4.0 and kw["cfg_normalize"] is True
    assert "width" not in kw and "strength" not in kw

    be.apply_scheduler(pipe, None)           # next request: defaults again
    kw = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=50, guidance_scale=4.0,
        clip_skip=None, chunk_size=1, generator=None, image=src)
    assert kw["layers"] == be.DEFAULT_LAYERS and kw["resolution"] == be.DEFAULT_RESOLUTION

    with pytest.raises(ValueError):
        be.build_inference_kwargs(
            width=1024, height=1024, num_steps=50, guidance_scale=4.0,
            clip_skip=None, chunk_size=1, generator=None)


def test_compose_and_split_output():
    red = _layer((255, 0, 0, 255), (0, 0, 64, 64))
    blue = _layer((0, 0, 255, 255), (16, 16, 48, 48))
    out = compose_layers([red, blue])
    assert out.mode == "RGB"
    assert out.getpixel((2, 2)) == (255, 0, 0) and out.getpixel((32, 32)) == (0, 0, 255)

    images, layers = QwenImageLayeredBackend().split_output([[red, blue]])
    assert len(images) == 1 and images[0].size == (64, 64)
    assert layers == [[red, blue]]


def test_default_backends_have_no_layers():
    from imference_engine.zimage import ZImageBackend
    imgs = [object(), object()]
    assert ZImageBackend().split_output(imgs) == (imgs, None)


def test_engine_returns_layers_next_to_the_recomposed_image():
    """The chunk loop keeps one image per seed in ``media`` and the RGBA layers
    index-aligned in ``MediaResult.layers``."""
    from imference_engine import Engine, RuntimeConfig

    red = _layer((255, 0, 0, 255), (0, 0, 64, 64))
    blue = _layer((0, 0, 255, 255), (16, 16, 48, 48))

    class _Out:
        images = [[red, blue]]

    class _Pipe:
        scheduler = object()

        def __call__(self, **kwargs):
            assert kwargs["layers"] == 2
            return _Out()

    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    backend = engine._backends["qwenimage-layered"]
    backend.apply_scheduler(_Pipe(), None, layers=2)
    result = engine._run_chunked(
        pipe=_Pipe(), backend=backend, prompt_kwargs={"negative_prompt": " "},
        width=1024, height=1024, num_steps=2, guidance_scale=4.0, clip_skip=None,
        seeds=[7], is_img2img=True, source_image=Image.new("RGB", (64, 64), "white"))
    assert result.ok and len(result.images) == 1
    assert result.images[0].mode == "RGB"
    assert result.layers == [[red, blue]]


def test_other_backends_leave_layers_unset():
    from imference_engine import Engine, RuntimeConfig

    img = Image.new("RGB", (64, 64))
    for x in range(64):
        img.putpixel((x, x), (255, 255, 255))

    class _Out:
        images = [img]

    class _Pipe:
        def __call__(self, **kwargs):
            return _Out()

    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    result = engine._run_chunked(
        pipe=_Pipe(), backend=engine._backends["zimage"], prompt_kwargs={"prompt": "x"},
        width=64, height=64, num_steps=1, guidance_scale=1.0, clip_skip=None,
        seeds=[1], is_img2img=False)
    assert result.ok and result.layers is None
