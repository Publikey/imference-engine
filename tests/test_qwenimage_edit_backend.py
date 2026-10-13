"""No-torch assertions on the Qwen-Image-Edit backend."""
from __future__ import annotations

import pytest

from imference_engine.qwenimage import QwenImageBackend, QwenImageEditBackend


def test_engine_key_and_family():
    assert QwenImageEditBackend.engine == "qwenimage-edit"
    assert issubclass(QwenImageEditBackend, QwenImageBackend)
    assert QwenImageEditBackend.instruction_edit


def test_defaults_follow_2511_recipe():
    d = QwenImageEditBackend().engine_defaults()
    assert d.num_steps == 40 and d.guidance_scale == 4.0


def test_base_patterns_add_processor_and_skip_transformer_weights():
    pats = QwenImageEditBackend.BASE_PATTERNS
    assert "processor/*" in pats and "text_encoder/*" in pats and "vae/*" in pats
    assert "transformer/config.json" in pats and "transformer/*" not in pats


def test_resident_pipe_is_the_edit_pipe():
    pipe = object()
    assert QwenImageEditBackend().make_img2img(pipe) is pipe


def test_inference_kwargs_condition_on_images_without_strength():
    be = QwenImageEditBackend()
    kw = be.build_inference_kwargs(width=1024, height=1024, num_steps=40, guidance_scale=4.0, clip_skip=None,
                                   chunk_size=1, generator=None, image="IMG", strength=0.3)
    assert kw["image"] == ["IMG"] and kw["true_cfg_scale"] == 4.0
    assert "strength" not in kw and "width" not in kw
    kw = be.build_inference_kwargs(width=1, height=1, num_steps=8, guidance_scale=1.0, clip_skip=None,
                                   chunk_size=1, generator=None, image=["A", "B"])
    assert kw["image"] == ["A", "B"]
    with pytest.raises(ValueError):
        be.build_inference_kwargs(width=1, height=1, num_steps=8, guidance_scale=1.0, clip_skip=None,
                                  chunk_size=1, generator=None)


def test_registered_and_model_backend():
    from imference_engine import Engine, RuntimeConfig
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends.get("qwenimage-edit"), QwenImageEditBackend)
    engine.register_model("edit", backend="qwenimage-edit", weights_path="/x.safetensors")
    assert engine.model_backend("edit") == "qwenimage-edit"
    assert engine.model_backend("nope") is None
