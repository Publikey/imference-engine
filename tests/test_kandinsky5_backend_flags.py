"""No-torch assertions on the Kandinsky 5 Image Lite backends: defaults, the
shared-base / per-checkpoint split, inference kwargs, the `.images` output
shim, the storage build table and registration."""
from __future__ import annotations

import types

import pytest

from imference_engine.kandinsky5 import (
    AUTO,
    BUILDS,
    Kandinsky5Backend,
    Kandinsky5EditBackend,
    recommend_build,
)
from imference_engine.kandinsky5 import backend as mod


def test_engine_keys_and_defaults():
    t2i, edit = Kandinsky5Backend(), Kandinsky5EditBackend()
    assert t2i.engine == "kandinsky5" and edit.engine == "kandinsky5-edit"
    for be in (t2i, edit):
        d = be.engine_defaults()
        assert d.num_steps == 50 and d.guidance_scale == 3.5
    assert edit.instruction_edit is True and not getattr(t2i, "instruction_edit", False)
    assert t2i.resolve_weights(AUTO) == "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers"
    assert edit.resolve_weights(AUTO) == "kandinskylab/Kandinsky-5.0-I2I-Lite-sft-Diffusers"
    assert t2i.supports_loras is False


def test_the_encoders_are_mirrored_once_and_clip_only_as_safetensors():
    shared, ckpt = Kandinsky5Backend.SHARED_PATTERNS, Kandinsky5Backend.CHECKPOINT_PATTERNS
    assert "text_encoder/*" in shared and "vae/*" in shared and "tokenizer_2/*" in shared
    # the repo also ships CLIP as flax / tf / pickle: only config + safetensors
    assert "text_encoder_2/*" not in shared
    assert "text_encoder_2/model.safetensors" in shared and "text_encoder_2/config.json" in shared
    assert "text_encoder/*" not in ckpt and "transformer/*" in ckpt
    assert set(Kandinsky5Backend.BASE_PATTERNS) == set(shared) | set(ckpt)


def test_t2i_kwargs_and_prompts():
    be = Kandinsky5Backend()
    kw = be.build_inference_kwargs(
        width=1280, height=768, num_steps=50, guidance_scale=3.5,
        clip_skip=None, chunk_size=2, generator=None)
    assert kw == {"num_inference_steps": 50, "guidance_scale": 3.5, "width": 1280,
                  "height": 768, "generator": None, "num_images_per_prompt": 2}
    assert be.encode_prompts(None, "a fox", None) == {"prompt": "a fox"}
    assert be.encode_prompts(None, "a fox", "blurry") == {"prompt": "a fox", "negative_prompt": "blurry"}
    with pytest.raises(NotImplementedError):
        be.make_img2img(object())


def test_edit_kwargs_need_an_image():
    be = Kandinsky5EditBackend()
    src = object()
    kw = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=50, guidance_scale=3.5,
        clip_skip=None, chunk_size=1, generator=None, image=src, strength=0.4)
    assert kw["image"] is src and "width" not in kw and "strength" not in kw
    with pytest.raises(ValueError):
        be.build_inference_kwargs(
            width=1024, height=1024, num_steps=50, guidance_scale=3.5,
            clip_skip=None, chunk_size=1, generator=None)
    pipe = object()
    assert be.make_img2img(pipe) is pipe


def test_output_shim_exposes_images():
    """The Kandinsky 5 pipelines return `.image`; the engine reads `.images`."""

    class _Base:
        def __init__(self, out):
            self._out = out

        def __call__(self, *args, **kwargs):
            return self._out

    cls = mod._with_images_output(_Base)
    assert cls.__name__ == "_Base"
    assert cls(types.SimpleNamespace(image=["a", "b"]))().images == ["a", "b"]
    assert cls(types.SimpleNamespace(image="a"))().images == ["a"]
    already = types.SimpleNamespace(images=["x"])
    assert cls(already)() is already


def test_build_table():
    floors = [b.min_vram_gb for b in BUILDS]
    assert floors == sorted(floors, reverse=True) and BUILDS[-1].min_vram_gb == 0.0
    for b in BUILDS:
        assert b.filename == "" and b.transformer in ("bf16", "fp8")
    none = recommend_build(None)
    assert (none.transformer, none.encoder, none.offload) == ("bf16", "bf16", "model")
    assert recommend_build(48.0) is BUILDS[0] and recommend_build(1.0) is BUILDS[-1]


def test_registered_as_backends(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text(
        "models:\n"
        "  - name: k5\n    engine: kandinsky5\n    weights: auto\n"
        "  - name: k5-edit\n    engine: kandinsky5-edit\n    weights: auto\n",
        encoding="utf-8")
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends["kandinsky5"], Kandinsky5Backend)
    assert isinstance(engine._backends["kandinsky5-edit"], Kandinsky5EditBackend)
