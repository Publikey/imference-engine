"""No-torch assertions on the LongCat-Image backends: defaults per checkpoint,
the split between shared base and per-checkpoint files, inference kwargs,
storage resolution, the build table and registration."""
from __future__ import annotations

import pytest

from imference_engine.longcat import (
    AUTO,
    BUILDS,
    LongCatBackend,
    LongCatEditBackend,
    recommend_build,
)
from imference_engine.longcat import backend as mod


def test_engine_keys_and_defaults():
    t2i, edit = LongCatBackend(), LongCatEditBackend()
    assert t2i.engine == "longcat" and edit.engine == "longcat-edit"
    d = t2i.engine_defaults()
    assert d.num_steps == 50 and d.guidance_scale == 4.0
    d = edit.engine_defaults()                      # Edit-Turbo, the default checkpoint
    assert d.num_steps == 8 and d.guidance_scale == 1.0
    assert edit.instruction_edit is True and not getattr(t2i, "instruction_edit", False)


def test_auto_is_the_default_checkpoint_of_each_backend():
    assert LongCatBackend().resolve_weights(AUTO) == "meituan-longcat/LongCat-Image"
    assert LongCatEditBackend().resolve_weights(AUTO) == "meituan-longcat/LongCat-Image-Edit-Turbo"
    assert LongCatBackend().resolve_weights("/w/ft.safetensors") == "/w/ft.safetensors"


def test_undistilled_edit_checkpoint_gets_its_own_recipe():
    edit = LongCatEditBackend()
    assert edit.weights_defaults(AUTO) is None
    assert edit.weights_defaults("meituan-longcat/LongCat-Image-Edit-Turbo") is None
    d = edit.weights_defaults("meituan-longcat/LongCat-Image-Edit")
    assert d.num_steps == 50 and d.guidance_scale == 4.5


def test_the_encoder_is_mirrored_once():
    """A checkpoint contributes only index / scheduler / transformer; the 16.6 GB
    encoder, the VAE, tokenizer and processor come from the base repo."""
    shared, ckpt = LongCatBackend.SHARED_PATTERNS, LongCatBackend.CHECKPOINT_PATTERNS
    assert "text_encoder/*" in shared and "vae/*" in shared and "text_processor/*" in shared
    assert "text_encoder/*" not in ckpt and "vae/*" not in ckpt
    assert "transformer/*" in ckpt and "scheduler/*" in ckpt and "model_index.json" in ckpt
    assert "transformer/*" not in LongCatBackend.CONFIG_PATTERNS
    assert set(LongCatBackend.BASE_PATTERNS) == set(shared) | set(ckpt)


def test_t2i_kwargs_and_prompt_rewrite_is_opt_in():
    be = LongCatBackend()
    be.apply_scheduler(None, "euler_a")
    kw = be.build_inference_kwargs(
        width=1344, height=768, num_steps=50, guidance_scale=4.0,
        clip_skip=None, chunk_size=2, generator=None)
    assert kw["enable_prompt_rewrite"] is False          # upstream default is True
    assert kw["enable_cfg_renorm"] is True
    assert kw["width"] == 1344 and kw["height"] == 768 and kw["num_images_per_prompt"] == 2

    be.apply_scheduler(None, None, prompt_rewrite=True)
    kw = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=50, guidance_scale=4.0,
        clip_skip=None, chunk_size=1, generator=None)
    assert kw["enable_prompt_rewrite"] is True
    be.apply_scheduler(None, None)                       # does not leak into the next request
    kw = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=50, guidance_scale=4.0,
        clip_skip=None, chunk_size=1, generator=None)
    assert kw["enable_prompt_rewrite"] is False

    assert be.encode_prompts(None, "a fox", None) == {"prompt": "a fox", "negative_prompt": ""}
    with pytest.raises(NotImplementedError):
        be.make_img2img(object())


def test_edit_kwargs_need_an_image_and_follow_its_size():
    be = LongCatEditBackend()
    src = object()
    kw = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=8, guidance_scale=1.0,
        clip_skip=None, chunk_size=1, generator=None, image=src, strength=0.5)
    assert kw["image"] is src
    assert "width" not in kw and "strength" not in kw and "enable_prompt_rewrite" not in kw
    with pytest.raises(ValueError):
        be.build_inference_kwargs(
            width=1024, height=1024, num_steps=8, guidance_scale=1.0,
            clip_skip=None, chunk_size=1, generator=None)
    pipe = object()
    assert be.make_img2img(pipe) is pipe


def test_storage_resolution(monkeypatch):
    monkeypatch.delenv("LONGCAT_TRANSFORMER", raising=False)
    monkeypatch.setattr(mod, "_vram_gb", lambda: 24.0)
    assert LongCatBackend._resolve_storage("LONGCAT_TRANSFORMER", "fp8") == "fp8"
    assert LongCatBackend._resolve_storage("LONGCAT_TRANSFORMER", "") == "bf16"
    monkeypatch.setenv("LONGCAT_TRANSFORMER", "bf16")
    assert LongCatBackend._resolve_storage("LONGCAT_TRANSFORMER", "fp8") == "bf16"
    monkeypatch.setenv("LONGCAT_TRANSFORMER", "fp8")
    monkeypatch.setattr(mod, "_vram_gb", lambda: None)   # float8 needs CUDA
    assert LongCatBackend._resolve_storage("LONGCAT_TRANSFORMER", "fp8") == "bf16"


def test_build_table_is_ordered_and_has_a_fallback():
    floors = [b.min_vram_gb for b in BUILDS]
    assert floors == sorted(floors, reverse=True) and BUILDS[-1].min_vram_gb == 0.0
    for b in BUILDS:
        assert b.transformer in ("bf16", "fp8") and b.encoder in ("bf16", "fp8")
        assert b.filename == ""                          # a storage choice, not a file
    none = recommend_build(None)
    assert (none.transformer, none.encoder, none.offload) == ("bf16", "bf16", "model")
    assert recommend_build(48.0) is BUILDS[0]
    assert recommend_build(1.0) is BUILDS[-1]
    picks = {v: (recommend_build(v).name, recommend_build(v).transformer, recommend_build(v).offload)
             for v in (24.0, 16.0, 12.0, 8.0)}
    assert picks == {24.0: ("resident", "fp8", None), 16.0: ("quality", "bf16", "model"),
                     12.0: ("offload", "fp8", "model"), 8.0: ("minimal", "fp8", "group")}


def test_fp8_storage_keeps_the_first_parameter_module():
    """core/fp8.py: the module owning the encoder's first parameter is kept in
    bf16 so `model.dtype` (which pipelines cast embeds and latents to) stays bf16."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch without float8")
    from imference_engine.core.fp8 import apply_fp8_storage

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.proj = torch.nn.Linear(4, 4)

    m = _M().to(torch.bfloat16)
    apply_fp8_storage(m)
    assert m.embed_tokens.weight.dtype == torch.bfloat16
    assert m.proj.weight.dtype == torch.float8_e4m3fn
    assert next(m.parameters()).dtype == torch.bfloat16


def test_registered_as_backends_and_catalog_rows(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text(
        "models:\n"
        "  - name: longcat\n"
        "    engine: longcat\n"
        "    weights: auto\n"
        "  - name: longcat-edit\n"
        "    engine: longcat-edit\n"
        "    weights: auto\n",
        encoding="utf-8",
    )
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends["longcat"], LongCatBackend)
    assert isinstance(engine._backends["longcat-edit"], LongCatEditBackend)
    assert engine._models.config_for("longcat-edit").weights_path == "auto"
