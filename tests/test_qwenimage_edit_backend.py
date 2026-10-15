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


# ---------------------------------------------------------------- GGUF lite build

def test_lite_settings_from_env(monkeypatch):
    from imference_engine.qwenimage import edit_lite

    monkeypatch.delenv("QWEN_EDIT_LIGHTNING", raising=False)
    assert edit_lite.lightning_setting() == "4"
    monkeypatch.setenv("QWEN_EDIT_LIGHTNING", "none")
    assert edit_lite.lightning_setting() == "none"
    monkeypatch.setenv("QWEN_EDIT_ENCODER", "bf16")
    assert edit_lite.encoder_mode() == "bf16"


def test_lite_base_skips_encoder_and_transformer_weights():
    from imference_engine.qwenimage.edit_lite import LITE_BASE_PATTERNS

    assert "text_encoder/*" not in LITE_BASE_PATTERNS and "transformer/*" not in LITE_BASE_PATTERNS
    assert {"text_encoder/config.json", "transformer/config.json", "processor/*", "vae/*"} <= set(LITE_BASE_PATTERNS)


def test_gguf_weights_take_the_lite_path(monkeypatch):
    from imference_engine.qwenimage import edit_lite

    seen = {}

    def fake(path, base, **kw):
        seen.update(path=path, base=base, **kw)
        return "PIPE"

    monkeypatch.setattr(edit_lite, "load_lite_pipeline", fake)
    be = QwenImageEditBackend(cache_dir="/c", cdn_base="https://cdn")
    assert be.load_pipeline(local_path="/m/qwen-image-edit-2511-Q4_K_M.GGUF") == "PIPE"
    assert seen["base"] == "Qwen/Qwen-Image-Edit-2511" and seen["cache_dir"] == "/c"
    assert seen["full_base_patterns"] == QwenImageEditBackend.BASE_PATTERNS


def test_prefetch_uses_lite_patterns_with_nf4(monkeypatch):
    from imference_engine.qwenimage import edit_lite
    from imference_engine.runtime import offline

    calls = []
    monkeypatch.setattr(offline, "local_repo_dir", lambda repo, pats, *a, **k: calls.append((repo, pats)))
    monkeypatch.setattr(edit_lite, "encoder_mode", lambda: "nf4")
    QwenImageEditBackend().prefetch_base()
    monkeypatch.setattr(edit_lite, "encoder_mode", lambda: "bf16")
    QwenImageEditBackend().prefetch_base()
    assert calls == [("Qwen/Qwen-Image-Edit-2511", edit_lite.LITE_BASE_PATTERNS),
                     ("Qwen/Qwen-Image-Edit-2511", QwenImageEditBackend.BASE_PATTERNS)]


def test_text_encoder_weights_present_or_local_need_no_download(tmp_path, monkeypatch):
    import json

    from imference_engine.qwenimage.edit_lite import ensure_text_encoder_weights

    te = tmp_path / "text_encoder"
    te.mkdir()
    (te / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s1.safetensors"}}))
    (te / "s1.safetensors").write_bytes(b"x")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("downloaded"))
    ensure_text_encoder_weights("Qwen/Qwen-Image-Edit-2511", str(tmp_path), cdn_base=None)
    (te / "s1.safetensors").unlink()
    ensure_text_encoder_weights(str(tmp_path), str(tmp_path), cdn_base=None)  # local base dir: as-is


def test_gguf_registration_gets_the_lightning_recipe(monkeypatch):
    from imference_engine import Engine, RuntimeConfig
    from imference_engine.catalog.defaults import GenerationDefaults

    monkeypatch.delenv("QWEN_EDIT_LIGHTNING", raising=False)
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    engine.register_model("lite", backend="qwenimage-edit", weights_path="/m/qwen-image-edit-2511-Q4_K_M.gguf")
    engine.register_model("bf16", backend="qwenimage-edit", weights_path="/m/edit.safetensors")
    engine.register_model("row", backend="qwenimage-edit", weights_path="/m/x.gguf",
                          defaults=GenerationDefaults(num_steps=8))
    d = {n: engine._models.config_for(n).defaults for n in ("lite", "bf16", "row")}
    assert (d["lite"].num_steps, d["lite"].guidance_scale) == (4, 1.0)
    assert d["bf16"].num_steps is None and d["row"].num_steps == 8
    monkeypatch.setenv("QWEN_EDIT_LIGHTNING", "none")
    assert QwenImageEditBackend().weights_defaults("/m/x.gguf") is None


def test_keep_encoder_setting_and_reserve(monkeypatch):
    from imference_engine.qwenimage import edit_lite

    monkeypatch.delenv("QWEN_EDIT_KEEP_ENCODER", raising=False)
    assert edit_lite.keep_encoder_setting() == "auto"
    for v, want in (("1", "1"), ("always", "1"), ("0", "0"), ("off", "0"), ("weird", "auto")):
        monkeypatch.setenv("QWEN_EDIT_KEEP_ENCODER", v)
        assert edit_lite.keep_encoder_setting() == want
    monkeypatch.delenv("QWEN_EDIT_DENOISE_RESERVE_GB", raising=False)
    monkeypatch.delenv("QWEN_EDIT_RESERVE_GB_PER_MP", raising=False)
    one_mp = edit_lite.denoise_reserve_bytes(1000, 1000)
    four_mp = edit_lite.denoise_reserve_bytes(2000, 2000)
    assert one_mp == int(4.0 * (1 << 30))
    assert four_mp > one_mp
    monkeypatch.setenv("QWEN_EDIT_DENOISE_RESERVE_GB", "1")
    monkeypatch.setenv("QWEN_EDIT_RESERVE_GB_PER_MP", "0")
    assert edit_lite.denoise_reserve_bytes(None, None) == 1 << 30


def test_keep_encoder_decision(monkeypatch):
    """auto keeps the encoder only when the free VRAM covers the reserve; a
    pipeline without the on-demand loader never keeps anything."""
    from imference_engine.qwenimage import edit_lite

    cls = edit_lite._pipeline_class()
    pipe = cls.__new__(cls)
    object.__setattr__(pipe, "_encoder_loader", None)
    assert pipe._keep_encoder(1024, 1024) is False

    import torch
    object.__setattr__(pipe, "_encoder_loader", lambda: None)
    monkeypatch.setattr(cls, "text_encoder", object(), raising=False)
    monkeypatch.setenv("QWEN_EDIT_KEEP_ENCODER", "auto")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (10 << 30, 24 << 30))
    assert pipe._keep_encoder(1024, 1024) is True
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (2 << 30, 24 << 30))
    assert pipe._keep_encoder(1024, 1024) is False
    monkeypatch.setenv("QWEN_EDIT_KEEP_ENCODER", "1")
    assert pipe._keep_encoder(1024, 1024) is True
    monkeypatch.setenv("QWEN_EDIT_KEEP_ENCODER", "0")
    assert pipe._keep_encoder(1024, 1024) is False


def test_prefetch_lite_extras_fetches_lora_and_nf4(monkeypatch, tmp_path):
    from imference_engine.qwenimage import edit_lite
    from imference_engine.runtime import offline

    calls = []
    monkeypatch.setattr(offline, "fetch", lambda repo, f, cdn, cache, namespace=None: calls.append(("fetch", repo, f)))
    monkeypatch.setattr(offline, "_cdn_snapshot", lambda cdn, repo, dest: calls.append(("snap", repo, dest)))
    monkeypatch.setattr(edit_lite, "encoder_mode", lambda: "nf4")
    monkeypatch.delenv("QWEN_EDIT_LIGHTNING", raising=False)
    edit_lite.prefetch_lite_extras(cache_dir=str(tmp_path), cdn_base="https://cdn.example/image")
    assert ("fetch", edit_lite.LIGHTNING_REPO, edit_lite.LIGHTNING_FILE) in calls
    assert any(c[0] == "snap" and c[1] == edit_lite.NF4_ENCODER_REPO for c in calls)

    # Already on disk -> no second snapshot; no CDN -> nothing to pull.
    calls.clear()
    d = edit_lite.nf4_encoder_dir(str(tmp_path))
    import os
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "config.json"), "w").close()
    open(os.path.join(d, "model.safetensors"), "w").close()
    edit_lite.prefetch_lite_extras(cache_dir=str(tmp_path), cdn_base="https://cdn.example/image")
    assert not any(c[0] == "snap" for c in calls)
    calls.clear()
    monkeypatch.setenv("QWEN_EDIT_LIGHTNING", "none")
    edit_lite.prefetch_lite_extras(cache_dir=str(tmp_path), cdn_base=None)
    assert calls == []


def test_edit_dimensions_follow_small_inputs(monkeypatch):
    from imference_engine.qwenimage import edit_lite

    monkeypatch.delenv("QWEN_EDIT_MIN_MP", raising=False)
    monkeypatch.delenv("QWEN_EDIT_MAX_MP", raising=False)
    # >= 1 MP: diffusers' default sizing is kept untouched.
    assert edit_lite.edit_dimensions(1024, 1024) is None
    assert edit_lite.edit_dimensions(3000, 2000) is None
    # Small inputs are floored at 0.5 MP, keep their ratio, multiples of 32.
    w, h = edit_lite.edit_dimensions(512, 512)
    assert (w, h) == (736, 736)
    w, h = edit_lite.edit_dimensions(548, 342)
    assert w % 32 == 0 and h % 32 == 0 and abs(w / h - 548 / 342) < 0.1
    assert 0.45 * edit_lite.MEGAPIXEL < w * h < 0.55 * edit_lite.MEGAPIXEL
    # Between the bounds: the input's own area.
    w, h = edit_lite.edit_dimensions(896, 896)
    assert (w, h) == (896, 896)
    monkeypatch.setenv("QWEN_EDIT_MIN_MP", "0.25")
    assert edit_lite.edit_dimensions(512, 512) == (512, 512)
    monkeypatch.setenv("QWEN_EDIT_MIN_MP", "1")
    assert edit_lite.edit_dimensions(512, 512) is None
    assert edit_lite.edit_dimensions(0, 512) is None


def test_inference_kwargs_size_small_inputs(monkeypatch):
    from PIL import Image

    monkeypatch.delenv("QWEN_EDIT_MIN_MP", raising=False)
    monkeypatch.delenv("QWEN_EDIT_MAX_MP", raising=False)
    be = QwenImageEditBackend()
    small, big = Image.new("RGB", (512, 512)), Image.new("RGB", (1600, 1200))
    kw = be.build_inference_kwargs(width=1, height=1, num_steps=4, guidance_scale=1.0, clip_skip=None,
                                   chunk_size=1, generator=None, image=small)
    assert (kw["width"], kw["height"]) == (736, 736)
    kw = be.build_inference_kwargs(width=1, height=1, num_steps=4, guidance_scale=1.0, clip_skip=None,
                                   chunk_size=1, generator=None, image=big)
    assert "width" not in kw and "height" not in kw
    # diffusers sizes on the last image: so do we.
    kw = be.build_inference_kwargs(width=1, height=1, num_steps=4, guidance_scale=1.0, clip_skip=None,
                                   chunk_size=1, generator=None, image=[big, small])
    assert (kw["width"], kw["height"]) == (736, 736)


def test_vae_condition_area_is_scoped():
    pytest.importorskip("diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus")
    from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as mod
    from imference_engine.qwenimage import edit_lite

    before = mod.VAE_IMAGE_SIZE
    with edit_lite._vae_condition_area(736 * 736):
        assert mod.VAE_IMAGE_SIZE == 736 * 736
    assert mod.VAE_IMAGE_SIZE == before
    with pytest.raises(RuntimeError):
        with edit_lite._vae_condition_area(512 * 512):
            raise RuntimeError("boom")
    assert mod.VAE_IMAGE_SIZE == before
    with edit_lite._vae_condition_area(None):
        assert mod.VAE_IMAGE_SIZE == before
