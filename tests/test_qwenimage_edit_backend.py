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
