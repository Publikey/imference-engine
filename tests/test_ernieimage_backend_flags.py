"""No-torch assertions on the ERNIE-Image backend: engine defaults, the offline
base-component patterns (no prompt enhancer), inference kwargs, the scheduler
shift override, the per-machine build table and registration."""
from __future__ import annotations

import pytest

from imference_engine.ernieimage import (
    AUTO,
    BUILDS,
    ErnieImageBackend,
    build_named,
    recommend_build,
)


def test_engine_key():
    assert ErnieImageBackend.engine == "ernie-image"


def test_engine_defaults_are_the_turbo_recipe():
    d = ErnieImageBackend().engine_defaults()
    assert d.num_steps == 8
    assert d.guidance_scale == 1.0
    assert d.scheduler is None and d.backend_options == {}


def test_base_patterns_skip_the_prompt_enhancer_and_transformer_weights():
    pats = ErnieImageBackend.BASE_PATTERNS
    for p in ("model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*",
              "vae/*", "transformer/config.json", "LICENSE"):
        assert p in pats
    assert not any(p.startswith("pe") for p in pats)        # 7.7 GB never pulled
    assert "transformer/*" not in pats
    assert "transformer/*" in ErnieImageBackend.REPO_PATTERNS
    assert not any(p.startswith("pe") for p in ErnieImageBackend.REPO_PATTERNS)


def test_encode_prompts_passes_raw_strings():
    be = ErnieImageBackend()
    assert be.encode_prompts(None, "a cat", None) == {"prompt": "a cat"}
    assert be.encode_prompts(None, "a cat", "blurry") == {
        "prompt": "a cat", "negative_prompt": "blurry"}


def test_build_inference_kwargs_always_disable_the_enhancer():
    be = ErnieImageBackend()
    kw = be.build_inference_kwargs(
        width=1264, height=848, num_steps=8, guidance_scale=1.0,
        clip_skip=None, chunk_size=2, generator=None)
    assert kw["use_pe"] is False            # upstream default is True
    assert kw["width"] == 1264 and kw["height"] == 848
    assert kw["num_images_per_prompt"] == 2
    assert "clip_skip" not in kw and "image" not in kw and "strength" not in kw


def test_img2img_is_refused():
    with pytest.raises(NotImplementedError):
        ErnieImageBackend().make_img2img(object())


def test_scheduler_name_is_ignored_and_default_is_restored():
    """Without a `shift` option the base scheduler object is left / put back —
    a previous request's override must not leak into the next one."""

    class _Pipe:
        scheduler = object()

    pipe = _Pipe()
    base = pipe.scheduler
    be = ErnieImageBackend()
    be.apply_scheduler(pipe, "euler_a")
    assert pipe.scheduler is base
    pipe.scheduler = object()                 # as if an override had been applied
    be.apply_scheduler(pipe, None)
    assert pipe.scheduler is base


@pytest.mark.parametrize("vram,name,encoder,offload", [
    (None, "bf16", "bf16", None),       # no CUDA reading
    (32.0, "quality", "bf16", None),
    (24.0, "quality", "bf16", None),
    (16.0, "balanced", "fp8", None),
    (12.0, "offload", "fp8", "model"),
    (8.0, "compact", "fp8", "model"),
    (6.0, "minimal", "fp8", "group"),
])
def test_recommend_build_per_vram_tier(vram, name, encoder, offload):
    b = recommend_build(vram)
    assert (b.name, b.encoder, b.offload) == (name, encoder, offload)


def test_builds_leave_headroom_over_the_measured_peak():
    for b in BUILDS:
        assert b.min_vram_gb == 0.0 or b.min_vram_gb >= b.peak_vram_gb + 1.4, b.name
    floors = [b.min_vram_gb for b in BUILDS]
    assert floors == sorted(floors, reverse=True)
    assert BUILDS[-1].min_vram_gb == 0.0
    assert build_named("minimal").runtime_kwargs() == {
        "enable_offload": True, "offload_mode": "group"}


def test_encoder_mode_follows_the_vram_table(monkeypatch):
    from imference_engine.ernieimage import backend as mod

    monkeypatch.delenv("ERNIE_IMAGE_ENCODER", raising=False)
    monkeypatch.setattr(mod, "_vram_gb", lambda: 24.0)
    assert ErnieImageBackend._resolve_encoder_mode() == "bf16"
    monkeypatch.setattr(mod, "_vram_gb", lambda: 16.0)
    assert ErnieImageBackend._resolve_encoder_mode() == "fp8"
    monkeypatch.setenv("ERNIE_IMAGE_ENCODER", "bf16")
    assert ErnieImageBackend._resolve_encoder_mode() == "bf16"
    monkeypatch.setenv("ERNIE_IMAGE_ENCODER", "fp8")
    monkeypatch.setattr(mod, "_vram_gb", lambda: None)   # float8 needs CUDA
    assert ErnieImageBackend._resolve_encoder_mode() == "bf16"


def test_auto_resolves_to_the_recommended_file(monkeypatch):
    from imference_engine.ernieimage import backend as mod
    from imference_engine.runtime import offline

    seen = {}

    def fake_fetch(repo, filename, cdn_base, cache_dir, *, namespace):
        seen.update(repo=repo, filename=filename, cdn=cdn_base, ns=namespace)
        return f"/tree/{repo}/{filename}"

    monkeypatch.setattr(offline, "fetch", fake_fetch)
    monkeypatch.setattr(mod, "_vram_gb", lambda: 16.0)
    be = ErnieImageBackend(cdn_base="https://cdn.example/image")
    assert be.resolve_weights(AUTO) == (
        "/tree/unsloth/ERNIE-Image-Turbo-GGUF/ernie-image-turbo-Q5_K_M.gguf")
    assert seen["cdn"] == "https://cdn.example/image" and seen["ns"] == "image"
    assert be.resolve_weights("/w/x.gguf") == "/w/x.gguf"


def test_registered_as_a_backend_and_catalog_row(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text(
        "models:\n"
        "  - name: ernie-turbo\n"
        "    engine: ernie-image\n"
        "    weights: auto\n"
        "  - name: ernie-base\n"
        "    engine: ernie-image\n"
        "    weights: /cache/ernie-image.safetensors\n"
        "    base_model: baidu/ERNIE-Image\n"
        "    defaults: {num_steps: 50, guidance_scale: 4.0}\n",
        encoding="utf-8",
    )
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends["ernie-image"], ErnieImageBackend)
    assert engine._models.config_for("ernie-turbo").weights_path == "auto"
    base = engine._models.config_for("ernie-base")
    assert base.base_model == "baidu/ERNIE-Image"
    assert base.defaults.num_steps == 50 and base.defaults.guidance_scale == 4.0
