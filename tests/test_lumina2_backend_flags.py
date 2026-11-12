"""No-torch assertions on the Lumina-Image 2.0 backend: engine defaults, base
patterns, all-in-one detection, per-request options, the slim CDN base, the
per-machine build table and registration."""
from __future__ import annotations

import pytest

from imference_engine.lumina2 import (
    AUTO,
    BUILDS,
    Lumina2Backend,
    build_named,
    recommend_build,
)
from imference_engine.lumina2 import backend as mod


def test_engine_key():
    assert Lumina2Backend.engine == "lumina2"


def test_engine_defaults_are_true_cfg():
    d = Lumina2Backend().engine_defaults()
    assert d.num_steps == 30
    assert d.guidance_scale == 4.0
    assert d.scheduler is None and d.backend_options == {}


def test_base_patterns_carry_the_licence_texts_but_only_the_transformer_config():
    pats = Lumina2Backend.BASE_PATTERNS
    for p in ("model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*",
              "vae/*", "transformer/config.json", "LICENSE*", "NOTICE*", "GEMMA_TERMS*"):
        assert p in pats
    assert "transformer/*" not in pats
    assert "transformer/*" in Lumina2Backend.REPO_PATTERNS


def test_all_in_one_detection():
    aio = {"model.diffusion_model.cap_embedder.0.weight": {}, "vae.decoder.x": {},
           "text_encoders.gemma2_2b.y": {}, "__metadata__": {}}
    prefixed_only = {"model.diffusion_model.cap_embedder.0.weight": {}}
    bare = {"cap_embedder.0.weight": {}, "layers.0.attention.qkv.weight": {}}
    assert mod.is_all_in_one(aio) is True
    assert mod.is_all_in_one(prefixed_only) is False   # diffusers strips this itself
    assert mod.is_all_in_one(bare) is False


def test_encode_prompts_keeps_the_negative():
    be = Lumina2Backend()
    assert be.encode_prompts(None, "a cat", None) == {"prompt": "a cat"}
    assert be.encode_prompts(None, "a cat", "blurry") == {
        "prompt": "a cat", "negative_prompt": "blurry"}


def test_img2img_is_refused():
    with pytest.raises(NotImplementedError):
        Lumina2Backend().make_img2img(object())


def test_backend_options_reach_the_call_and_do_not_leak():
    class _Pipe:
        scheduler = object()

    be, pipe = Lumina2Backend(), _Pipe()
    base = pipe.scheduler
    be.apply_scheduler(pipe, None, cfg_trunc_ratio=0.25, system_prompt="sys")
    kw = be.build_inference_kwargs(
        width=1024, height=768, num_steps=30, guidance_scale=4.0,
        clip_skip=None, chunk_size=2, generator=None)
    assert kw["cfg_trunc_ratio"] == 0.25 and kw["system_prompt"] == "sys"
    assert kw["width"] == 1024 and kw["height"] == 768 and kw["num_images_per_prompt"] == 2
    assert pipe.scheduler is base            # no shift -> base scheduler

    # the next request sets nothing: the previous options must be gone
    pipe.scheduler = object()                # as if a shift override had been applied
    be.apply_scheduler(pipe, "euler_a")
    kw = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=30, guidance_scale=4.0,
        clip_skip=None, chunk_size=1, generator=None)
    assert "cfg_trunc_ratio" not in kw and "system_prompt" not in kw
    assert "clip_skip" not in kw and "image" not in kw
    assert pipe.scheduler is base


@pytest.mark.parametrize("vram,name,encoder,offload", [
    (None, "full", "bf16", None),       # no CUDA reading
    (24.0, "full", "bf16", None),
    (16.0, "balanced", "fp8", None),
    (12.0, "compact", "fp8", None),
    (8.0, "offload", "fp8", "model"),
    (6.0, "offload", "fp8", "model"),
    (4.0, "minimal", "fp8", "group"),
])
def test_recommend_build_per_vram_tier(vram, name, encoder, offload):
    b = recommend_build(vram)
    assert (b.name, b.encoder, b.offload) == (name, encoder, offload)


def test_builds_leave_headroom_over_the_measured_peak():
    for b in BUILDS:
        assert b.min_vram_gb == 0.0 or b.min_vram_gb >= b.peak_vram_gb + 2.0, b.name
    floors = [b.min_vram_gb for b in BUILDS]
    assert floors == sorted(floors, reverse=True)
    assert BUILDS[-1].min_vram_gb == 0.0
    assert build_named("offload").runtime_kwargs() == {
        "enable_offload": True, "offload_mode": "model"}


def test_encoder_mode_follows_the_vram_table(monkeypatch):
    monkeypatch.delenv("LUMINA2_ENCODER", raising=False)
    monkeypatch.setattr(mod, "_vram_gb", lambda: 24.0)
    assert Lumina2Backend._resolve_encoder_mode() == "bf16"
    monkeypatch.setattr(mod, "_vram_gb", lambda: 16.0)
    assert Lumina2Backend._resolve_encoder_mode() == "fp8"
    monkeypatch.setenv("LUMINA2_ENCODER", "bf16")
    assert Lumina2Backend._resolve_encoder_mode() == "bf16"
    monkeypatch.setenv("LUMINA2_ENCODER", "fp8")
    monkeypatch.setattr(mod, "_vram_gb", lambda: None)   # float8 needs CUDA
    assert Lumina2Backend._resolve_encoder_mode() == "bf16"


def test_auto_resolves_to_the_recommended_file(monkeypatch):
    from imference_engine.runtime import offline

    seen = {}

    def fake_fetch(repo, filename, cdn_base, cache_dir, *, namespace):
        seen.update(repo=repo, filename=filename)
        return f"/tree/{repo}/{filename}"

    monkeypatch.setattr(offline, "fetch", fake_fetch)
    monkeypatch.setattr(mod, "_vram_gb", lambda: 8.0)
    be = Lumina2Backend(cdn_base="https://cdn.example/image")
    assert be.resolve_weights(AUTO) == "/tree/calcuis/lumina-gguf/lumina2-q8_0.gguf"
    assert be.resolve_weights("/w/x.safetensors") == "/w/x.safetensors"


def test_slim_base_is_used_when_the_cdn_serves_it(monkeypatch, tmp_path):
    """With a CDN, the default base resolves to the bf16-encoder mirror; a CDN
    without it (or a custom base_model) falls back to the upstream repo."""
    from imference_engine.runtime import offline

    calls = []
    monkeypatch.setattr(offline, "_cdn_snapshot",
                        lambda cdn, repo, dest, patterns=None: calls.append(("cdn", repo)))
    monkeypatch.setattr(offline, "local_repo_dir",
                        lambda repo, *a, **k: calls.append(("upstream", repo)) or f"/up/{repo}")
    be = Lumina2Backend(cache_dir=str(tmp_path), cdn_base="https://cdn.example/image")
    d = be._base_dir(mod.DEFAULT_BASE)
    assert d.replace("\\", "/").endswith(mod.SLIM_BASE)
    assert calls == [("cdn", mod.SLIM_BASE)]

    calls.clear()

    def missing(cdn, repo, dest, patterns=None):
        raise OSError("404")

    monkeypatch.setattr(offline, "_cdn_snapshot", missing)
    assert be._base_dir(mod.DEFAULT_BASE) == f"/up/{mod.DEFAULT_BASE}"
    assert be._base_dir("someone/custom-base") == "/up/someone/custom-base"
    assert calls == [("upstream", mod.DEFAULT_BASE), ("upstream", "someone/custom-base")]

    # no CDN at all: straight to upstream
    calls.clear()
    assert Lumina2Backend(cache_dir=str(tmp_path))._base_dir(mod.DEFAULT_BASE) == (
        f"/up/{mod.DEFAULT_BASE}")
    assert calls == [("upstream", mod.DEFAULT_BASE)]


def test_registered_as_a_backend_and_catalog_row(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text(
        "models:\n"
        "  - name: lumina\n"
        "    engine: lumina2\n"
        "    weights: auto\n"
        "  - name: netayume\n"
        "    engine: lumina2\n"
        "    weights: /cache/NetaYume_v4_all_in_one.safetensors\n"
        "    defaults: {num_steps: 40, guidance_scale: 5.0}\n",
        encoding="utf-8",
    )
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends["lumina2"], Lumina2Backend)
    assert engine._models.config_for("lumina").weights_path == "auto"
    yume = engine._models.config_for("netayume")
    assert yume.defaults.num_steps == 40 and yume.defaults.guidance_scale == 5.0
