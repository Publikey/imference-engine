"""No-torch assertions on the FLUX.2 klein backend: engine defaults, the
offline base-component patterns, inference kwargs (t2i vs reference editing),
the per-machine build table and registration through the Engine."""
from __future__ import annotations

import pytest

from imference_engine.flux2klein import (
    AUTO,
    BUILDS,
    Flux2KleinBackend,
    build_named,
    recommend_build,
)


class _Cfg:
    def __init__(self, is_distilled):
        self.is_distilled = is_distilled


class _Pipe:
    def __init__(self, is_distilled=True):
        self.config = _Cfg(is_distilled)
        self.encoded = []

    def encode_prompt(self, prompt, max_sequence_length):
        self.encoded.append((prompt, max_sequence_length))
        return "EMBEDS", "IDS"


def test_engine_key():
    assert Flux2KleinBackend.engine == "flux2-klein"


def test_engine_defaults_are_the_distilled_recipe():
    d = Flux2KleinBackend().engine_defaults()
    assert d.num_steps == 4
    assert d.guidance_scale == 1.0
    assert d.scheduler is None and d.backend_options == {}


def test_base_patterns_carry_encoder_vae_license_but_only_the_transformer_config():
    pats = Flux2KleinBackend.BASE_PATTERNS
    for p in ("model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*",
              "vae/*", "transformer/config.json", "LICENSE.md"):
        assert p in pats
    assert "transformer/*" not in pats
    # repo mode adds the transformer weights, never the duplicate root single file
    assert "transformer/*" in Flux2KleinBackend.REPO_PATTERNS
    assert not any(p.endswith(".safetensors") for p in Flux2KleinBackend.REPO_PATTERNS)


def test_distilled_ignores_negative_prompt():
    be, pipe = Flux2KleinBackend(), _Pipe(is_distilled=True)
    assert be.encode_prompts(pipe, "a cat", "blurry") == {"prompt": "a cat"}
    assert pipe.encoded == []


def test_base_checkpoint_encodes_the_negative_prompt():
    be, pipe = Flux2KleinBackend(), _Pipe(is_distilled=False)
    out = be.encode_prompts(pipe, "a cat", "blurry")
    assert out == {"prompt": "a cat", "negative_prompt_embeds": "EMBEDS"}
    assert pipe.encoded == [("blurry", Flux2KleinBackend.MAX_SEQUENCE_LENGTH)]
    # nothing to encode for an empty / blank negative
    assert be.encode_prompts(pipe, "a cat", " ") == {"prompt": "a cat"}
    assert be.encode_prompts(pipe, "a cat", None) == {"prompt": "a cat"}


def test_build_inference_kwargs_t2i_and_reference_edit():
    be = Flux2KleinBackend()
    t2i = be.build_inference_kwargs(
        width=1024, height=768, num_steps=4, guidance_scale=1.0,
        clip_skip=None, chunk_size=2, generator=None)
    assert t2i["width"] == 1024 and t2i["height"] == 768
    assert t2i["num_images_per_prompt"] == 2
    assert "image" not in t2i and "strength" not in t2i and "clip_skip" not in t2i

    one = object()
    edit = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=4, guidance_scale=1.0,
        clip_skip=None, chunk_size=1, generator=None, image=one, strength=0.6)
    assert edit["image"] == [one]
    assert "strength" not in edit                        # a condition, not a noised start
    assert "width" not in edit and "height" not in edit  # size follows the reference

    two = [object(), object()]
    multi = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=4, guidance_scale=1.0,
        clip_skip=None, chunk_size=1, generator=None, image=two)
    assert multi["image"] == two


def test_make_img2img_is_the_same_pipe():
    pipe = _Pipe()
    assert Flux2KleinBackend().make_img2img(pipe) is pipe


def test_resolve_weights_passes_explicit_paths_through():
    be = Flux2KleinBackend()
    assert be.resolve_weights("/w/flux-2-klein-4b.safetensors") == "/w/flux-2-klein-4b.safetensors"


@pytest.mark.parametrize("vram,name,encoder,offload", [
    (None, "full", "bf16", None),       # no CUDA reading
    (48.0, "full", "bf16", None),
    (24.0, "full", "bf16", None),
    (16.0, "balanced", "fp8", None),
    (12.0, "compact", "fp8", None),
    (10.0, "offload", "fp8", "model"),
    (8.0, "offload", "fp8", "model"),
    (6.0, "minimal", "fp8", "group"),
    (4.0, "minimal", "fp8", "group"),
])
def test_recommend_build_per_vram_tier(vram, name, encoder, offload):
    b = recommend_build(vram)
    assert (b.name, b.encoder, b.offload) == (name, encoder, offload)


def test_builds_leave_headroom_over_the_measured_peak():
    """A resident build must not be recommended for a card smaller than its
    measured peak plus the allocator's reserve."""
    for b in BUILDS:
        if b.offload is None:
            assert b.min_vram_gb >= b.peak_vram_gb + 1.5, b.name
    floors = [b.min_vram_gb for b in BUILDS]
    assert floors == sorted(floors, reverse=True)
    assert BUILDS[-1].min_vram_gb == 0.0   # always a fallback


def test_runtime_kwargs_match_the_offload_mode():
    assert build_named("full").runtime_kwargs() == {"enable_offload": False}
    assert build_named("offload").runtime_kwargs() == {
        "enable_offload": True, "offload_mode": "model"}
    assert build_named("minimal").runtime_kwargs() == {
        "enable_offload": True, "offload_mode": "group"}
    with pytest.raises(KeyError):
        build_named("nope")


def test_encoder_and_fp8_env_settings(monkeypatch):
    from imference_engine.flux2klein import backend as mod

    monkeypatch.delenv("FLUX2_KLEIN_ENCODER", raising=False)
    monkeypatch.delenv("FLUX2_KLEIN_FP8_STORAGE", raising=False)
    assert mod.encoder_setting() == "auto" and mod.fp8_storage_setting() == "auto"
    monkeypatch.setenv("FLUX2_KLEIN_ENCODER", "FP8")
    monkeypatch.setenv("FLUX2_KLEIN_FP8_STORAGE", "0")
    assert mod.encoder_setting() == "fp8" and mod.fp8_storage_setting() == "0"
    monkeypatch.setenv("FLUX2_KLEIN_ENCODER", "nf4")   # unknown -> auto
    assert mod.encoder_setting() == "auto"


def test_encoder_mode_follows_the_vram_table(monkeypatch):
    from imference_engine.flux2klein import backend as mod

    monkeypatch.delenv("FLUX2_KLEIN_ENCODER", raising=False)
    monkeypatch.setattr(mod, "_vram_gb", lambda: 24.0)
    assert Flux2KleinBackend._resolve_encoder_mode() == "bf16"
    monkeypatch.setattr(mod, "_vram_gb", lambda: 12.0)
    assert Flux2KleinBackend._resolve_encoder_mode() == "fp8"
    monkeypatch.setenv("FLUX2_KLEIN_ENCODER", "bf16")
    assert Flux2KleinBackend._resolve_encoder_mode() == "bf16"
    # float8 storage needs CUDA: without a VRAM reading it is always bf16
    monkeypatch.setenv("FLUX2_KLEIN_ENCODER", "fp8")
    monkeypatch.setattr(mod, "_vram_gb", lambda: None)
    assert Flux2KleinBackend._resolve_encoder_mode() == "bf16"


def test_auto_resolves_to_the_recommended_file(monkeypatch):
    from imference_engine.flux2klein import backend as mod
    from imference_engine.runtime import offline

    seen = {}

    def fake_fetch(repo, filename, cdn_base, cache_dir, *, namespace):
        seen.update(repo=repo, filename=filename, cdn=cdn_base, ns=namespace)
        return f"/tree/{repo}/{filename}"

    monkeypatch.setattr(offline, "fetch", fake_fetch)
    monkeypatch.setattr(mod, "_vram_gb", lambda: 16.0)
    be = Flux2KleinBackend(cdn_base="https://cdn.example/image")
    path = be.resolve_weights(AUTO)
    assert path == "/tree/unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q8_0.gguf"
    assert seen == {"repo": "unsloth/FLUX.2-klein-4B-GGUF",
                    "filename": "flux-2-klein-4b-Q8_0.gguf",
                    "cdn": "https://cdn.example/image", "ns": "image"}


def test_registered_as_a_backend_and_catalog_row(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text(
        "models:\n"
        "  - name: klein\n"
        "    engine: flux2-klein\n"
        "    weights: auto\n"
        "  - name: klein-base\n"
        "    engine: flux2-klein\n"
        "    weights: /cache/flux-2-klein-base-4b.safetensors\n"
        "    base_model: black-forest-labs/FLUX.2-klein-base-4B\n"
        "    defaults: {num_steps: 50, guidance_scale: 4.0}\n",
        encoding="utf-8",
    )
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends["flux2-klein"], Flux2KleinBackend)
    assert engine._models.config_for("klein").weights_path == "auto"
    base = engine._models.config_for("klein-base")
    assert base.base_model == "black-forest-labs/FLUX.2-klein-base-4B"
    assert base.defaults.num_steps == 50 and base.defaults.guidance_scale == 4.0
