"""No-torch assertions on the Ovis-Image backend: defaults, repo patterns,
inference kwargs, the storage build table and registration."""
from __future__ import annotations

import pytest

from imference_engine.ovisimage import AUTO, BUILDS, OvisImageBackend, recommend_build


def test_engine_key_and_defaults():
    be = OvisImageBackend()
    assert be.engine == "ovis-image"
    d = be.engine_defaults()
    assert d.num_steps == 50 and d.guidance_scale == 5.0
    assert be.resolve_weights(AUTO) == "AIDC-AI/Ovis-Image-7B"
    assert be.resolve_weights("/w/x.safetensors") == "/w/x.safetensors"


def test_patterns_take_the_diffusers_part_of_the_repo_only():
    pats = OvisImageBackend.BASE_PATTERNS
    for p in ("model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*",
              "vae/*", "transformer/*", "LICENSE", "NOTICE"):
        assert p in pats
    # not the 29 GB original checkpoint nor the full VLM at the repo root
    assert not any("ovis_image" in p or "Ovis2.5" in p or p == "*" for p in pats)
    assert "transformer/*" not in OvisImageBackend.CONFIG_PATTERNS
    assert "transformer/config.json" in OvisImageBackend.CONFIG_PATTERNS


def test_kwargs_and_prompts():
    be = OvisImageBackend()
    kw = be.build_inference_kwargs(
        width=1344, height=768, num_steps=50, guidance_scale=5.0,
        clip_skip=None, chunk_size=2, generator=None)
    assert kw == {"num_inference_steps": 50, "guidance_scale": 5.0, "width": 1344,
                  "height": 768, "generator": None, "num_images_per_prompt": 2}
    assert be.encode_prompts(None, "a sign", None) == {"prompt": "a sign", "negative_prompt": ""}
    assert be.encode_prompts(None, "a sign", "blurry")["negative_prompt"] == "blurry"
    with pytest.raises(NotImplementedError):
        be.make_img2img(object())


@pytest.mark.parametrize("vram,name,transformer,offload", [
    (None, "bf16", "bf16", "model"),
    (24.0, "quality", "bf16", None),
    (20.0, "resident", "fp8", None),
    (16.0, "offload", "fp8", "model"),
    (12.0, "offload", "fp8", "model"),
    (8.0, "minimal", "fp8", "group"),
])
def test_recommend_build_per_vram_tier(vram, name, transformer, offload):
    b = recommend_build(vram)
    assert (b.name, b.transformer, b.offload) == (name, transformer, offload)


def test_build_table_is_ordered():
    floors = [b.min_vram_gb for b in BUILDS]
    assert floors == sorted(floors, reverse=True) and BUILDS[-1].min_vram_gb == 0.0
    for b in BUILDS:
        assert b.filename == "" and b.encoder == "fp8"
        if b.offload is None:
            assert b.min_vram_gb >= b.peak_vram_gb + 2.0, b.name


def test_registered_as_a_backend(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text("models:\n  - name: ovis\n    engine: ovis-image\n    weights: auto\n",
                 encoding="utf-8")
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends["ovis-image"], OvisImageBackend)
    assert engine._models.config_for("ovis").weights_path == "auto"
