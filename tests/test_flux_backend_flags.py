"""No-torch assertions on the FLUX backend: engine defaults, the offline
base-component mirror, and registration through the Engine.

Guards the offline contract (from_single_file(config=base_dir/transformer) needs
the transformer config in the mirrored tree, NOT its weights) and the
guidance-distilled family defaults.
"""
from __future__ import annotations

from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.flux import FluxBackend


def test_engine_key():
    assert FluxBackend.engine == "flux"


def test_engine_defaults_are_guidance_distilled_family_norm():
    d = FluxBackend().engine_defaults()
    assert d.guidance_scale == 3.5   # dev embedded-guidance norm
    assert d.num_steps == 28
    assert d.scheduler is None        # flow matching, name ignored
    assert d.backend_options == {}    # shift is per-model, not engine


def test_base_patterns_include_both_text_encoders_and_vae():
    pats = FluxBackend.BASE_PATTERNS
    assert "model_index.json" in pats and "scheduler/*" in pats
    # CLIP-L + T5-XXL, both tokenizers, VAE = full weights
    assert "text_encoder/*" in pats and "text_encoder_2/*" in pats
    assert "tokenizer/*" in pats and "tokenizer_2/*" in pats
    assert "vae/*" in pats
    # transformer = config only (layout for from_single_file)
    assert "transformer/config.json" in pats


def test_base_patterns_never_pull_transformer_weights():
    """The ~24 GB transformer weights come from the checkpoint, not the base."""
    pats = FluxBackend.BASE_PATTERNS
    assert "transformer/*" not in pats
    assert not any(p.startswith("transformer/") and "safetensors" in p for p in pats)


def test_encode_prompts_drops_negative():
    """Guidance-distilled: negative prompt is ignored, positive passed through."""
    be = FluxBackend()
    out = be.encode_prompts(pipe=None, prompt="a cat", negative_prompt="blurry")
    assert out == {"prompt": "a cat"}


def test_build_inference_kwargs_t2i_and_img2img():
    be = FluxBackend()
    t2i = be.build_inference_kwargs(
        width=1024, height=768, num_steps=28, guidance_scale=3.5,
        clip_skip=None, chunk_size=2, generator=None)
    assert t2i["width"] == 1024 and t2i["height"] == 768
    assert t2i["num_images_per_prompt"] == 2
    assert t2i["max_sequence_length"] == FluxBackend.MAX_SEQUENCE_LENGTH
    assert "clip_skip" not in t2i  # FLUX has none

    i2i = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=28, guidance_scale=3.5,
        clip_skip=None, chunk_size=1, generator=None,
        image=object(), strength=0.6)
    assert "image" in i2i and i2i["strength"] == 0.6
    assert "width" not in i2i and "height" not in i2i  # size from source image


def test_registered_as_a_backend():
    from imference_engine import Engine, RuntimeConfig
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert "flux" in engine._backends
    assert isinstance(engine._backends["flux"], FluxBackend)


def test_flux_catalog_row_carries_schnell_style_defaults(tmp_path):
    """A schnell-style checkpoint expresses its 4-step recipe as MODEL defaults;
    the engine layer stays dev-oriented (guidance 3.5)."""
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text(
        "models:\n"
        "  - name: flux-schnell\n"
        "    engine: flux\n"
        "    weights: /cache/flux-schnell.safetensors\n"
        "    base_model: black-forest-labs/FLUX.1-schnell\n"
        "    defaults: {num_steps: 4}\n",
        encoding="utf-8",
    )
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    meta = engine._models.config_for("flux-schnell")
    assert meta.backend == "flux"
    assert meta.base_model == "black-forest-labs/FLUX.1-schnell"
    assert meta.defaults.num_steps == 4
    # engine-layer guidance 3.5 still applies (model didn't override it)
    assert isinstance(meta.defaults, GenerationDefaults)
