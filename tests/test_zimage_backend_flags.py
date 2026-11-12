"""No-torch assertions on the Z-Image backend's offline base-component mirror.

Guards the offline contract: from_single_file(config=base_dir) needs the
transformer + scheduler *configs* in the mirrored tree (the transformer weights
come from the checkpoint). Omitting transformer/config.json made diffusers fall
back to a HF snapshot_download — crashing under HF_HUB_OFFLINE=1.
"""
from imference_engine.zimage import ZImageBackend


def test_base_patterns_include_transformer_and_scheduler_configs():
    pats = ZImageBackend.BASE_PATTERNS
    assert "model_index.json" in pats
    # shared components = full weights
    assert "tokenizer/*" in pats and "text_encoder/*" in pats and "vae/*" in pats
    # transformer + scheduler = config only (layout for from_single_file)
    assert "transformer/config.json" in pats
    assert "scheduler/*" in pats


def test_base_patterns_never_pull_transformer_weights():
    """The transformer .safetensors come from the checkpoint, not the base repo —
    mirroring them would bloat the CDN by GBs."""
    pats = ZImageBackend.BASE_PATTERNS
    assert "transformer/*" not in pats  # would grab the weights
    assert not any(p.startswith("transformer/") and "safetensors" in p for p in pats)


def test_base_checkpoint_is_a_catalog_row(tmp_path):
    """Z-Image base (undistilled) needs no code path: its true-CFG recipe is the
    row's defaults, and guidance + negative prompt already reach the pipeline."""
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text(
        "models:\n"
        "  - name: z-image-base\n"
        "    engine: zimage\n"
        "    weights: /cache/z_image_bf16.safetensors\n"
        "    base_model: Tongyi-MAI/Z-Image\n"
        "    defaults: {num_steps: 40, guidance_scale: 4.0}\n"
        encoding="utf-8",
    )
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    meta = engine._models.config_for("z-image-base")
    assert meta.base_model == "Tongyi-MAI/Z-Image"
    assert meta.defaults.num_steps == 40 and meta.defaults.guidance_scale == 4.0
    assert meta.defaults.backend_options == {}      # no Turbo shift carried over

    be = ZImageBackend()
    assert be.encode_prompts(None, "a fox", "snow") == {"prompt": "a fox", "negative_prompt": "snow"}
    kw = be.build_inference_kwargs(
        width=1024, height=1024, num_steps=40, guidance_scale=4.0,
        clip_skip=None, chunk_size=1, generator=None)
    assert kw["guidance_scale"] == 4.0 and kw["num_inference_steps"] == 40

