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
