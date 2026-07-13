"""No-network guards on the R2 staging tool (validation/stage_r2.py).

Locks the two contracts that silently break CDN serving if they drift:
  - the object-key layout the reader (offline.py::_cdn_snapshot) expects, and
  - per-engine base resolution (which repo + which patterns get staged), read
    from the backend classes so it can't diverge from the load path.
No torch, no boto3, no network.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "stage_r2", Path(__file__).resolve().parent.parent / "validation" / "stage_r2.py")
stage_r2 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage_r2)


def test_object_key_layout():
    # <prefix>/<repo>/<rel> — no leading slash, prefix slashes trimmed.
    assert stage_r2.object_key("", "a/b", "vae/config.json") == "a/b/vae/config.json"
    assert stage_r2.object_key("models", "a/b", "x.json") == "models/a/b/x.json"
    assert stage_r2.object_key("/models/", "a/b", "x.json") == "models/a/b/x.json"


def test_wan_gguf_key_matches_cdn_layout():
    # Wan reads GGUF at <WAN_MODEL_CDN>/<repo>/<file>; the staged key must match
    # (prefix = the WAN_MODEL_CDN tail, e.g. "wan22").
    key = stage_r2.object_key(
        "wan22", "bullerwins/Wan2.2-I2V-A14B-GGUF",
        "wan2.2_i2v_high_noise_14B_Q8_0.gguf")
    assert key == "wan22/bullerwins/Wan2.2-I2V-A14B-GGUF/wan2.2_i2v_high_noise_14B_Q8_0.gguf"


def test_resolve_transformer_base_uses_backend_patterns():
    from imference_engine.flux import FluxBackend
    repo, patterns = stage_r2.resolve_base(
        "flux", {"mode": "single_file", "base_model": "black-forest-labs/FLUX.1-dev"})
    assert repo == "black-forest-labs/FLUX.1-dev"
    assert patterns == list(FluxBackend.BASE_PATTERNS)
    # never drags the base transformer weights the checkpoint replaces
    assert "transformer/*" not in patterns


def test_resolve_sd15_falls_back_to_config_repo():
    from imference_engine.pipelines.sd15 import SD15Backend
    repo, patterns = stage_r2.resolve_base("sd15", {"mode": "single_file", "base_model": None})
    assert repo == SD15Backend.CONFIG_REPO
    assert patterns == list(SD15Backend.CONFIG_PATTERNS)


def test_resolve_repo_mode_stages_whole_repo():
    repo, patterns = stage_r2.resolve_base(
        "anima", {"mode": "repo", "repo": "circlestone-labs/Anima-Base-v1.0-Diffusers"})
    assert repo == "circlestone-labs/Anima-Base-v1.0-Diffusers"
    assert patterns is None  # all files


def test_default_engines_are_all_cdn_wired():
    # anima is now wired through local_repo_dir (see AnimaBackend.load_pipeline),
    # so it's staged by default alongside the transformer-only backends.
    assert set(stage_r2.DEFAULT_ENGINES) == {"flux", "chroma", "sd15", "qwenimage", "anima"}


def test_anima_backend_is_cdn_wired():
    # Guard the wiring the staging tool relies on: a repo-id weights_path must be
    # resolved into the flat tree (local_repo_dir) before ModularPipeline load.
    import inspect

    from imference_engine.anima import AnimaBackend
    src = inspect.getsource(AnimaBackend.load_pipeline)
    assert "local_repo_dir" in src
    assert AnimaBackend.BASE_PATTERNS == ["*"]  # whole modular repo
    assert AnimaBackend.SENTINEL == "modular_model_index.json"
