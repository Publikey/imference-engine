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


def test_anima_not_in_default_engines():
    # anima isn't CDN-wired yet -> must not be staged by default.
    assert "anima" not in stage_r2.DEFAULT_ENGINES
    assert set(stage_r2.DEFAULT_ENGINES) == {"flux", "chroma", "sd15", "qwenimage"}
