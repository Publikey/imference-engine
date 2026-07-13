"""Tests for the catalog YAML loader (catalog/loader.py).

Parsing is pure (no torch): valid docs produce ModelConfig with populated
per-model defaults; malformed docs raise CatalogError with useful context. Also
covers the Engine wiring (catalog_path -> registered models + defaults) using the
real 'sdxl' backend key so registration succeeds without loading weights.
"""
from __future__ import annotations

import pytest

from imference_engine.catalog.loader import CatalogError, ModelConfig, load, loads

VALID = """
version: 1
models:
  - name: z-image-turbo
    engine: zimage
    weights: /cache/z.safetensors
    base_model: Tongyi-MAI/Z-Image-Turbo
    defaults:
      num_steps: 8
      guidance_scale: 1.0
      backend_options:
        shift: 3.0
  - name: dreamshaper-xl
    engine: sdxl
    weights: /cache/dreamshaper.safetensors
    defaults:
      scheduler: DPMSolverMultistepScheduler
      clip_skip: 2
"""


def test_valid_catalog_parses():
    configs = loads(VALID)
    assert [c.name for c in configs] == ["z-image-turbo", "dreamshaper-xl"]

    z = configs[0]
    assert isinstance(z, ModelConfig)
    assert z.engine == "zimage"
    assert z.weights == "/cache/z.safetensors"
    assert z.base_model == "Tongyi-MAI/Z-Image-Turbo"
    assert z.defaults.num_steps == 8
    assert z.defaults.guidance_scale == 1.0
    assert z.defaults.backend_options == {"shift": 3.0}

    d = configs[1]
    assert d.base_model is None
    assert d.defaults.scheduler == "DPMSolverMultistepScheduler"
    assert d.defaults.clip_skip == 2
    assert d.defaults.backend_options == {}


def test_empty_document_is_empty_list():
    assert loads("") == []
    assert loads("version: 1\nmodels: []") == []


def test_known_engines_accepts_registered():
    configs = loads(VALID, known_engines={"sdxl", "zimage"})
    assert len(configs) == 2


def test_unknown_engine_rejected():
    with pytest.raises(CatalogError, match="unknown engine 'flux'"):
        loads(
            "models:\n  - {name: x, engine: flux, weights: /w}",
            known_engines={"sdxl", "zimage"},
        )


def test_missing_models_key():
    with pytest.raises(CatalogError, match="missing required 'models'"):
        loads("version: 1")


def test_models_not_a_list():
    with pytest.raises(CatalogError, match="'models' must be a list"):
        loads("models: {a: 1}")


def test_missing_required_field():
    with pytest.raises(CatalogError, match="missing required 'weights'"):
        loads("models:\n  - {name: x, engine: sdxl}")


def test_duplicate_name_rejected():
    doc = """
models:
  - {name: dup, engine: sdxl, weights: /a}
  - {name: dup, engine: sdxl, weights: /b}
"""
    with pytest.raises(CatalogError, match="duplicate model name 'dup'"):
        loads(doc)


def test_unknown_defaults_key_rejected():
    """A typo like 'stpes' must fail loudly, not be silently dropped."""
    doc = "models:\n  - {name: x, engine: sdxl, weights: /w, defaults: {stpes: 8}}"
    with pytest.raises(CatalogError, match="unknown defaults key"):
        loads(doc)


def test_unknown_top_level_key_rejected():
    doc = "models:\n  - {name: x, engine: sdxl, weights: /w, base: oops}"
    with pytest.raises(CatalogError, match="unknown key"):
        loads(doc)


def test_backend_options_must_be_mapping():
    doc = "models:\n  - {name: x, engine: sdxl, weights: /w, defaults: {backend_options: 3}}"
    with pytest.raises(CatalogError, match="backend_options must be a mapping"):
        loads(doc)


def test_non_numeric_default_rejected():
    doc = "models:\n  - {name: x, engine: sdxl, weights: /w, defaults: {num_steps: abc}}"
    with pytest.raises(CatalogError, match="num_steps must be int"):
        loads(doc)


def test_bool_not_accepted_as_int():
    doc = "models:\n  - {name: x, engine: sdxl, weights: /w, defaults: {num_steps: true}}"
    with pytest.raises(CatalogError, match="num_steps must be int, got bool"):
        loads(doc)


def test_quoted_numeric_is_coerced():
    doc = "models:\n  - {name: x, engine: sdxl, weights: /w, defaults: {num_steps: '8', guidance_scale: '6.5'}}"
    c = loads(doc)[0]
    assert c.defaults.num_steps == 8
    assert c.defaults.guidance_scale == 6.5


def test_load_from_file(tmp_path):
    f = tmp_path / "models.yml"
    f.write_text(VALID, encoding="utf-8")
    configs = load(f)
    assert [c.name for c in configs] == ["z-image-turbo", "dreamshaper-xl"]


def test_load_missing_file_raises():
    with pytest.raises(CatalogError, match="cannot read catalog"):
        load("/nonexistent/models.yml")


# --------------------------------------------------------------------------
# Engine wiring
# --------------------------------------------------------------------------

def test_engine_loads_catalog_at_load(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text(
        "models:\n"
        "  - name: dreamshaper-xl\n"
        "    engine: sdxl\n"
        "    weights: /cache/dreamshaper.safetensors\n"
        "    defaults: {num_steps: 8, clip_skip: 2}\n",
        encoding="utf-8",
    )
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    meta = engine._models.config_for("dreamshaper-xl")
    assert meta.backend == "sdxl"
    assert meta.weights_path == "/cache/dreamshaper.safetensors"
    assert meta.defaults.num_steps == 8
    assert meta.defaults.clip_skip == 2


def test_engine_load_catalog_rejects_unknown_engine(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "bad.yml"
    f.write_text("models:\n  - {name: x, engine: nope, weights: /w}", encoding="utf-8")
    with pytest.raises(CatalogError, match="unknown engine 'nope'"):
        Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
