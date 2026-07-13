"""Phase 5 — unified catalog: `kind: image|video` in one models.yml.

load() reads image rows, load_video() reads video rows; the Wan side converts a
VideoModelConfig to a WanVariant. A single file feeds both engines.
"""
from __future__ import annotations

import pytest

from imference_engine.catalog.loader import (CatalogError, VideoModelConfig,
                                             loads, loads_video)

MIXED = """
version: 1
models:
  - name: dreamshaper-xl
    kind: image
    engine: sdxl
    weights: /cache/dreamshaper.safetensors
    defaults: {clip_skip: 2}
  - name: sdxl-default-kind
    engine: sdxl
    weights: /cache/x.safetensors
  - name: my-wan-t2v
    kind: video
    engine: wan
    mode: t2v
    base_repo: Wan-AI/Wan2.2-T2V-A14B-Diffusers
    gguf_repo: QuantStack/Wan2.2-T2V-A14B-GGUF
    flow_shift: 3.0
    gguf_high_template: HighNoise/foo-{quant}.gguf
    gguf_low_template: LowNoise/foo-{quant}.gguf
    loras:
      - {repo: lightx2v/W, high_weight_name: h.safetensors, low_weight_name: l.safetensors}
"""


def test_image_load_ignores_video_rows():
    imgs = loads(MIXED)
    assert [m.name for m in imgs] == ["dreamshaper-xl", "sdxl-default-kind"]  # kind default = image
    assert imgs[0].defaults.clip_skip == 2


def test_video_load_ignores_image_rows():
    vids = loads_video(MIXED)
    assert len(vids) == 1
    v = vids[0]
    assert isinstance(v, VideoModelConfig)
    assert v.name == "my-wan-t2v" and v.arch == "wan" and v.mode == "t2v"
    assert v.spec["base_repo"].endswith("T2V-A14B-Diffusers")
    assert v.spec["flow_shift"] == 3.0
    assert v.spec["loras"][0]["repo"] == "lightx2v/W"


def test_video_unknown_arch_rejected():
    doc = "models:\n  - {name: x, kind: video, engine: ltx, mode: t2v}"
    with pytest.raises(CatalogError, match="unknown video arch 'ltx'"):
        loads_video(doc, known_archs={"wan"})


def test_video_bad_mode_rejected():
    doc = "models:\n  - {name: x, kind: video, engine: wan, mode: zzz}"
    with pytest.raises(CatalogError, match="'mode' must be t2v|i2v"):
        loads_video(doc)


def test_bad_kind_rejected():
    doc = "models:\n  - {name: x, kind: audio, engine: wan}"
    with pytest.raises(CatalogError, match="'kind' must be image|video"):
        loads(doc)


def test_variant_from_catalog_builds_wanvariant():
    from imference_engine.wan.presets import variant_from_catalog
    v = loads_video(MIXED)[0]
    var = variant_from_catalog(v)
    assert var.name == "my-wan-t2v" and var.mode == "t2v" and var.arch == "wan"
    assert var.gguf_repo.endswith("T2V-A14B-GGUF")
    assert var.flow_shift == 3.0
    assert len(var.loras) == 1 and var.loras[0].repo == "lightx2v/W"


def test_variant_from_catalog_requires_gguf_repo():
    from imference_engine.wan.presets import variant_from_catalog
    v = VideoModelConfig(name="x", arch="wan", mode="t2v",
                         spec={"base_repo": "b"})  # no gguf_repo
    with pytest.raises(CatalogError, match="missing 'gguf_repo'"):
        variant_from_catalog(v)


def test_wan_engine_loads_video_catalog(tmp_path):
    from imference_engine.wan import WanEngine, WanRuntimeConfig
    f = tmp_path / "models.yml"
    f.write_text(MIXED, encoding="utf-8")
    eng = WanEngine(catalog_path=f, runtime=WanRuntimeConfig(device="cpu")).load()
    assert "my-wan-t2v" in eng.list_variants()          # catalog variant registered
    assert "wan22-t2v-lightning" in eng.list_variants()  # built-ins still present
    var = eng._variants["my-wan-t2v"]
    assert var.mode == "t2v" and var.gguf_repo.endswith("T2V-A14B-GGUF")
