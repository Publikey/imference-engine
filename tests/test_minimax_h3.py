"""No-torch tests for the MiniMax-H3 sub-package + its VideoBackend.

Same posture as test_video_backend.py: constraints math, config resolution,
presets/catalog parsing, the pipe(**call) kwargs the backend builds, loader
delegation (monkeypatched) and teardown. The e2e path needs the unreleased
diffusers PR #14355 + a GPU — that lives in validation/validate_h3.py.
"""
from __future__ import annotations

import pytest

from imference_engine.core.result import MediaResult
from imference_engine.minimax_h3 import (BUILTIN_VARIANTS, H3MemoryProfile,
                                         H3Variant, MiniMaxH3Engine,
                                         MiniMaxH3RuntimeConfig, OFFICIAL_REPO)
from imference_engine.minimax_h3.constraints import (align_num_frames,
                                                     check_canvas,
                                                     check_num_frames)
from imference_engine.video import (KNOWN_VIDEO_ARCHS, MiniMaxH3Backend,
                                    VideoBackend, VideoBuildContext)


# ------------------------------------------------------------------
# Constraints (mirrors of the upstream 17n+5 / 5-15 s / mod-32 rules)
# ------------------------------------------------------------------

def test_align_num_frames_grid():
    assert align_num_frames(124) == 124        # already aligned
    assert align_num_frames(120) == 124        # snaps UP
    assert align_num_frames(125) == 141        # next 17n+5
    assert align_num_frames(5) == 5            # smallest aligned count
    with pytest.raises(ValueError):
        align_num_frames(0)


def test_check_num_frames_duration_window():
    assert check_num_frames(124) == 124        # 5.17 s — ok
    assert check_num_frames(345) == 345        # 14.375 s — max aligned count
    with pytest.raises(ValueError, match="15"):
        check_num_frames(346)                  # aligns to 362 = 15.083 s (upstream doc case)
    with pytest.raises(ValueError, match="5"):
        check_num_frames(60)                   # aligns to 73 = 3.04 s, under the floor


def test_check_canvas():
    check_canvas(None, None)                   # model-native canvas
    check_canvas(960, 544)                     # the fast doc'd canvas
    with pytest.raises(ValueError, match="together"):
        check_canvas(960, None)
    with pytest.raises(ValueError, match="32"):
        check_canvas(960, 540)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

def test_profile_auto_is_int8():
    assert H3MemoryProfile.for_hardware(24.0) is H3MemoryProfile.INT8
    assert H3MemoryProfile.for_hardware(80.0) is H3MemoryProfile.INT8  # bf16 = explicit only


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("H3_PROFILE", "int8")
    monkeypatch.setenv("H3_OFFLOAD_MODE", "leaf")
    monkeypatch.setenv("H3_MODEL_CDN", "https://cdn.x/video")
    monkeypatch.setenv("H3_VAE_TILING", "0")
    monkeypatch.setenv("H3_MAX_RESIDENT", "2")
    cfg = MiniMaxH3RuntimeConfig.from_env()
    assert cfg.memory_profile == "int8"
    assert cfg.offload_mode == "leaf"
    assert cfg.model_cdn == "https://cdn.x/video"
    assert cfg.vae_tiling is False
    assert cfg.max_resident_variants == 2
    assert cfg.device == "auto"                # default untouched


def test_engine_resolves_profile_and_offload_strings():
    assert MiniMaxH3Engine._resolve_profile("auto") is H3MemoryProfile.INT8
    assert MiniMaxH3Engine._resolve_profile("bf16") is H3MemoryProfile.BF16
    assert MiniMaxH3Engine._resolve_profile(H3MemoryProfile.INT8) is H3MemoryProfile.INT8
    with pytest.raises(ValueError):
        MiniMaxH3Engine._resolve_profile("int4")   # not a profile (convrot is a V2 story)


# ------------------------------------------------------------------
# Presets + catalog
# ------------------------------------------------------------------

def test_builtin_variant():
    v = BUILTIN_VARIANTS["minimax-h3"]
    assert v.repo == OFFICIAL_REPO and v.arch == "minimax_h3"
    assert v.num_steps >= 2


def test_variant_validation():
    with pytest.raises(ValueError, match="repo"):
        H3Variant(name="x", repo="")
    with pytest.raises(ValueError, match="num_steps"):
        H3Variant(name="x", num_steps=1)


_SHARED_CATALOG = """
models:
  - name: h3-int8
    kind: video
    engine: minimax_h3
    mode: t2v
    repo: my-org/MiniMax-H3-int8
    num_steps: 40
  - name: my-wan
    kind: video
    engine: wan
    mode: t2v
    base_repo: Wan-AI/Wan2.2-T2V-A14B-Diffusers
    gguf_repo: QuantStack/Wan2.2-T2V-A14B-GGUF
"""


def test_catalog_row_to_variant():
    from imference_engine.catalog.loader import loads_video
    from imference_engine.minimax_h3.presets import variant_from_catalog
    rows = loads_video(_SHARED_CATALOG, known_archs=KNOWN_VIDEO_ARCHS)
    h3 = [r for r in rows if r.arch == "minimax_h3"]
    assert len(h3) == 1 and len(rows) == 2     # wan row coexists in the shared file
    v = variant_from_catalog(h3[0])
    assert v.name == "h3-int8" and v.repo == "my-org/MiniMax-H3-int8"
    assert v.num_steps == 40


def test_engine_catalog_registers_only_h3_rows(tmp_path):
    path = tmp_path / "models.yml"
    path.write_text(_SHARED_CATALOG, encoding="utf-8")
    engine = MiniMaxH3Engine(catalog_path=path).load()   # cpu fallback, no torch
    assert "h3-int8" in engine.list_variants()
    assert "my-wan" not in engine.list_variants()
    assert "minimax-h3" in engine.list_variants()        # builtin kept


def test_wan_engine_tolerates_h3_rows_in_shared_catalog(tmp_path):
    # The counterpart guarantee: WanEngine must not choke on minimax_h3 rows.
    from imference_engine.wan.engine import WanEngine
    path = tmp_path / "models.yml"
    path.write_text(_SHARED_CATALOG, encoding="utf-8")
    engine = WanEngine(catalog_path=path)
    engine._backends = {"wan": object()}                  # _setup needs a GPU; fake the registry
    engine._register_catalog(path)
    assert "my-wan" in engine.list_variants()
    assert "h3-int8" not in engine.list_variants()


# ------------------------------------------------------------------
# Backend
# ------------------------------------------------------------------

def _ctx(**over):
    base = dict(quant="int8", device="cuda:0", enable_offload=True, vae_tiling=True,
                cache_dir="/cache", cdn_base="https://cdn.x/video",
                offload_mode="block", attention_backend=None)
    base.update(over)
    return VideoBuildContext(**base)


def test_backend_identity():
    assert MiniMaxH3Backend.engine == "minimax_h3"
    assert isinstance(MiniMaxH3Backend(), VideoBackend)
    assert "minimax_h3" in KNOWN_VIDEO_ARCHS and "wan" in KNOWN_VIDEO_ARCHS


def test_backend_no_shared_components():
    assert MiniMaxH3Backend().load_shared(_ctx()) is None


def test_build_call_t2va():
    call = MiniMaxH3Backend().build_call(
        prompt="a fox", negative_prompt=None, width=None, height=None,
        num_frames=124, num_steps=50, guidance_scale=None, guidance_scale_2=None,
        image=None, generator="G")
    assert call["prompt"] == "a fox"
    assert call["num_frames"] == 124 and call["num_inference_steps"] == 50
    assert call["output_type"] == "pil" and call["generator"] == "G"
    # None canvas -> pipeline derives the native one; ignored knobs never leak.
    for absent in ("width", "height", "image", "last_image",
                   "negative_prompt", "guidance_scale"):
        assert absent not in call


def test_build_call_fl2va_keyframes_pass_through():
    # Keyframes are NOT pre-resized — the setup block owns canvas placement.
    call = MiniMaxH3Backend().build_call(
        prompt="p", negative_prompt="bad", width=960, height=544,
        num_frames=141, num_steps=40, guidance_scale=5.0, guidance_scale_2=None,
        image="IMG", generator="G", last_image="LAST")
    assert call["image"] == "IMG" and call["last_image"] == "LAST"
    assert call["width"] == 960 and call["height"] == 544
    assert "negative_prompt" not in call and "guidance_scale" not in call


def test_build_delegates_to_loader(monkeypatch):
    from imference_engine.minimax_h3 import loader as L
    seen = {}
    monkeypatch.setattr(L, "build_pipeline",
                        lambda spec, **kw: seen.update(kw=kw, spec=spec) or "PIPE")
    var = H3Variant(name="v")
    assert MiniMaxH3Backend().build(var, None, _ctx(offload_mode=None)) == "PIPE"
    kw = seen["kw"]
    assert seen["spec"] is var
    assert kw["profile"] == "int8" and kw["device"] == "cuda:0"
    assert kw["offload_mode"] == "block"       # None -> the backend default
    assert kw["cdn_base"] == "https://cdn.x/video" and kw["cache_dir"] == "/cache"


def test_teardown_drops_big_modules_and_hooks():
    class FakePipe:
        def __init__(self):
            self.transformer = object()
            self.text_encoder = object()
            self.vae = object()
            self.audio_vae = object()
            self.hooks_removed = False
        def remove_all_hooks(self):
            self.hooks_removed = True

    p = FakePipe()
    MiniMaxH3Backend().teardown(p)
    assert p.hooks_removed is True
    assert p.transformer is None and p.text_encoder is None
    assert p.vae is None and p.audio_vae is None


# ------------------------------------------------------------------
# Engine surface + result
# ------------------------------------------------------------------

def test_engine_guards():
    engine = MiniMaxH3Engine()
    with pytest.raises(RuntimeError, match="load"):
        engine.generate_video(prompt="p")
    engine.load()
    with pytest.raises(KeyError, match="unknown variant"):
        engine.generate_video(prompt="p", variant="nope")


def test_generate_constraint_failure_returns_errors_not_raise():
    # Constraint violations surface as MediaResult errors (the generate_video
    # never-raises contract) — checked BEFORE any torch import or model load.
    engine = MiniMaxH3Engine().load()
    res = engine.generate_video(prompt="p", num_frames=400, seed=7)
    assert not res.ok and res.error is not None
    assert "15" in res.error.error and res.seeds == [7]
    res2 = engine.generate_video(prompt="p", width=100, height=100)
    assert not res2.ok and "32" in res2.error.error


def test_media_result_audio_fields():
    r = MediaResult(kind="video", media=[], seeds=[1])
    assert r.audio is None and r.sample_rate is None     # video-only backends unaffected
    r2 = MediaResult(kind="video", media=["f"], seeds=[1],
                     audio="WAVE", sample_rate=48000)
    assert r2.audio == "WAVE" and r2.sample_rate == 48000 and r2.ok
