"""No-torch assertions on the HiDream-I1 backend: per-checkpoint recipes, the
base / Llama repo split, inference kwargs, the build table and registration."""
from __future__ import annotations

import pytest

from imference_engine.hidream import AUTO, BUILDS, HiDreamBackend, recommend_build
from imference_engine.hidream import backend as mod


def test_engine_key_and_fast_defaults():
    be = HiDreamBackend()
    assert be.engine == "hidream"
    d = be.engine_defaults()
    assert d.num_steps == 16 and d.guidance_scale == 0.0
    assert be.supports_loras is False


def test_dev_and_full_files_carry_their_recipe():
    be = HiDreamBackend()
    assert be.weights_defaults(AUTO) is None
    assert be.weights_defaults("/w/hidream-i1-fast-Q4_K_M.gguf") is None
    dev = be.weights_defaults("/w/hidream-i1-dev-Q4_K_M.gguf")
    assert dev.num_steps == 28 and dev.guidance_scale == 0.0
    full = be.weights_defaults("/w/hidream_i1_full_fp16.safetensors")
    assert full.num_steps == 50 and full.guidance_scale == 5.0


def test_base_patterns_and_the_separate_llama_repo(monkeypatch):
    pats = HiDreamBackend.BASE_PATTERNS
    for p in ("model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
              "text_encoder_3/*", "tokenizer_3/*", "vae/*", "transformer/config.json"):
        assert p in pats
    assert "transformer/*" not in pats
    assert not any("text_encoder_4" in p for p in pats)      # Llama is another repo
    llama = HiDreamBackend.LLAMA_PATTERNS
    assert "model-*.safetensors" in llama and "tokenizer.json" in llama
    assert any(p.startswith("LICENSE") for p in llama) and any(p.startswith("USE_POLICY") for p in llama)
    assert "original/*" not in llama and "*" not in llama     # never the 16 GB .pth copy

    monkeypatch.delenv("HIDREAM_LLAMA_REPO", raising=False)
    assert mod.llama_repo() == "unsloth/Meta-Llama-3.1-8B-Instruct"
    monkeypatch.setenv("HIDREAM_LLAMA_REPO", "/models/llama")
    assert mod.llama_repo() == "/models/llama"


def test_kwargs_and_prompts():
    be = HiDreamBackend()
    kw = be.build_inference_kwargs(
        width=1024, height=768, num_steps=16, guidance_scale=0.0,
        clip_skip=None, chunk_size=2, generator=None)
    assert kw == {"num_inference_steps": 16, "guidance_scale": 0.0, "width": 1024,
                  "height": 768, "generator": None, "num_images_per_prompt": 2}
    assert be.encode_prompts(None, "a fox", None) == {"prompt": "a fox"}
    assert be.encode_prompts(None, "a fox", "blurry") == {"prompt": "a fox", "negative_prompt": "blurry"}
    with pytest.raises(NotImplementedError):
        be.make_img2img(object())


def test_build_table_is_gguf_with_offload_everywhere():
    floors = [b.min_vram_gb for b in BUILDS]
    assert floors == sorted(floors, reverse=True) and BUILDS[-1].min_vram_gb == 0.0
    for b in BUILDS:
        assert b.filename.endswith(".gguf") and b.offload in ("model", "group")
        assert b.encoder == "fp8"
    assert recommend_build(24.0).filename.endswith("Q8_0.gguf")
    assert recommend_build(16.0).filename.endswith("Q4_K_M.gguf")
    assert recommend_build(8.0).offload == "group"
    none = recommend_build(None)
    assert none.encoder == "bf16" and none.offload == "model"


def test_auto_resolves_to_the_recommended_gguf(monkeypatch):
    from imference_engine.runtime import offline

    monkeypatch.setattr(offline, "fetch",
                        lambda repo, fn, cdn, cache, *, namespace: f"/tree/{repo}/{fn}")
    monkeypatch.setattr(mod, "_vram_gb", lambda: 16.0)
    be = HiDreamBackend(cdn_base="https://cdn.example/image")
    assert be.resolve_weights(AUTO) == "/tree/city96/HiDream-I1-Fast-gguf/hidream-i1-fast-Q4_K_M.gguf"
    assert be.resolve_weights("/w/x.gguf") == "/w/x.gguf"


def test_registered_as_a_backend(tmp_path):
    from imference_engine import Engine, RuntimeConfig

    f = tmp_path / "models.yml"
    f.write_text("models:\n  - name: hidream\n    engine: hidream\n    weights: auto\n",
                 encoding="utf-8")
    engine = Engine(catalog_path=f, runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends["hidream"], HiDreamBackend)
