"""No-torch assertions on the Qwen-Image lite build (GGUF + Lightning + NF4
encoder) wiring: Lightning selection, weights defaults, auto resolution, the
build table and the lite two-phase pipeline class (with a fake base class)."""
from __future__ import annotations

import pytest

from imference_engine.qwenimage import QwenImageBackend, builds, lite


def test_lightning_setting(monkeypatch):
    monkeypatch.delenv("QWEN_IMAGE_LIGHTNING", raising=False)
    assert lite.lightning_steps() == 4
    monkeypatch.setenv("QWEN_IMAGE_LIGHTNING", "8")
    assert lite.lightning_steps() == 8
    monkeypatch.setenv("QWEN_IMAGE_LIGHTNING", "none")
    assert lite.lightning_steps() is None


def test_lightning_family_follows_the_checkpoint():
    assert lite.lightning_family("/w/qwen-image-2512-Q4_K_M.gguf") == "2512"
    assert lite.lightning_family("/w/qwen-image-Q4_K_M.gguf") == "base"
    for family, (repo, files) in lite.LIGHTNING.items():
        assert set(files) == {4, 8}, family
        assert repo.startswith("lightx2v/")


def test_lite_base_patterns_skip_the_encoder_weights():
    pats = lite.LITE_BASE_PATTERNS
    assert "text_encoder/config.json" in pats and "text_encoder/*" not in pats
    assert "transformer/config.json" in pats and "transformer/*" not in pats
    assert "vae/*" in pats and "tokenizer/*" in pats


def test_weights_defaults_carry_the_lightning_recipe(monkeypatch):
    be = QwenImageBackend()
    monkeypatch.delenv("QWEN_IMAGE_LIGHTNING", raising=False)
    d = be.weights_defaults("/w/qwen-image-2512-Q4_K_M.gguf")
    assert d.num_steps == 4 and d.guidance_scale == 1.0
    assert be.weights_defaults("auto").num_steps == 4
    monkeypatch.setenv("QWEN_IMAGE_LIGHTNING", "8")
    assert be.weights_defaults("auto").num_steps == 8
    # bf16 single files and Lightning-less GGUFs keep the engine defaults (true CFG)
    assert be.weights_defaults("/w/qwen_image_2512_bf16.safetensors") is None
    monkeypatch.setenv("QWEN_IMAGE_LIGHTNING", "none")
    assert be.weights_defaults("/w/qwen-image-2512-Q4_K_M.gguf") is None


def test_auto_resolves_to_the_2512_gguf(monkeypatch):
    from imference_engine.managers import builds as shared
    from imference_engine.runtime import offline

    seen = {}

    def fake_fetch(repo, filename, cdn_base, cache_dir, *, namespace):
        seen.update(repo=repo, filename=filename, cdn=cdn_base)
        return f"/tree/{repo}/{filename}"

    monkeypatch.setattr(offline, "fetch", fake_fetch)
    monkeypatch.setattr(shared, "cuda_vram_gb", lambda: 16.0)
    be = QwenImageBackend(cdn_base="https://cdn.example/image")
    assert be.resolve_weights("auto") == (
        "/tree/unsloth/Qwen-Image-2512-GGUF/qwen-image-2512-Q4_K_M.gguf")
    assert seen["cdn"] == "https://cdn.example/image"
    assert be.resolve_weights("/w/x.safetensors") == "/w/x.safetensors"


@pytest.mark.parametrize("vram,name,offload", [
    (None, "minimal", "group"),
    (32.0, "resident", None),
    (24.0, "resident", None),
    (20.0, "offload", "model"),
    (15.48, "minimal", "group"),      # what a real 16 GB card reports
    (16.0, "minimal", "group"),
    (12.0, "minimal", "group"),
    (8.0, "minimal", "group"),
])
def test_recommend_build_per_vram_tier(vram, name, offload):
    b = builds.recommend_build(vram)
    assert (b.name, b.offload, b.encoder) == (name, offload, "nf4")
    assert b.filename.endswith(".gguf")


class _FakeBase:
    """Stands in for a diffusers Qwen-Image pipeline class."""

    def __init__(self):
        self.text_encoder = None
        self.calls = []

    _execution_device = "cuda"

    def register_modules(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def _get_qwen_prompt_embeds(self, prompt=None, device=None, dtype=None):
        self.calls.append(("embeds", device))

        class _T:
            def to(self, _):
                return self
        return _T(), _T()

    def prepare_latents(self, *args, **kwargs):
        self.calls.append(("latents", self.text_encoder is not None))
        return "latents"

    def __call__(self, fail=False):
        self._get_qwen_prompt_embeds("p")
        self.prepare_latents()
        if fail:
            raise RuntimeError("boom")
        return "out"


class _Enc:
    device = "cuda"


def test_lite_class_loads_the_encoder_for_encoding_and_releases_it(monkeypatch):
    monkeypatch.setenv("QWEN_EDIT_KEEP_ENCODER", "0")
    cls = lite.lite_pipeline_class(_FakeBase)
    assert cls.__name__ == "_FakeBase"            # no "Pipeline" suffix to rewrite
    pipe = cls()
    loads = []
    pipe._encoder_loader = lambda: loads.append(1) or _Enc()
    assert pipe() == "out"
    assert loads == [1]
    # encoded with the encoder present, denoised without it
    assert pipe.calls == [("embeds", "cuda"), ("latents", False)]
    assert pipe.text_encoder is None


def test_lite_class_keeps_the_encoder_when_asked_and_drops_it_on_failure(monkeypatch):
    monkeypatch.setenv("QWEN_EDIT_KEEP_ENCODER", "1")
    pipe = lite.lite_pipeline_class(_FakeBase)()
    pipe._encoder_loader = _Enc
    pipe()
    assert pipe.text_encoder is not None          # kept across requests
    with pytest.raises(RuntimeError):
        pipe(fail=True)
    assert pipe.text_encoder is None              # never left behind after a failure


def test_img2img_wrapper_always_releases(monkeypatch):
    monkeypatch.setenv("QWEN_EDIT_KEEP_ENCODER", "1")
    pipe = lite.lite_pipeline_class(_FakeBase)()
    pipe._encoder_loader = _Enc
    pipe._always_release = True
    pipe()
    assert pipe.text_encoder is None
