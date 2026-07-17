"""No-torch assertions on the Anima backend (Modular Diffusers).

Anima is the one backend not on the standard pipeline API, so the guarantees
differ: img2img is unsupported, and it carries no opinionated engine defaults.
"""
from __future__ import annotations

import pytest

from imference_engine.anima import AnimaBackend


def test_engine_key():
    assert AnimaBackend.engine == "anima"


def test_no_engine_defaults_opinion():
    """Anima's recommended steps/guidance aren't documented → no opinion; the
    request/GLOBAL_DEFAULTS drive."""
    d = AnimaBackend().engine_defaults()
    assert d.num_steps is None
    assert d.guidance_scale is None
    assert d.scheduler is None
    assert d.backend_options == {}


def test_img2img_unsupported():
    with pytest.raises(NotImplementedError, match="no documented img2img"):
        AnimaBackend().make_img2img(object())


def test_encode_prompts_omits_empty_negative():
    """Only pass negative_prompt when provided (the doc example omits it)."""
    be = AnimaBackend()
    assert be.encode_prompts(pipe=None, prompt="a girl", negative_prompt=None) == {
        "prompt": "a girl"}
    assert be.encode_prompts(pipe=None, prompt="a girl", negative_prompt="") == {
        "prompt": "a girl"}
    assert be.encode_prompts(pipe=None, prompt="a girl", negative_prompt="blurry") == {
        "prompt": "a girl", "negative_prompt": "blurry"}


def test_build_inference_kwargs_standard_set():
    kw = AnimaBackend().build_inference_kwargs(
        width=1024, height=1024, num_steps=28, guidance_scale=6.0,
        clip_skip=None, chunk_size=1, generator=None)
    assert kw["num_inference_steps"] == 28
    assert kw["num_images_per_prompt"] == 1
    assert kw["width"] == 1024 and kw["height"] == 1024
    assert "clip_skip" not in kw  # Anima has none
    # Anima's modular __call__ ignores guidance_scale (separate Guider block) —
    # forwarding it triggers a diffusers "Unexpected input" warning, so it must
    # NOT be in the kwargs.
    assert "guidance_scale" not in kw


class _FakeSpec:
    def __init__(self, path, subfolder="", method="from_pretrained"):
        self.pretrained_model_name_or_path = path
        self.subfolder = subfolder
        self.default_creation_method = method


class _FakePipe:
    """Stands in for a ModularPipeline: specs name a HUB REPO ID (as
    modular_model_index.json does), and load_components only populates the
    components whose resolved path is a real local dir — mirroring diffusers,
    which warns rather than raises on a failed component."""

    def __init__(self, specs):
        self._specs = specs
        for name in specs:
            setattr(self, name, None)
        self.load_kwargs = None

    @property
    def null_component_names(self):
        return [n for n in self._specs if getattr(self, n, None) is None]

    def get_component_spec(self, name):
        return self._specs[name]

    def load_components(self, names=None, **kwargs):
        import os
        self.load_kwargs = kwargs
        paths = kwargs.get("pretrained_model_name_or_path") or {}
        for name in names:
            src = paths.get(name, self._specs[name].pretrained_model_name_or_path)
            if os.path.isdir(src):
                setattr(self, name, f"loaded:{name}")


def test_load_components_repoints_hub_repo_ids_at_the_local_tree(tmp_path):
    """modular_model_index.json names each component by hub repo id, so a pipe
    built from a local dir still points at huggingface.co — fatal under
    HF_HUB_OFFLINE. The specs must be repointed at the mirrored tree."""
    pytest.importorskip("torch")
    for sub in ("text_encoder", "vae", "scheduler"):
        (tmp_path / sub).mkdir()
    pipe = _FakePipe({
        "text_encoder": _FakeSpec("circlestone-labs/Anima-Base-v1.0-Diffusers",
                                  "text_encoder"),
        "vae": _FakeSpec("circlestone-labs/Anima-Base-v1.0-Diffusers", "vae"),
        "scheduler": _FakeSpec("circlestone-labs/Anima-Base-v1.0-Diffusers",
                               "scheduler"),
    })

    AnimaBackend()._load_components(pipe, str(tmp_path))

    assert pipe.load_kwargs["pretrained_model_name_or_path"] == {
        "text_encoder": str(tmp_path), "vae": str(tmp_path),
        "scheduler": str(tmp_path)}
    assert pipe.text_encoder == "loaded:text_encoder"


def test_load_components_raises_on_a_component_that_stayed_none(tmp_path):
    """diffusers only WARNS when a component fails to load, so a None text_encoder
    used to surface much later as 'NoneType' object has no attribute 'dtype'
    inside a denoise block. Fail at load, naming the component."""
    pytest.importorskip("torch")
    (tmp_path / "vae").mkdir()  # text_encoder/ is absent → cannot be repointed
    pipe = _FakePipe({
        "text_encoder": _FakeSpec("circlestone-labs/Anima-Base-v1.0-Diffusers",
                                  "text_encoder"),
        "vae": _FakeSpec("circlestone-labs/Anima-Base-v1.0-Diffusers", "vae"),
    })

    with pytest.raises(RuntimeError, match="text_encoder"):
        AnimaBackend()._load_components(pipe, str(tmp_path))


def test_load_components_leaves_already_local_and_unloadable_specs_alone(tmp_path):
    """A spec already holding a local path needs no repoint; a from_config
    component (no pretrained path) is not ours to load — and must not trip the
    strict None check."""
    pytest.importorskip("torch")
    (tmp_path / "vae").mkdir()
    pipe = _FakePipe({
        "vae": _FakeSpec(str(tmp_path / "vae")),
        "guider": _FakeSpec(None, method="from_config"),
    })

    AnimaBackend()._load_components(pipe, str(tmp_path))

    assert pipe.load_kwargs["pretrained_model_name_or_path"] == {}
    assert pipe.vae == "loaded:vae"


def test_registered_as_a_backend():
    from imference_engine import Engine, RuntimeConfig
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    assert isinstance(engine._backends.get("anima"), AnimaBackend)
