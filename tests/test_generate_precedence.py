"""End-to-end tests of the defaults precedence chain through Engine.generate().

No torch / no GPU: a recording FakeBackend captures the EFFECTIVE params the
engine resolves and passes to build_inference_kwargs / apply_scheduler /
encode_prompts. That is the true observation point for
`request > model > engine > GLOBAL_DEFAULTS`.

Also pins BACKWARD COMPATIBILITY: with no catalog defaults, an explicit request
behaves exactly as before the Optional refactor, and an empty request falls back
to the historical 1024 / 28 / 6.0 values.
"""
from __future__ import annotations

from imference_engine import Engine, RuntimeConfig
from imference_engine.catalog.defaults import GenerationDefaults
from imference_engine.pipelines.base import PipelineBackend


class _FakeVAE:
    def enable_slicing(self): pass
    def disable_slicing(self): pass
    def enable_tiling(self): pass
    def disable_tiling(self): pass


class _FakeOut:
    def __init__(self, images): self.images = images


class _FakePipe:
    def __init__(self): self.vae = _FakeVAE()
    def to(self, device): return self
    def __call__(self, **kwargs):
        n = kwargs.get("num_images_per_prompt", 1)
        return _FakeOut(images=[object() for _ in range(n)])


class RecordingBackend(PipelineBackend):
    """Captures the effective params the engine resolves. Optional
    `engine_defaults_override` lets a test inject the engine layer."""

    engine = "fake"

    def __init__(self, engine_defaults_override: GenerationDefaults | None = None):
        self._override = engine_defaults_override
        self.infer_kwargs: dict | None = None
        self.scheduler_arg = None
        self.scheduler_backend_options: dict | None = None
        self.encode_negative = "__unset__"

    def engine_defaults(self):
        return self._override if self._override is not None else GenerationDefaults()

    def load_pipeline(self, *, local_path, base_model=None):
        return _FakePipe()

    def make_img2img(self, t2i_pipe):
        return t2i_pipe

    def get_compute_module(self, pipe):
        return pipe

    def encode_prompts(self, pipe, prompt, negative_prompt):
        self.encode_negative = negative_prompt
        return {"prompt": prompt, "negative_prompt": negative_prompt or ""}

    def apply_scheduler(self, pipe, scheduler, **kwargs):
        self.scheduler_arg = scheduler
        self.scheduler_backend_options = kwargs

    def build_inference_kwargs(self, *, width, height, num_steps, guidance_scale,
                               clip_skip, chunk_size, generator, image=None,
                               strength=0.75):
        self.infer_kwargs = {
            "width": width, "height": height, "num_steps": num_steps,
            "guidance_scale": guidance_scale, "clip_skip": clip_skip,
            "strength": strength,
        }
        return {"num_images_per_prompt": chunk_size}

    def make_generator(self, seed, device):
        return None


def _engine_with(backend: RecordingBackend, *, defaults=None):
    """Build a CPU engine and swap in the recording backend under key 'fake'."""
    engine = Engine(runtime=RuntimeConfig(device="cpu")).load()
    engine._backends["fake"] = backend
    engine._models._backends["fake"] = backend
    engine.register_model("m", backend="fake", weights_path="m", defaults=defaults)
    return engine


def test_backward_compat_explicit_request_reaches_inference():
    """An explicit request with no catalog defaults passes those exact params
    through — identical to pre-refactor behavior."""
    be = RecordingBackend()
    engine = _engine_with(be)
    engine.generate(model="m", prompt="cat", num_steps=15, guidance_scale=6.0,
                    width=768, height=768, clip_skip=2, seed=1)
    assert be.infer_kwargs == {
        "width": 768, "height": 768, "num_steps": 15, "guidance_scale": 6.0,
        "clip_skip": 2, "strength": 0.75,
    }


def test_backward_compat_empty_request_uses_historical_globals():
    """No catalog, no request params → the historical 1024/28/6.0 defaults."""
    be = RecordingBackend()
    engine = _engine_with(be)
    engine.generate(model="m", prompt="cat", seed=1)
    assert be.infer_kwargs["width"] == 1024
    assert be.infer_kwargs["height"] == 1024
    assert be.infer_kwargs["num_steps"] == 28
    assert be.infer_kwargs["guidance_scale"] == 6.0
    assert be.infer_kwargs["strength"] == 0.75
    assert be.infer_kwargs["clip_skip"] is None


def test_model_defaults_fill_what_request_omits():
    be = RecordingBackend()
    engine = _engine_with(be, defaults=GenerationDefaults(
        num_steps=8, backend_options={"shift": 3.0}))
    engine.generate(model="m", prompt="cat", seed=1)  # no num_steps in request
    assert be.infer_kwargs["num_steps"] == 8              # model default reached
    assert be.scheduler_backend_options == {"shift": 3.0}  # model shift reached


def test_request_overrides_model_default():
    be = RecordingBackend()
    engine = _engine_with(be, defaults=GenerationDefaults(num_steps=8))
    engine.generate(model="m", prompt="cat", num_steps=50, seed=1)
    assert be.infer_kwargs["num_steps"] == 50  # request wins over model's 8


def test_engine_layer_participates_below_model_and_request():
    """engine < model < request end-to-end: engine sets scheduler, model sets
    steps, request overrides steps."""
    be = RecordingBackend(engine_defaults_override=GenerationDefaults(
        scheduler="EulerAncestralDiscreteScheduler", guidance_scale=6.0))
    engine = _engine_with(be, defaults=GenerationDefaults(num_steps=8))
    engine.generate(model="m", prompt="cat", num_steps=50, seed=1)
    assert be.scheduler_arg == "EulerAncestralDiscreteScheduler"  # engine layer
    assert be.infer_kwargs["num_steps"] == 50                     # request over model
    assert be.infer_kwargs["guidance_scale"] == 6.0              # engine layer


def test_backend_options_merge_across_model_and_request():
    be = RecordingBackend()
    engine = _engine_with(be, defaults=GenerationDefaults(
        backend_options={"shift": 3.0, "keep": 1}))
    engine.generate(model="m", prompt="cat",
                    backend_options={"shift": 5.0, "extra": 2}, seed=1)
    assert be.scheduler_backend_options == {"shift": 5.0, "keep": 1, "extra": 2}


def test_model_default_negative_prompt_reaches_encoder():
    be = RecordingBackend()
    engine = _engine_with(be, defaults=GenerationDefaults(
        negative_prompt="lowres, blurry"))
    engine.generate(model="m", prompt="cat", seed=1)  # no negative in request
    assert be.encode_negative == "lowres, blurry"
