"""Tests for the engine env contract: RuntimeConfig/WanRuntimeConfig.from_env
and the runtime.env parsing helpers. Runs without torch.

These lock down the decoupling seam: the engine reads a documented, static env
contract (param > env > default); hardware "auto" resolution stays worker-side,
so MAX_*_MODELS=auto must leave the field None for the worker to fill.
"""
from imference_engine.engine import RuntimeConfig
from imference_engine.wan.config import MemoryProfile, WanRuntimeConfig
from imference_engine.runtime.env import env_bool, env_int_or_none, env_str


# --------------------------------------------------------------------------
# env helpers
# --------------------------------------------------------------------------

def test_env_str_unset_and_empty_fall_back(monkeypatch):
    monkeypatch.delenv("X_FOO", raising=False)
    assert env_str("X_FOO", "def") == "def"
    monkeypatch.setenv("X_FOO", "")
    assert env_str("X_FOO", "def") == "def"
    monkeypatch.setenv("X_FOO", "val")
    assert env_str("X_FOO", "def") == "val"
    assert env_str("X_MISSING") is None


def test_env_bool_truthy_set(monkeypatch):
    for v in ("1", "true", "TRUE", "Yes", "on"):
        monkeypatch.setenv("X_B", v)
        assert env_bool("X_B", False) is True
    for v in ("0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("X_B", v)
        assert env_bool("X_B", True) is False


def test_env_bool_default(monkeypatch):
    monkeypatch.delenv("X_B", raising=False)
    assert env_bool("X_B", True) is True
    assert env_bool("X_B", False) is False


def test_env_int_or_none(monkeypatch):
    monkeypatch.delenv("X_N", raising=False)
    assert env_int_or_none("X_N") is None
    for v in ("auto", "AUTO", "", "notanint"):
        monkeypatch.setenv("X_N", v)
        assert env_int_or_none("X_N") is None
    monkeypatch.setenv("X_N", "4")
    assert env_int_or_none("X_N") == 4


# --------------------------------------------------------------------------
# RuntimeConfig.from_env (image: SDXL + Z-Image)
# --------------------------------------------------------------------------

_IMAGE_VARS = [
    "IMAGE_DEVICE", "IMAGE_MODEL_CACHE", "IMAGE_MODEL_CDN",
    "MAX_GPU_MODELS", "MAX_CPU_MODELS",
    "IMAGE_USE_TINY_VAE", "IMAGE_ENABLE_CPU_OFFLOAD",
]


def _clear(monkeypatch, names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_image_from_env_defaults(monkeypatch):
    _clear(monkeypatch, _IMAGE_VARS)
    cfg = RuntimeConfig.from_env()
    assert cfg == RuntimeConfig()  # bit-for-bit the dataclass defaults
    assert cfg.device == "auto"
    assert cfg.model_cache_dir is None and cfg.model_cdn is None
    assert cfg.max_gpu_models is None and cfg.max_cpu_models is None
    assert cfg.use_tiny_vae is False and cfg.enable_cpu_offload is False


def test_image_from_env_populated(monkeypatch):
    _clear(monkeypatch, _IMAGE_VARS)
    monkeypatch.setenv("IMAGE_DEVICE", "cuda:1")
    monkeypatch.setenv("IMAGE_MODEL_CACHE", "/workspace/image-tree")
    monkeypatch.setenv("IMAGE_MODEL_CDN", "https://cdn.example/")
    monkeypatch.setenv("MAX_GPU_MODELS", "2")
    monkeypatch.setenv("MAX_CPU_MODELS", "6")
    monkeypatch.setenv("IMAGE_USE_TINY_VAE", "true")
    monkeypatch.setenv("IMAGE_ENABLE_CPU_OFFLOAD", "1")
    cfg = RuntimeConfig.from_env()
    assert cfg.device == "cuda:1"
    assert cfg.model_cache_dir == "/workspace/image-tree"
    assert cfg.model_cdn == "https://cdn.example/"
    assert cfg.max_gpu_models == 2 and cfg.max_cpu_models == 6
    assert cfg.use_tiny_vae is True and cfg.enable_cpu_offload is True


def test_image_from_env_auto_models_stay_none(monkeypatch):
    """The worker resolves 'auto' -> number; the engine must keep None."""
    _clear(monkeypatch, _IMAGE_VARS)
    monkeypatch.setenv("MAX_GPU_MODELS", "auto")
    monkeypatch.setenv("MAX_CPU_MODELS", "auto")
    cfg = RuntimeConfig.from_env()
    assert cfg.max_gpu_models is None
    assert cfg.max_cpu_models is None


# --------------------------------------------------------------------------
# WanRuntimeConfig.from_env
# --------------------------------------------------------------------------

_WAN_VARS = [
    "WAN_DEVICE", "WAN_PROFILE", "WAN_MAX_RESIDENT", "WAN_MODEL_CACHE",
    "WAN_MODEL_CDN", "WAN_TEXT_ENCODER_QUANT", "WAN_VAE_TILING",
    "WAN_ENABLE_OFFLOAD",
]


def test_wan_from_env_defaults(monkeypatch):
    _clear(monkeypatch, _WAN_VARS)
    cfg = WanRuntimeConfig.from_env()
    assert cfg.device == "auto"
    # from_env defaults memory_profile to "auto" (resolve at load), NOT GGUF_Q8
    assert cfg.memory_profile == "auto"
    assert cfg.max_resident_variants == 1
    assert cfg.model_cache_dir is None and cfg.model_cdn is None
    assert cfg.text_encoder_quant == "int8"
    assert cfg.vae_tiling is True and cfg.enable_offload is True


def test_wan_from_env_populated(monkeypatch):
    _clear(monkeypatch, _WAN_VARS)
    monkeypatch.setenv("WAN_DEVICE", "cuda")
    monkeypatch.setenv("WAN_PROFILE", "gguf_q6")
    monkeypatch.setenv("WAN_MAX_RESIDENT", "3")
    monkeypatch.setenv("WAN_MODEL_CACHE", "/workspace/wan-tree")
    monkeypatch.setenv("WAN_MODEL_CDN", "https://cdn.example/")
    monkeypatch.setenv("WAN_TEXT_ENCODER_QUANT", "none")
    monkeypatch.setenv("WAN_VAE_TILING", "false")
    monkeypatch.setenv("WAN_ENABLE_OFFLOAD", "no")
    cfg = WanRuntimeConfig.from_env()
    assert cfg.device == "cuda"
    assert cfg.memory_profile == "gguf_q6"  # consumed by MemoryProfile at load
    assert cfg.max_resident_variants == 3
    assert cfg.model_cache_dir == "/workspace/wan-tree"
    assert cfg.model_cdn == "https://cdn.example/"
    assert cfg.text_encoder_quant == "none"
    assert cfg.vae_tiling is False and cfg.enable_offload is False


def test_wan_from_env_max_resident_auto(monkeypatch):
    _clear(monkeypatch, _WAN_VARS)
    monkeypatch.setenv("WAN_MAX_RESIDENT", "auto")
    assert WanRuntimeConfig.from_env().max_resident_variants == 1


def test_wan_profile_string_resolves_to_enum():
    """A 'gguf_q6' string from env is accepted by MemoryProfile downstream."""
    assert MemoryProfile("gguf_q6") is MemoryProfile.GGUF_Q6
