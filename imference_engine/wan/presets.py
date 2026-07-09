"""Variant definitions — the per-model recipe, validated by the P1d/P1e spikes.

A *variant* binds: mode (t2v/i2v), the diffusers base repo (for config + shared
text_encoder/VAE), the GGUF repo for the two MoE experts, whether Lightning is
already baked (→ no LoRA), and any LoRAs to stack at runtime (applied with
``set_adapters``, never fused — that path is validated and dodges bug #12047).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Official base diffusers repos (config + shared UMT5 text_encoder + Wan VAE)
BASE_T2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
BASE_I2V = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Official base GGUF repos (NOT distilled → need the Lightning LoRA)
GGUF_T2V = "QuantStack/Wan2.2-T2V-A14B-GGUF"
GGUF_I2V = "QuantStack/Wan2.2-I2V-A14B-GGUF"

# Official 4-step Lightning distill LoRA
_LIGHTNING_REPO = "lightx2v/Wan2.2-Lightning"
_T2V_LORA_HIGH = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors"
_T2V_LORA_LOW = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors"
_I2V_LORA_HIGH = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors"
_I2V_LORA_LOW = "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors"


@dataclass
class WanLora:
    """One LoRA, routed per MoE expert (high → transformer, low → transformer_2)."""

    repo: str
    high_weight_name: Optional[str] = None  # safetensors path for the high-noise expert
    low_weight_name: Optional[str] = None   # for the low-noise expert (transformer_2)
    high_weight: float = 1.0
    low_weight: float = 1.0


@dataclass
class WanVariant:
    """A loadable Wan model variant."""

    name: str
    mode: str                       # "t2v" | "i2v"
    base_repo: str                  # diffusers repo: config + shared text_encoder/vae
    gguf_repo: str                  # GGUF repo for the two experts
    arch: str = "wan"               # video backend key — routes to a VideoBackend
    lightning_baked: bool = False   # True for merges (SmoothMix/DaSiWa) → no LoRA
    loras: list[WanLora] = field(default_factory=list)
    flow_shift: float = 3.0         # 3.0 ≈ 480p, 5.0 ≈ 720p
    # GGUF filename resolution, in priority order:
    #  1. gguf_*_name  — exact, quant-fixed (for odd repos)
    #  2. gguf_*_template — has a "{quant}" placeholder; DETERMINISTIC + offline-safe
    #     (no list_repo_files API call → works with HF_HUB_OFFLINE=1 from a mirror)
    #  3. neither → online auto-discovery (convenience; needs network)
    gguf_high_name: Optional[str] = None
    gguf_low_name: Optional[str] = None
    gguf_high_template: Optional[str] = None
    gguf_low_template: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in ("t2v", "i2v"):
            raise ValueError(f"mode must be t2v|i2v, got {self.mode!r}")
        if self.lightning_baked and self.loras:
            raise ValueError(
                f"variant {self.name!r}: lightning_baked=True but loras given — "
                "a baked merge must not stack the Lightning LoRA (double distill)."
            )


def _lightning_lora(mode: str) -> WanLora:
    if mode == "t2v":
        return WanLora(_LIGHTNING_REPO, _T2V_LORA_HIGH, _T2V_LORA_LOW)
    return WanLora(_LIGHTNING_REPO, _I2V_LORA_HIGH, _I2V_LORA_LOW)


# Built-in presets (validated). Callers can register their own via WanEngine.
# GGUF templates make filename resolution deterministic (no list_repo_files at
# runtime) → these variants run fully offline from a mirrored cache.
BUILTIN_VARIANTS: dict[str, WanVariant] = {
    "wan22-t2v-lightning": WanVariant(
        name="wan22-t2v-lightning", mode="t2v",
        base_repo=BASE_T2V, gguf_repo=GGUF_T2V,
        loras=[_lightning_lora("t2v")], flow_shift=3.0,
        gguf_high_template="HighNoise/Wan2.2-T2V-A14B-HighNoise-{quant}.gguf",
        gguf_low_template="LowNoise/Wan2.2-T2V-A14B-LowNoise-{quant}.gguf",
    ),
    "wan22-i2v-lightning": WanVariant(
        name="wan22-i2v-lightning", mode="i2v",
        base_repo=BASE_I2V, gguf_repo=GGUF_I2V,
        loras=[_lightning_lora("i2v")], flow_shift=3.0,
        gguf_high_template="HighNoise/Wan2.2-I2V-A14B-HighNoise-{quant}.gguf",
        gguf_low_template="LowNoise/Wan2.2-I2V-A14B-LowNoise-{quant}.gguf",
    ),
    # Example Civitai merges with Lightning baked in (→ lightning_baked, no LoRA).
    "smoothmix-i2v": WanVariant(
        name="smoothmix-i2v", mode="i2v",
        base_repo=BASE_I2V, gguf_repo="Bedovyy/smoothMixWan22-I2V-GGUF",
        lightning_baked=True, flow_shift=3.0,
        gguf_high_template="HighNoise/smoothMixWan22I2VT2V_i2vHigh-{quant}.gguf",
        gguf_low_template="LowNoise/smoothMixWan22I2VT2V_i2vLow-{quant}.gguf",
    ),
    "dasiwa-i2v": WanVariant(
        name="dasiwa-i2v", mode="i2v",
        base_repo=BASE_I2V, gguf_repo="Bedovyy/dasiwaWAN22I2V14B-GGUF",
        lightning_baked=True, flow_shift=3.0,
        gguf_high_template="HighNoise/dasiwaWAN22I2V14B_midnightflirtHigh-{quant}.gguf",
        gguf_low_template="LowNoise/dasiwaWAN22I2V14B_midnightflirtLow-{quant}.gguf",
    ),
}
