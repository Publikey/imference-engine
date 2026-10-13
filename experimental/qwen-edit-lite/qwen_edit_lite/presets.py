"""Run presets: one named combination of the memory levers.

A preset is plain data (no torch import) so it can be listed, printed and
tested anywhere. Spec strings:

    transformer: bf16 | fp8-layerwise | int8-torchao | fp8-torchao
                 | gguf:<QUANT>        (e.g. gguf:Q5_K_M — resolved in the GGUF repo)
                 | gguf:<path or URL>  (a .gguf file)
                 | nunchaku:<path>     (an SVDQuant .safetensors, NVIDIA only)
    encoder:     bf16 | nf4 | int8
    lightning:   none | 4 | 8 | <path to a LoRA .safetensors>
    offload:     none | model | sequential | group | group-leaf | group-disk
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Optional

TRANSFORMER_KINDS = ("bf16", "fp8-layerwise", "int8-torchao", "fp8-torchao", "gguf", "nunchaku")
ENCODER_KINDS = ("bf16", "nf4", "int8")
OFFLOAD_KINDS = ("none", "model", "sequential", "group", "group-leaf", "group-disk")


class PresetError(ValueError):
    pass


@dataclass
class Preset:
    name: str
    transformer: str = "bf16"
    encoder: str = "bf16"
    lightning: str = "none"
    offload: str = "group"
    unload_encoder: bool = True        # encode, free the encoder, then denoise
    vae_tiling: bool = True
    steps: Optional[int] = None        # None: 4/8 with Lightning, 40 otherwise
    cfg: Optional[float] = None        # None: 1.0 with Lightning, 4.0 otherwise
    fuse_lora: bool = True
    description: str = ""
    extra: dict = field(default_factory=dict)

    # -- derived ----------------------------------------------------------

    @property
    def transformer_kind(self) -> str:
        return self.transformer.split(":", 1)[0]

    @property
    def transformer_arg(self) -> Optional[str]:
        return self.transformer.split(":", 1)[1] if ":" in self.transformer else None

    @property
    def lightning_steps(self) -> Optional[int]:
        if self.lightning in ("4", "8"):
            return int(self.lightning)
        if self.lightning != "none":
            low = self.lightning.lower()
            if "8step" in low or "8-step" in low:
                return 8
            return 4
        return None

    @property
    def effective_steps(self) -> int:
        if self.steps:
            return self.steps
        return self.lightning_steps or 40

    @property
    def effective_cfg(self) -> float:
        if self.cfg is not None:
            return self.cfg
        return 1.0 if self.lightning != "none" else 4.0

    def validate(self) -> "Preset":
        if self.transformer_kind not in TRANSFORMER_KINDS:
            raise PresetError(f"{self.name}: transformer {self.transformer!r}; kinds: {TRANSFORMER_KINDS}")
        if self.transformer_kind in ("gguf", "nunchaku") and not self.transformer_arg:
            raise PresetError(f"{self.name}: {self.transformer_kind} needs a quant or a path (e.g. gguf:Q5_K_M)")
        if self.encoder not in ENCODER_KINDS:
            raise PresetError(f"{self.name}: encoder {self.encoder!r}; one of {ENCODER_KINDS}")
        if self.offload not in OFFLOAD_KINDS:
            raise PresetError(f"{self.name}: offload {self.offload!r}; one of {OFFLOAD_KINDS}")
        if self.encoder in ("nf4", "int8") and not self.unload_encoder and self.offload in ("model", "sequential"):
            raise PresetError(f"{self.name}: a bitsandbytes encoder cannot be CPU-offloaded; set unload_encoder")
        if self.transformer_kind == "nunchaku" and self.offload.startswith("group"):
            raise PresetError(f"{self.name}: nunchaku has its own per-layer offload; use offload none or model")
        return self

    def with_overrides(self, **kw) -> "Preset":
        kw = {k: v for k, v in kw.items() if v is not None}
        return replace(self, **kw).validate()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["effective_steps"] = self.effective_steps
        d["effective_cfg"] = self.effective_cfg
        return d


# Ordered from the reference to the lightest. VRAM/RAM targets are the study's
# estimates — measuring them is what this package is for.
BUILTIN = {p.name: p.validate() for p in [
    Preset("bf16-full", transformer="bf16", encoder="bf16", lightning="none", offload="group", unload_encoder=False,
           description="Reference quality: bf16, 40 steps, CFG 4. Needs ~64 GB RAM with group offload."),
    Preset("bf16-l4", transformer="bf16", encoder="bf16", lightning="4", offload="group",
           description="bf16 + Lightning 4 steps: the speed lever alone."),
    Preset("q8-l4", transformer="gguf:Q8_0", encoder="nf4", lightning="4",
           description="GGUF Q8_0 (21.8 GB) + Lightning 4 + NF4 encoder released after encode."),
    Preset("q6-l4", transformer="gguf:Q6_K", encoder="nf4", lightning="4",
           description="Recommended 16 GB VRAM / 32 GB RAM: Q6_K (16.9 GB) + Lightning 4 + NF4 encoder."),
    Preset("q5-l4", transformer="gguf:Q5_K_M", encoder="nf4", lightning="4",
           description="Recommended 12 GB VRAM / 24 GB RAM: Q5_K_M (15 GB) + Lightning 4 + NF4 encoder."),
    Preset("q5-l4-disk", transformer="gguf:Q5_K_M", encoder="nf4", lightning="4", offload="group-disk",
           description="16 GB RAM machines: Q5_K_M with group offload to disk."),
    Preset("q4-l4", transformer="gguf:Q4_K_M", encoder="nf4", lightning="4",
           description="Last resort for 8 GB cards: Q4_K_M (13.2 GB). Community: quality drops at Q4."),
    Preset("q6-full", transformer="gguf:Q6_K", encoder="nf4", lightning="none",
           description="Q6_K without Lightning (40 steps, CFG 4): isolates the quantization cost."),
    Preset("fp8-l4", transformer="fp8-layerwise", encoder="nf4", lightning="4",
           description="bf16 weights stored as fp8 (layerwise casting), any CUDA GPU."),
    Preset("int8ao-l4", transformer="int8-torchao", encoder="nf4", lightning="4", offload="model",
           description="torchao int8 weight-only (cloud-style, Ampere+)."),
]}


def get(name: str) -> Preset:
    try:
        return BUILTIN[name]
    except KeyError:
        raise PresetError(f"unknown preset {name!r}; known: {', '.join(BUILTIN)}") from None


def from_dict(d: dict) -> Preset:
    base = BUILTIN.get(d.get("base", ""), None)
    d = {k: v for k, v in d.items() if k != "base"}
    if base is not None:
        return replace(base, **d).validate()
    if "name" not in d:
        raise PresetError("a custom preset needs a name")
    allowed = set(Preset.__dataclass_fields__)
    unknown = set(d) - allowed
    if unknown:
        raise PresetError(f"unknown preset key(s) {sorted(unknown)}")
    return Preset(**d).validate()
