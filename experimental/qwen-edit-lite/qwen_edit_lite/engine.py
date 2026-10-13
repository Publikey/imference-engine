"""Build a Qwen-Image-Edit pipeline from a Preset and run two-phase edits.

Two phases per edit, so the 7B Qwen2.5-VL encoder and the 20B transformer are
never resident together:

1. encode — the instruction + condition images go through the encoder
   (bf16 with CPU offload, or NF4 / int8 on the GPU, cached quantized on disk);
   the encoder is then freed (``unload_encoder``) unless the preset keeps it;
2. denoise — the transformer (bf16 / GGUF / fp8 / torchao / nunchaku) runs
   from ``prompt_embeds`` with the chosen offload mode, then the VAE decodes.

Everything that can fail on a given machine (GGUF + group offload, a LoRA on a
GGUF transformer, nunchaku on diffusers 0.40) is caught and recorded in
``Engine.notes`` so a benchmark run reports it instead of dying.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

from qwen_edit_lite import hub
from qwen_edit_lite.metrics import Meter, free_memory
from qwen_edit_lite.presets import Preset

log = logging.getLogger("qwen_edit_lite")

# lightx2v's scheduler for the Qwen-Image Lightning LoRAs (generate_with_diffusers.py).
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def cache_dir() -> Path:
    d = Path(os.environ.get("QEL_CACHE", Path.home() / ".cache" / "qwen-edit-lite"))
    d.mkdir(parents=True, exist_ok=True)
    return d


class Engine:
    def __init__(self, preset: Preset, *, device: str = "cuda", base_repo: Optional[str] = None,
                 meter: Optional[Meter] = None) -> None:
        self.preset = preset.validate()
        self.device = device
        self.base_repo = base_repo or hub.BASE_REPO
        self.meter = meter or Meter()
        self.notes: dict = {}
        self._base_dir: Optional[str] = None
        self.pipe = None          # denoise pipeline (no text encoder)
        self.enc_pipe = None      # encode-only pipeline (no transformer / vae)

    # ------------------------------------------------------------ loading

    def base_dir(self, *, encoder_weights: bool = False) -> str:
        """Local snapshot of the base repo with only what this preset needs."""
        needs_tf = self.preset.transformer_kind in ("bf16", "fp8-layerwise", "int8-torchao", "fp8-torchao")
        if self._base_dir is None or encoder_weights:
            self._base_dir = hub.base_snapshot(self.base_repo, transformer_weights=needs_tf,
                                               encoder_weights=encoder_weights)
        return self._base_dir

    @property
    def dtype(self):
        import torch
        return torch.bfloat16

    def load_transformer(self):
        from diffusers import QwenImageTransformer2DModel

        p, kind, arg = self.preset, self.preset.transformer_kind, self.preset.transformer_arg
        if kind == "gguf":
            from diffusers import GGUFQuantizationConfig
            path = hub.resolve_gguf(arg)
            self.notes["gguf_file"] = path
            return QwenImageTransformer2DModel.from_single_file(
                path, quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
                config=self.base_dir(), subfolder="transformer", torch_dtype=self.dtype)
        if kind == "nunchaku":
            try:
                from nunchaku import NunchakuQwenImageTransformer2DModel
            except ImportError as e:
                raise RuntimeError("pip install nunchaku (NVIDIA only; wheels per torch/CUDA)") from e
            t = NunchakuQwenImageTransformer2DModel.from_pretrained(arg)
            if p.extra.get("nunchaku_offload", True):
                t.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=int(p.extra.get("num_blocks_on_gpu", 1)))
                self.notes["nunchaku_offload"] = True
            return t
        if kind in ("int8-torchao", "fp8-torchao"):
            from diffusers import TorchAoConfig
            try:
                from torchao.quantization import Float8WeightOnlyConfig, Int8WeightOnlyConfig
                qt = Int8WeightOnlyConfig() if kind == "int8-torchao" else Float8WeightOnlyConfig()
            except ImportError:
                qt = "int8wo" if kind == "int8-torchao" else "float8wo"
            return QwenImageTransformer2DModel.from_pretrained(
                self.base_dir(), subfolder="transformer", torch_dtype=self.dtype,
                quantization_config=TorchAoConfig(qt))
        # bf16, and fp8-layerwise (cast AFTER the Lightning LoRA is fused in bf16: see load()).
        return QwenImageTransformer2DModel.from_pretrained(self.base_dir(), subfolder="transformer", torch_dtype=self.dtype)

    def load_encoder(self):
        """Encode-only pipeline: tokenizer + processor + Qwen2.5-VL encoder."""
        import torch
        from diffusers import QwenImageEditPlusPipeline
        from transformers import Qwen2_5_VLForConditionalGeneration

        kind = self.preset.encoder
        cached = cache_dir() / f"text-encoder-{kind}-{self.base_repo.strip('/').replace('/', '--')}"
        have_cache = kind != "bf16" and (cached / "config.json").is_file()
        base = self.base_dir(encoder_weights=not have_cache)
        if kind == "bf16":
            te = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, subfolder="text_encoder", dtype=self.dtype)
        else:
            if have_cache:
                te = Qwen2_5_VLForConditionalGeneration.from_pretrained(cached, device_map={"": self.device})
                self.notes["encoder_cache"] = "hit"
            else:
                from transformers import BitsAndBytesConfig
                q = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16)
                     if kind == "nf4" else BitsAndBytesConfig(load_in_8bit=True))
                te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    base, subfolder="text_encoder", dtype=self.dtype, quantization_config=q,
                    device_map={"": self.device})
                try:
                    te.save_pretrained(cached)
                    self.notes["encoder_cache"] = "written"
                except Exception as e:  # serialization of bnb weights can fail on old versions
                    self.notes["encoder_cache"] = f"not written: {e}"
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            base, transformer=None, vae=None, text_encoder=te, torch_dtype=self.dtype)
        if kind == "bf16":
            pipe.enable_model_cpu_offload(device=self.device)
        return pipe

    def load(self) -> "Engine":
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline

        p = self.preset
        with self.meter.stage("load_transformer"):
            transformer = self.load_transformer()
        with self.meter.stage("load_pipeline"):
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                self.base_dir(), transformer=transformer, text_encoder=None, torch_dtype=self.dtype)
            if p.lightning != "none":
                pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
        if p.lightning != "none":
            with self.meter.stage("load_lora"):
                self._apply_lightning(pipe)
        if p.transformer_kind == "fp8-layerwise":
            import torch
            pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=self.dtype)
        with self.meter.stage("setup_offload"):
            self._apply_offload(pipe)
            if p.vae_tiling and getattr(pipe, "vae", None) is not None:
                pipe.vae.enable_tiling()
        self.pipe = pipe
        if not p.unload_encoder:
            with self.meter.stage("load_encoder"):
                self.enc_pipe = self.load_encoder()
        return self

    def _apply_lightning(self, pipe) -> None:
        path = hub.resolve_lightning(self.preset.lightning)
        self.notes["lightning_file"] = path
        try:
            pipe.load_lora_weights(path, adapter_name="lightning")
        except Exception as e:
            self.notes["lora"] = f"load failed: {type(e).__name__}: {e}"[:400]
            raise
        if self.preset.transformer_kind in ("gguf", "nunchaku"):
            # Quantized weights cannot absorb a fused delta (checked on diffusers
            # 0.40: fuse_lora on a GGUF Q8_0 transformer raises a shape error and
            # leaves the adapter intact). The unfused adapter costs one small
            # low-rank matmul per projection.
            self.notes["lora"] = "unfused (quantized transformer)"
            return
        if self.preset.fuse_lora:
            try:
                pipe.fuse_lora(lora_scale=1.0, components=["transformer"])
                pipe.unload_lora_weights()
                self.notes["lora"] = "fused"
                return
            except Exception as e:  # e.g. GGUF params cannot absorb a fused delta
                self.notes["lora_fuse_error"] = f"{type(e).__name__}: {e}"[:400]
        self.notes["lora"] = "unfused (adapter active)"

    def _apply_offload(self, pipe) -> None:
        import torch

        mode = self.preset.offload
        dev = torch.device(self.device)
        if mode == "none":
            pipe.to(dev)
        elif mode == "model":
            pipe.enable_model_cpu_offload(device=self.device)
        elif mode == "sequential":
            pipe.enable_sequential_cpu_offload(device=self.device)
        else:
            kw = dict(onload_device=dev, offload_device=torch.device("cpu"), use_stream=True,
                      low_cpu_mem_usage=True)
            if mode == "group-leaf":
                kw.update(offload_type="leaf_level")
            else:
                kw.update(offload_type="block_level", num_blocks_per_group=int(self.preset.extra.get("blocks_per_group", 1)))
            if mode == "group-disk":
                d = cache_dir() / "offload" / self.preset.name
                d.mkdir(parents=True, exist_ok=True)
                kw["offload_to_disk_path"] = str(d)
            try:
                pipe.transformer.enable_group_offload(**kw)
            except Exception as e:
                # Record and fall back: group offload on quantized params is the
                # least documented path (study uncertainty #1).
                self.notes["group_offload_error"] = f"{type(e).__name__}: {e}"[:400]
                pipe.enable_model_cpu_offload(device=self.device)
                self.notes["offload_fallback"] = "model"
                return
            # Small components stay on the GPU.
            if getattr(pipe, "vae", None) is not None:
                pipe.vae.to(dev)

    # ------------------------------------------------------------ inference

    def condition_images(self, images: list) -> list:
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import CONDITION_IMAGE_SIZE, calculate_dimensions

        out = []
        for img in images:
            w, h = img.size
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, w / h)
            out.append(self.pipe.image_processor.resize(img, ch, cw))
        return out

    def encode(self, instruction: str, images: list, negative: Optional[str]) -> dict:
        import torch

        enc = self.enc_pipe or self.load_encoder()
        cond = self.condition_images(images)
        with torch.inference_mode():
            pe, pm = enc.encode_prompt(prompt=instruction, image=cond, device=torch.device(self.device))
            out = {"prompt_embeds": pe, "prompt_embeds_mask": pm}
            if self.preset.effective_cfg > 1:
                ne, nm = enc.encode_prompt(prompt=negative or " ", image=cond, device=torch.device(self.device))
                out.update(negative_prompt_embeds=ne, negative_prompt_embeds_mask=nm)
        # Keep embeddings on the GPU in the transformer dtype; drop the encoder.
        out = {k: (v.to(self.device) if v is not None else None) for k, v in out.items()}
        if self.preset.unload_encoder:
            if self.enc_pipe is not None:
                self.enc_pipe = None
            del enc
            free_memory()
        else:
            self.enc_pipe = enc
        return out

    def edit(self, image, instruction: str, *, references: Optional[list] = None, seed: int = 0,
             negative: Optional[str] = None, steps: Optional[int] = None, cfg: Optional[float] = None) -> Any:
        """Run one edit. Returns a PIL image."""
        import torch

        if self.pipe is None:
            self.load()
        images = [image.convert("RGB")] + [r.convert("RGB") for r in (references or [])]
        with self.meter.stage("encode"):
            emb = self.encode(instruction, images, negative)
        cfg = self.preset.effective_cfg if cfg is None else cfg
        with self.meter.stage("denoise"):
            with torch.inference_mode():
                res = self.pipe(image=images if len(images) > 1 else images[0],
                                num_inference_steps=steps or self.preset.effective_steps,
                                true_cfg_scale=cfg, generator=torch.Generator("cpu").manual_seed(seed), **emb)
        return res.images[0]

    def close(self) -> None:
        self.pipe = None
        self.enc_pipe = None
        free_memory()
