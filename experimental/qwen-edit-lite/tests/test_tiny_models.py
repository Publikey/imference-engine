"""Real diffusers code paths on a tiny random Qwen-Image transformer (CPU).

Covers the study's open questions that do not need a GPU:
- GGUF loading via from_single_file(config=<base dir>, subfolder="transformer") on diffusers 0.40;
- a LoRA on a GGUF transformer: loads and applies unfused (fusing is skipped);
- group offload (block / leaf level) on a GGUF transformer, with the LoRA.
Real checkpoints may still use other tensor names than this synthetic GGUF.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("gguf")
pytest.importorskip("peft")

BF16 = torch.bfloat16


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    subprocess.run([sys.executable, str(Path(__file__).with_name("_tiny.py"))], cwd=d, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return d


def fwd(t):
    torch.manual_seed(1)
    hs = torch.randn(1, 32, 64, dtype=BF16)
    enc = torch.randn(1, 5, 64, dtype=BF16)
    with torch.no_grad():
        return t(hidden_states=hs, encoder_hidden_states=enc, encoder_hidden_states_mask=torch.ones(1, 5, dtype=torch.long),
                 timestep=torch.tensor([0.5], dtype=BF16), img_shapes=[[(1, 4, 4), (1, 4, 4)]],
                 return_dict=False)[0].float()


def holder(t):
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline
    return QwenImageEditPlusPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=None, text_encoder=None,
                                     tokenizer=None, processor=None, transformer=t)


def engine_for(tiny, **kw):
    from qwen_edit_lite.engine import Engine
    from qwen_edit_lite.presets import Preset
    p = Preset(name="t", transformer=f"gguf:{tiny / 'tiny-Q8_0.gguf'}", lightning=str(tiny / "tiny-lora.safetensors"),
               **kw)
    return Engine(p, device="cpu", base_repo=str(tiny / "base"))


def test_gguf_load_close_to_bf16(tiny):
    from diffusers import QwenImageTransformer2DModel
    ref = fwd(QwenImageTransformer2DModel.from_pretrained(tiny / "base", subfolder="transformer", torch_dtype=BF16))
    e = engine_for(tiny)
    t = e.load_transformer()
    out = fwd(t)
    assert float((out - ref).norm() / ref.norm()) < 0.03
    assert any(type(p).__name__ == "GGUFParameter" for p in t.parameters())


def test_lora_on_gguf_stays_unfused_and_applies(tiny):
    e = engine_for(tiny)
    base = fwd(e.load_transformer())
    pipe = holder(e.load_transformer())
    e._apply_lightning(pipe)
    assert e.notes["lora"].startswith("unfused")
    with_lora = fwd(pipe.transformer)
    assert float((with_lora - base).norm() / base.norm()) > 1e-4


@pytest.mark.parametrize("offload", ["group", "group-leaf", "group-disk"])
def test_group_offload_on_gguf_with_lora(tiny, offload, monkeypatch, tmp_path):
    monkeypatch.setenv("QEL_CACHE", str(tmp_path))
    e = engine_for(tiny, offload=offload)
    pipe = holder(e.load_transformer())
    e._apply_lightning(pipe)
    expected = fwd(pipe.transformer)
    pipe2 = holder(e.load_transformer())
    e._apply_lightning(pipe2)
    import qwen_edit_lite.engine as em
    # use_stream needs CUDA: on CPU exercise the hooks without streams.
    orig = pipe2.transformer.enable_group_offload

    def no_stream(**kw):
        kw["use_stream"] = False
        return orig(**kw)

    pipe2.transformer.enable_group_offload = no_stream
    e._apply_offload(pipe2)
    assert "group_offload_error" not in e.notes, e.notes
    got = fwd(pipe2.transformer)
    assert float((got - expected).norm() / expected.norm()) < 1e-3
    del em


def test_fp8_layerwise_casting(tiny):
    from diffusers import QwenImageTransformer2DModel
    t = QwenImageTransformer2DModel.from_pretrained(tiny / "base", subfolder="transformer", torch_dtype=BF16)
    ref = fwd(t)
    t.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=BF16)
    assert float((fwd(t) - ref).norm() / ref.norm()) < 0.1
