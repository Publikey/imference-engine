"""CPU-only tests: presets, file resolution, two-phase logic with fake models,
metrics, quality, bench plumbing. The real models are exercised by `bench` on a GPU."""
from __future__ import annotations

import json

import pytest
from PIL import Image

from qwen_edit_lite import hub
from qwen_edit_lite.cli import main, recommend
from qwen_edit_lite.presets import BUILTIN, Preset, PresetError, from_dict, get

UNSLOTH = [
    "README.md", "qwen-image-edit-2511-Q2_K.gguf", "qwen-image-edit-2511-Q4_K_M.gguf",
    "qwen-image-edit-2511-Q4_K_S.gguf", "qwen-image-edit-2511-Q5_K_M.gguf", "qwen-image-edit-2511-Q6_K.gguf",
    "qwen-image-edit-2511-Q8_0.gguf", "qwen-image-edit-2511-BF16.gguf",
]
LIGHTX2V = [
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors",
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
    "qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors",
]


# ------------------------------------------------------------------ presets

def test_builtin_presets_valid_and_defaults():
    assert {"bf16-full", "q6-l4", "q5-l4", "q4-l4", "q5-l4-disk"} <= set(BUILTIN)
    q6 = get("q6-l4")
    assert q6.transformer_kind == "gguf" and q6.transformer_arg == "Q6_K"
    assert q6.effective_steps == 4 and q6.effective_cfg == 1.0 and q6.unload_encoder
    full = get("bf16-full")
    assert full.effective_steps == 40 and full.effective_cfg == 4.0


@pytest.mark.parametrize("kw", [
    {"transformer": "int4"}, {"transformer": "gguf"}, {"encoder": "fp4"}, {"offload": "cloud"},
    {"transformer": "nunchaku:/x.safetensors", "offload": "group"},
    {"encoder": "nf4", "unload_encoder": False, "offload": "model"},
])
def test_invalid_presets(kw):
    with pytest.raises(PresetError):
        Preset(name="x", **kw).validate()


def test_overrides_and_from_dict():
    p = get("q6-l4").with_overrides(steps=8, transformer="gguf:Q5_K_M", cfg=None)
    assert p.effective_steps == 8 and p.transformer_arg == "Q5_K_M"
    c = from_dict({"base": "q5-l4", "name": "mine", "offload": "group-leaf"})
    assert c.name == "mine" and c.offload == "group-leaf" and c.encoder == "nf4"
    with pytest.raises(PresetError):
        from_dict({"name": "x", "nope": 1})
    assert Preset(name="l", lightning="/a/Lightning-8steps.safetensors").lightning_steps == 8


def test_recommend():
    assert recommend(24, 64)[0] == "bf16-l4"
    assert recommend(16, 32)[0] == "q6-l4"
    assert recommend(12, 24)[0] == "q5-l4"
    assert recommend(8, 16)[0] == "q5-l4-disk"
    assert recommend(0, 64) == []


# --------------------------------------------------------------------- hub

def test_pick_gguf_exact_tag():
    assert hub.pick_gguf(UNSLOTH, "Q5_K_M").endswith("Q5_K_M.gguf")
    assert hub.pick_gguf(UNSLOTH, "q6_k").endswith("Q6_K.gguf")
    assert hub.pick_gguf(UNSLOTH, "Q4_K_S").endswith("Q4_K_S.gguf")
    with pytest.raises(hub.ResolveError) as e:
        hub.pick_gguf(UNSLOTH, "Q3_K_M")
    assert "Q5_K_M" in str(e.value)


def test_pick_lightning_prefers_bf16_lora():
    assert hub.pick_lightning(LIGHTX2V, 4) == "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    with pytest.raises(hub.ResolveError):
        hub.pick_lightning(LIGHTX2V, 8)


def test_resolve_with_fake_hub(tmp_path):
    fetched = []

    def fetch(repo, f):
        fetched.append((repo, f))
        return f"/cache/{f}"

    assert hub.resolve_gguf("Q6_K", lister=lambda r: UNSLOTH, fetch=fetch).endswith("Q6_K.gguf")
    assert fetched[-1][0] == hub.GGUF_REPO
    local = tmp_path / "m.gguf"
    local.write_bytes(b"x")
    assert hub.resolve_gguf(str(local)) == str(local)
    assert hub.resolve_lightning("none") is None
    assert hub.resolve_lightning("4", lister=lambda r: LIGHTX2V, fetch=fetch).endswith("bf16.safetensors")
    with pytest.raises(hub.ResolveError):
        hub.resolve_gguf("whatever")


# ---------------------------------------------------- two-phase with fakes

class FakeEncPipe:
    def __init__(self):
        self.calls = []

    def encode_prompt(self, prompt, image, device):
        import torch
        self.calls.append((prompt, len(image)))
        return torch.zeros(1, 7, 8), None


class FakePipe:
    class _IP:
        @staticmethod
        def resize(img, h, w):
            return img.resize((w, h))

    image_processor = _IP()

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kw):
        self.kwargs = kw

        class R:
            images = [Image.new("RGB", (32, 32), "red")]
        return R()


def test_two_phase_edit_uses_embeddings_and_frees_encoder(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from qwen_edit_lite import engine as eng_mod

    enc = FakeEncPipe()
    e = eng_mod.Engine(get("q6-l4"), device="cpu")
    e.pipe = FakePipe()
    monkeypatch.setattr(e, "load_encoder", lambda: enc)
    out = e.edit(Image.new("RGB", (640, 480)), "make it red", references=[Image.new("RGB", (100, 100))], seed=5)
    assert out.size == (32, 32)
    kw = e.pipe.kwargs
    assert "prompt" not in kw and kw["prompt_embeds"].shape == (1, 7, 8)
    assert kw["num_inference_steps"] == 4 and kw["true_cfg_scale"] == 1.0
    assert isinstance(kw["image"], list) and len(kw["image"]) == 2
    assert "negative_prompt_embeds" not in kw                      # cfg 1: no negative pass
    assert enc.calls == [("make it red", 2)] and e.enc_pipe is None
    names = [s.name for s in e.meter.stages]
    assert names == ["encode", "denoise"]


def test_cfg_above_one_encodes_negative(monkeypatch):
    pytest.importorskip("diffusers")
    from qwen_edit_lite import engine as eng_mod

    enc = FakeEncPipe()
    e = eng_mod.Engine(get("q6-full"), device="cpu")
    e.pipe = FakePipe()
    monkeypatch.setattr(e, "load_encoder", lambda: enc)
    e.edit(Image.new("RGB", (64, 64)), "x", negative="blurry")
    assert [c[0] for c in enc.calls] == ["x", "blurry"]
    assert e.pipe.kwargs["true_cfg_scale"] == 4.0 and "negative_prompt_embeds" in e.pipe.kwargs
    assert e.pipe.kwargs["num_inference_steps"] == 40


def test_lightning_scheduler_config_builds():
    d = pytest.importorskip("diffusers")
    from qwen_edit_lite.engine import LIGHTNING_SCHEDULER
    s = d.FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    assert s.config.use_dynamic_shifting and s.config.time_shift_type == "exponential"


# ------------------------------------------------------- metrics / quality

def test_meter_records_stages():
    from qwen_edit_lite.metrics import Meter
    m = Meter(interval=0.01).start()
    with m.stage("a"):
        _ = bytearray(10_000_000)
    m.stop()
    d = m.to_dict()
    assert d["stages"][0]["name"] == "a" and d["ram_peak_gb"] > 0


def test_psnr():
    from qwen_edit_lite.quality import psnr
    a = Image.new("RGB", (8, 8), (10, 10, 10))
    assert psnr(a, a) == float("inf")
    assert 20 < psnr(a, Image.new("RGB", (8, 8), (20, 20, 20))) < 40


# ------------------------------------------------------------- bench / cli

def test_summarize_and_grids(tmp_path):
    from qwen_edit_lite.bench import summarize
    rows = []
    for name, color in (("bf16-l4", (0, 0, 0)), ("q6-l4", (4, 4, 4))):
        p = tmp_path / name / "sky.png"
        p.parent.mkdir()
        Image.new("RGB", (64, 64), color).save(p)
        rows.append({"preset": get(name).to_dict(), "cases": [{"id": "sky", "seconds": 3.0, "path": str(p)}],
                     "memory": {"vram_peak_gb": 9.1, "ram_peak_gb": 20.5, "stages": [{"name": "load_transformer", "seconds": 30}]},
                     "notes": {"lora": "fused"}})
    rows.append({"preset": get("q4-l4").to_dict(), "cases": [], "error": "process exited with code -9 (killed: likely out of host RAM)"})
    table = summarize(rows, tmp_path, "bf16-l4")
    assert "q6-l4" in table and "lora=fused" in table and "code -9" in table
    assert (tmp_path / "results.md").is_file() and (tmp_path / "grid-sky.png").is_file()


def test_bench_child_process_reports_failure(tmp_path):
    """The per-preset subprocess path: a preset whose load fails yields an error row, not a crash."""
    from qwen_edit_lite.bench import run_preset_subprocess
    img = tmp_path / "in.png"
    Image.new("RGB", (64, 64)).save(img)
    p = {k: v for k, v in get("q6-l4").to_dict().items() if k in Preset.__dataclass_fields__}
    p["transformer"] = f"gguf:{tmp_path / 'missing.gguf'}"     # not a file -> resolve error
    row = run_preset_subprocess(p, [{"id": "a", "image": str(img), "instruction": "x", "references": []}],
                                tmp_path, 1, timeout=300)
    assert row["error"] and row["cases"] == []


def test_cli_presets_and_dry_run(capsys):
    assert main(["presets"]) == 0
    assert "q6-l4" in capsys.readouterr().out
    assert main(["edit", "--preset", "q5-l4", "--steps", "8", "--image", "x.png", "--instruction", "y", "--dry-run"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["effective_steps"] == 8 and d["transformer"] == "gguf:Q5_K_M"


def test_cli_doctor_runs(capsys):
    assert main(["doctor"]) == 0
    assert "python" in capsys.readouterr().out


def test_base_snapshot_only_fetches_needed_weights():
    seen = {}

    def fetch(repo, allow_patterns):
        seen[repo] = allow_patterns
        return "/snap"

    assert hub.base_snapshot("Qwen/Qwen-Image-Edit-2511", transformer_weights=False, encoder_weights=False,
                             fetch=fetch) == "/snap"
    pats = seen["Qwen/Qwen-Image-Edit-2511"]
    assert "transformer/*" not in pats and "text_encoder/*" not in pats and "vae/*" in pats
    hub.base_snapshot("r/x", transformer_weights=True, encoder_weights=True, fetch=fetch)
    assert "transformer/*" in seen["r/x"] and "text_encoder/*" in seen["r/x"]
