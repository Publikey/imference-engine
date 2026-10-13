# qwen-edit-lite

Standalone runner + benchmark for **Qwen-Image-Edit-2511 on consumer hardware**.
It is a test bed, separate from `imference_engine`: it measures the levers from
the quantization study on a real GPU before the winning combination moves into
the engine's `qwenimage-edit` backend.

| Lever | How | Expected effect |
|---|---|---|
| Lightning 4-step LoRA (lightx2v, Apache-2.0) | 4 steps, CFG 1.0 (one pass per step), lightx2v scheduler | ~20x less transformer compute than 40 steps x CFG 4 |
| GGUF transformer (unsloth, Apache-2.0) | `from_single_file` + `GGUFQuantizationConfig` | Q6_K 16.9 GB, Q5_K_M 15 GB, Q4_K_M 13.2 GB (bf16: 41 GB) |
| NF4 text encoder, released after encoding | bitsandbytes 4-bit, cached quantized on disk | 16.6 GB -> ~5-6 GB, and gone before denoising |
| Group offload (block / leaf / disk) | `enable_group_offload(use_stream=True, low_cpu_mem_usage=True)` | a few GB of VRAM; `group-disk` for 16 GB RAM |
| fp8 layerwise / torchao int8 | storage fp8 / int8 weight-only | for comparison and cloud |

Already checked on CPU with a tiny random transformer (diffusers 0.40, `tests/test_tiny_models.py`):
GGUF loads through `from_single_file(config=..., subfolder="transformer")`; the
Lightning LoRA **loads and works on a GGUF transformer but cannot be fused**
(the engine keeps it as an adapter there); group offload (block / leaf / disk)
works on GGUF with the LoRA. Not checked: the real unsloth / lightx2v files, speed,
memory. That is what the benchmark below is for.

## Install (on the GPU machine)

```bash
cd imference-engine/experimental/qwen-edit-lite
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# 1. torch for your GPU first (NVIDIA example):
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2. the package with GGUF + NF4 + LPIPS:
pip install -e ".[all,dev]"
qwen-edit-lite doctor            # GPU, RAM, libs, and which presets to start with
```

Disk: ~0.3 GB of base files + the GGUF you test (13-22 GB each) + 16.6 GB once
for the encoder (quantized to NF4 on first use, then cached in
`~/.cache/qwen-edit-lite`; set `QEL_CACHE` to move it). `bf16-*` presets also
download the 41 GB bf16 transformer — skip them on small disks.

## Run

```bash
qwen-edit-lite presets                     # what each preset combines
qwen-edit-lite files                       # exact GGUF / Lightning files on the Hub

# one edit
qwen-edit-lite edit --preset q6-l4 --image photo.jpg \
    --instruction "Replace the sky with a sunset" -o out.png
qwen-edit-lite edit --preset q5-l4 --image person.jpg --ref hat.jpg \
    --instruction "Put the hat from image 2 on the person in image 1"

# benchmark: each preset in its own process (clean peak RAM/VRAM, an OOM kills one row)
qwen-edit-lite bench --image photo.jpg --presets auto --out bench/
qwen-edit-lite bench --cases examples/cases.yaml --presets q8-l4,q6-l4,q5-l4,q4-l4 --reference q8-l4
```

`bench/results.md` holds the table (load time, seconds per edit, VRAM / RAM
peaks, PSNR + LPIPS vs the reference preset, notes such as the LoRA mode or an
offload fallback); `grid-<case>.png` puts the presets side by side;
`results.json` has everything, including tracebacks.

Overrides on any preset: `--transformer gguf:Q4_K_S`, `--encoder bf16`,
`--lightning none|4|<file>`, `--offload none|model|sequential|group|group-leaf|group-disk`,
`--steps`, `--cfg`, `--keep-encoder`, `--no-fuse`. Custom presets: `examples/presets.yaml`.

## What to send back after a run

`bench/results.md`, `bench/results.json`, the grids, and `qwen-edit-lite doctor`
output. The questions they answer:

1. Do the real unsloth 2511 GGUF files load (tensor names) and run with group offload?
2. Does the real lightx2v 2511 LoRA load on the GGUF transformer (key format)?
3. Peak RAM / VRAM per preset on your machine — does `q5-l4` fit 24 GB RAM? does `q5-l4-disk` fit 16 GB?
4. Seconds per edit per preset; quality drift Q8 -> Q6 -> Q5 -> Q4 (LPIPS + your eyes on the grids).

## Known limits

- NVIDIA (CUDA) is the main target. AMD ROCm: GGUF works with torch-ROCm, bitsandbytes ROCm support varies — use `--encoder bf16` if NF4 fails (the encoder then runs with CPU offload).
- Each edit re-loads the encoder unless `--keep-encoder` (the NF4 cache makes it a few seconds).
- The 2511 Lightning release is 4-step only; `--lightning 8` fails with a clear message.
- Nunchaku (fastest on NVIDIA) has official checkpoints for 2509 only: see `examples/presets.yaml`.
