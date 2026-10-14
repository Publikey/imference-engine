# RTX 4070 Laptop (8 GB VRAM) · 32 GB RAM · Windows 11 — 2026-10-14

torch 2.11 cu128 · diffusers 0.40.0 · transformers 5.4.0 · bitsandbytes 0.50.2 · gguf 0.19.
Files: unsloth `qwen-image-edit-2511-Q4_K_M/Q5_K_M.gguf`, lightx2v
`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`. Inputs: OpenCV
sample images (`cases.yaml`), seed 42, ~1 MP outputs. ~19 GB of RAM was taken
by desktop apps during the run.

| preset | s/edit | denoise (4 steps) | VRAM peak | RAM peak (stage) | RAM while editing |
|---|---|---|---|---|---|
| q4-l4-disk | 79.9 | 71 s | 6.5 GB | 21.2 GB (load_transformer) | 2.8 GB |
| q5-l4-disk | 79.3 | 74 s | 6.5 GB | 20.6 GB (load_transformer) | 2.8 GB |
| q4-l4 (group, RAM) | 53.8 | 46 s | 6.5 GB | 22.4 GB (load_transformer) | 16.6 GB |
| q5-l4 (group, RAM) | 52.3 | 49 s | 6.5 GB | 23.8 GB (load_transformer) | 17.9 GB |

Answers to the README questions:

1. The real unsloth 2511 GGUF files load (one ignored key, `__index_timestep_zero__`,
   a ComfyUI marker; the 2511 config from the base repo already drives it) and run
   with group offload, to RAM and to disk.
2. The real lightx2v 2511 4-step LoRA loads on the GGUF transformer, unfused (adapter).
3. VRAM peak 6.5 GB everywhere (the NF4 encoder during encoding; denoise ~2 GB).
   **The RAM peak is the GGUF load (~21-24 GB), not the edit**: after loading,
   disk offload edits with 2.8 GB of RAM. Making the load stream would let
   `*-disk` fit 16 GB machines; today it does not.
4. ~52 s per edit with RAM offload, ~80 s to disk (PCIe/disk bound). Q4 vs Q5:
   PSNR 32.3 dB, LPIPS 0.038 — visually identical on all four cases (grids).
   All presets put the ladybug on the lemon instead of the lime (instruction
   following, not quantization). First run only: +20 s to quantize the encoder
   to NF4 (cached afterwards).
