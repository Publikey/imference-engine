# qwen-edit-lite benchmark

| preset | transformer | encoder | steps | cfg | offload | load s | s/edit | VRAM peak GB | RAM peak GB | PSNR vs q5-l4-disk | LPIPS | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q4-l4-disk | gguf:Q4_K_M | nf4 | 4 | 1.0 | group-disk | 34 | 79.9 | 6.47 | 21.18 | 32.32 | 0.038 | lora=unfused (quantized transformer) |
| q5-l4-disk | gguf:Q5_K_M | nf4 | 4 | 1.0 | group-disk | 48 | 79.3 | 6.47 | 20.56 | - | - | lora=unfused (quantized transformer) |
| q4-l4 | gguf:Q4_K_M | nf4 | 4 | 1.0 | group | 23 | 53.8 | 6.47 | 22.44 | 32.32 | 0.038 | lora=unfused (quantized transformer) |
| q5-l4 | gguf:Q5_K_M | nf4 | 4 | 1.0 | group | 23 | 52.3 | 6.47 | 23.81 | inf | 0.000 | lora=unfused (quantized transformer) |
