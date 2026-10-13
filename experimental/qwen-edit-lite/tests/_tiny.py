import os, torch, numpy as np, gguf
from diffusers import QwenImageTransformer2DModel
torch.manual_seed(0)
cfg = dict(patch_size=2, in_channels=64, out_channels=16, num_layers=2, attention_head_dim=32, num_attention_heads=2,
           joint_attention_dim=64, axes_dims_rope=(8, 12, 12), zero_cond_t=True)
m = QwenImageTransformer2DModel(**cfg).to(torch.bfloat16)
os.makedirs("base/transformer", exist_ok=True)
m.save_pretrained("base/transformer")
sd = {k: v.float().numpy() for k, v in m.state_dict().items()}
w = gguf.GGUFWriter("tiny-Q8_0.gguf", arch="qwen_image")
nq = 0
for k, v in sd.items():
    if v.ndim == 2 and v.shape[-1] % 32 == 0:
        q = gguf.quants.quantize(v, gguf.GGMLQuantizationType.Q8_0)
        w.add_tensor(k, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0); nq += 1
    else:
        w.add_tensor(k, v.astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("quantized tensors:", nq, "of", len(sd))
# tiny LoRA on attn.to_q / to_k of each block (diffusers/peft key format)
from safetensors.torch import save_file
lora = {}
for i in range(2):
    for proj in ("to_q", "to_k", "to_v", "add_q_proj"):
        mod = m.transformer_blocks[i].attn.__getattr__(proj)
        r = 4
        lora[f"transformer.transformer_blocks.{i}.attn.{proj}.lora_A.weight"] = torch.randn(r, mod.in_features) * 0.01
        lora[f"transformer.transformer_blocks.{i}.attn.{proj}.lora_B.weight"] = torch.randn(mod.out_features, r) * 0.01
save_file(lora, "tiny-lora.safetensors")
print("ok")
