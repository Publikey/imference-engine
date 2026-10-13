"""qwen-edit-lite: run Qwen-Image-Edit-2511 on consumer hardware, and measure it.

A standalone test bed (not imported by imference-engine) for the levers found
in the quantization study: GGUF transformer, Lightning step distillation,
NF4 text encoder released after encoding, offload modes. Once a combination is
validated on real GPUs it moves into imference-engine's qwenimage-edit backend.
"""
__version__ = "0.1.0"
