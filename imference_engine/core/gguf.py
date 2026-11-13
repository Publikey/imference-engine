"""GGUF loading helpers shared by the image backends.

diffusers 0.40's GGUF loader (``from_single_file`` + ``GGUFQuantizationConfig``)
only knows how to RUN ``nn.Linear`` layers quantized. Community GGUF files
often quantize every 2-D-ish weight, including the patch-embedding ``Conv2d``;
diffusers then leaves that weight as packed uint8 and the first forward fails
with "Input type (CUDABFloat16Type) and weight type (torch.cuda.ByteTensor)".
"""
from __future__ import annotations

from typing import Any


def dequantize_non_linear(model: Any, dtype: Any) -> list:
    """Restore GGUF-quantized weights that sit on layers diffusers cannot run
    quantized (anything but ``nn.Linear``) to plain ``dtype`` tensors.

    Returns the names of the parameters it converted. A ``Conv2d`` weight comes
    back from the GGUF as a flattened 2-D tensor and is reshaped to the layer's
    kernel layout.
    """
    import torch
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor

    converted = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            continue
        for pname, param in list(module.named_parameters(recurse=False)):
            if not isinstance(param, GGUFParameter) or param.dtype.is_floating_point:
                continue
            weight = dequantize_gguf_tensor(param).to(dtype)
            if isinstance(module, torch.nn.Conv2d) and pname == "weight":
                weight = weight.reshape(
                    module.out_channels, module.in_channels // module.groups, *module.kernel_size)
            setattr(module, pname, torch.nn.Parameter(weight, requires_grad=False))
            converted.append(f"{name}.{pname}")
    return converted
