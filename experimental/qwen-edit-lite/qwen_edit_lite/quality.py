"""Image similarity against a reference run: PSNR (always) and LPIPS (if installed).

These measure drift from the bf16 reference, not absolute quality: a Lightning
run is expected to differ (different sampler), so compare quantized runs
against a reference with the SAME step recipe (e.g. q6-l4 vs bf16-l4).
"""
from __future__ import annotations

from typing import Optional


def psnr(a, b) -> float:
    import numpy as np

    if a.size != b.size:
        b = b.resize(a.size)
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


_LPIPS = None


def lpips(a, b) -> Optional[float]:
    global _LPIPS
    try:
        import lpips as _lp
        import numpy as np
        import torch
    except ImportError:
        return None
    if _LPIPS is None:
        _LPIPS = _lp.LPIPS(net="alex", verbose=False)
    if a.size != b.size:
        b = b.resize(a.size)

    def t(img):
        x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0)
        return x.permute(2, 0, 1)[None]

    with torch.no_grad():
        return float(_LPIPS(t(a), t(b)).item())
