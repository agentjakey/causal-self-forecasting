"""Device and dtype resolution, in one place.

Every other module asks this module what to run on. That is the whole reason it exists: the
development machine has no CUDA device, so the code must move to a rented GPU by changing a
config value and nothing else.

Intel XPU is deliberately not auto-selected. The machine this was written on has an Intel
integrated GPU, and torch can sometimes see it, but nothing in this pipeline has been
measured on XPU. Silently preferring an untested backend would mean every later numerical
discrepancy has an extra suspect. XPU can be requested explicitly if someone wants to
measure it.
"""

from __future__ import annotations

from typing import Any

import torch

from ..logging_utils import info, warn

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


def available_devices() -> dict[str, Any]:
    """Report what torch can actually see on this machine."""
    mps = getattr(torch.backends, "mps", None)
    xpu = getattr(torch, "xpu", None)
    return {
        "cuda": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps": bool(mps and mps.is_available()),
        "xpu": bool(xpu and xpu.is_available()),
        "cpu": True,
    }


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a requested device string to a real device.

    `auto` prefers cuda, then mps, then cpu. An explicit request that is unavailable falls
    back to cpu with a warning rather than raising: a config written for a GPU box should
    still run here at smoke scale, and the run manifest records the device that was actually
    used so no one can mistake the result for a GPU run.
    """
    available = available_devices()

    if requested == "auto":
        if available["cuda"]:
            device = torch.device("cuda")
        elif available["mps"]:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        info("resolved device", requested=requested, device=str(device), **available)
        return device

    device = torch.device(requested)
    if device.type != "cpu" and not available.get(device.type, False):
        warn(
            "requested device is not available, falling back to cpu",
            requested=requested,
            device="cpu",
        )
        return torch.device("cpu")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Resolve a dtype name, refusing combinations that silently misbehave.

    float16 on CPU is allowed by torch but is slow and numerically poor, and this project
    compares small logit differences. Promoting it to float32 keeps a copied GPU config
    usable on CPU without quietly degrading the measurements it produces.
    """
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}; supported: {sorted(_DTYPES)}")
    dtype = _DTYPES[name]
    if device.type == "cpu" and dtype is torch.float16:
        warn("float16 is not usable on cpu for this workload, using float32", requested=name)
        return torch.float32
    return dtype


def dtype_name(dtype: torch.dtype) -> str:
    for name, candidate in _DTYPES.items():
        if candidate is dtype:
            return name
    return str(dtype)
