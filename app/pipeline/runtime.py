"""Runtime helpers for selecting model execution devices and dtypes."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal


@lru_cache(maxsize=1)
def cuda_is_usable() -> bool:
    """Return True only when CUDA is both visible and executable."""
    try:
        import torch
    except Exception:
        return False

    try:
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False

    try:
        device_count = torch.cuda.device_count()
    except Exception:
        device_count = 0

    if device_count < 1:
        return False

    try:
        major, minor = torch.cuda.get_device_capability(0)
        current_arch = f"sm_{major}{minor}"
    except Exception:
        current_arch = None

    try:
        supported_arches = set(torch.cuda.get_arch_list() or [])
    except Exception:
        supported_arches = set()

    if current_arch and supported_arches and current_arch not in supported_arches:
        return False

    try:
        device = torch.device("cuda:0")
        probe = torch.tensor([1.0], device=device)
        _ = (probe + 1).sum().item()
        torch.cuda.synchronize()
    except Exception:
        return False

    return True


def select_device() -> Literal["cuda", "cpu"]:
    """Choose the best available torch device."""
    try:
        return "cuda" if cuda_is_usable() else "cpu"
    except Exception:
        return "cpu"


def select_dtype(device: str):
    """Choose an appropriate torch dtype for the selected device."""
    import torch

    return torch.float16 if device == "cuda" else torch.float32
