"""Runtime helpers for selecting model execution devices and dtypes."""

from __future__ import annotations

from typing import Literal


def select_device() -> Literal["cuda", "cpu"]:
    """Choose the best available torch device."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def select_dtype(device: str):
    """Choose an appropriate torch dtype for the selected device."""
    import torch

    return torch.float16 if device == "cuda" else torch.float32
