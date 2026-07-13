"""Compatibility helpers for torch, transformers, and pyarrow edge cases."""

from __future__ import annotations

import contextlib
import inspect
from typing import Iterator


def ensure_pyarrow_compat() -> bool:
    """Alias ``PyExtensionType`` for newer pyarrow releases when needed."""
    try:
        import pyarrow as pa
    except Exception:
        return False

    if not hasattr(pa, "PyExtensionType") and hasattr(pa, "ExtensionType"):
        try:
            pa.PyExtensionType = pa.ExtensionType
        except Exception:
            return False

    return hasattr(pa, "PyExtensionType")


def install_safe_torch_load_default() -> None:
    """Default ``torch.load`` to ``weights_only=True`` when supported."""
    try:
        import torch
    except Exception:
        return

    if getattr(torch, "_structured_safe_torch_load_installed", False):
        return

    try:
        signature = inspect.signature(torch.load)
    except Exception:
        return

    if "weights_only" not in signature.parameters:
        return

    original_load = getattr(torch, "_structured_original_torch_load", torch.load)

    def _safe_torch_load(f, *args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = True
        return original_load(f, *args, **kwargs)

    torch._structured_original_torch_load = original_load
    torch.load = _safe_torch_load
    torch._structured_safe_torch_load_installed = True


@contextlib.contextmanager
def allow_trusted_torch_pickle() -> Iterator[None]:
    """
    Temporarily restore the original ``torch.load`` and bypass transformers'
    torch-load safety gate for trusted local checkpoints.
    Also registers safe globals for UltraLytics and other trusted packages.
    """
    try:
        import torch
    except Exception:
        yield
        return

    # Register safe globals for UltraLytics YOLO models and their common modules.
    try:
        import torch.serialization as _ts
        try:
            from ultralytics.nn.tasks import DetectionModel
            _ts.add_safe_globals([DetectionModel])
        except Exception:
            pass

        try:
            import ultralytics.nn.modules as ultralytics_modules
            safe_globals = []
            for value in vars(ultralytics_modules).values():
                if inspect.isclass(value) and getattr(value, "__module__", "").startswith("ultralytics"):
                    safe_globals.append(value)
            if safe_globals:
                _ts.add_safe_globals(safe_globals)
        except Exception:
            pass
    except Exception:
        pass

    # Register additional commonly-used torch globals that appear in checkpoints
    # (e.g., Sequential, ModuleList, OrderedDict) so weights-only loading succeeds
    try:
        import torch.serialization as _ts
        from collections import OrderedDict
        from torch.nn.modules.container import ModuleList, Sequential

        safe_list = [Sequential, ModuleList, OrderedDict]
        if safe_list:
            try:
                _ts.add_safe_globals(safe_list)
            except Exception:
                try:
                    _ts.safe_globals(safe_list)
                except Exception:
                    pass
    except Exception:
        pass

    original_load = getattr(torch, "_structured_original_torch_load", torch.load)
    current_load = torch.load

    def _trusted_torch_load(f, *args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(f, *args, **kwargs)

    patched_checks: list[tuple[object, str, object]] = []

    try:
        torch.load = _trusted_torch_load

        try:
            import transformers.utils.import_utils as import_utils

            if hasattr(import_utils, "check_torch_load_is_safe"):
                patched_checks.append(
                    (import_utils, "check_torch_load_is_safe", import_utils.check_torch_load_is_safe)
                )
                import_utils.check_torch_load_is_safe = lambda: None
        except Exception:
            pass

        try:
            import transformers.modeling_utils as modeling_utils

            if hasattr(modeling_utils, "check_torch_load_is_safe"):
                patched_checks.append(
                    (modeling_utils, "check_torch_load_is_safe", modeling_utils.check_torch_load_is_safe)
                )
                modeling_utils.check_torch_load_is_safe = lambda: None
        except Exception:
            pass

        yield
    finally:
        for module, attribute, original_value in reversed(patched_checks):
            setattr(module, attribute, original_value)
        torch.load = current_load
