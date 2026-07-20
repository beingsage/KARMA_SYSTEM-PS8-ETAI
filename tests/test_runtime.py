import importlib
import sys
import types

from pytest import MonkeyPatch


class _FakeTensor:
    def __add__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 1.0


def _make_fake_torch(arch_list: list[str]) -> types.ModuleType:
    fake_torch = types.ModuleType("torch")

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_capability=lambda index=0: (6, 0),
        get_arch_list=lambda: arch_list,
        synchronize=lambda: None,
    )

    fake_torch.cuda = fake_cuda
    fake_torch.device = lambda spec: spec
    fake_torch.tensor = lambda data, device=None: _FakeTensor()
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.__version__ = "0.0.0"

    return fake_torch


def test_cuda_is_usable_rejects_arch_mismatch(monkeypatch: MonkeyPatch) -> None:
    fake_torch = _make_fake_torch(["sm_70"])
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from app.pipeline import runtime

    importlib.reload(runtime)

    try:
        assert runtime.cuda_is_usable() is False
        assert runtime.select_device() == "cpu"
    finally:
        runtime.cuda_is_usable.cache_clear()


def test_cuda_is_usable_accepts_matching_arch(monkeypatch: MonkeyPatch) -> None:
    fake_torch = _make_fake_torch(["sm_60"])
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from app.pipeline import runtime

    importlib.reload(runtime)

    try:
        assert runtime.cuda_is_usable() is True
        assert runtime.select_device() == "cuda"
    finally:
        runtime.cuda_is_usable.cache_clear()
