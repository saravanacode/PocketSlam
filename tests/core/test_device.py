import pytest
import torch

from pocketslam.core import device


def test_auto_picks_first_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "available_backends", lambda: ["mps", "cpu"])
    assert device.select_device() == torch.device("mps")


def test_cpu_always_available() -> None:
    assert device.available_backends()[-1] == "cpu"
    assert device.select_device("CPU") == torch.device("cpu")


def test_unknown_backend_rejected() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        device.select_device("tpu")


def test_unavailable_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device, "available_backends", lambda: ["cpu"])
    with pytest.raises(RuntimeError, match="only"):
        device.select_device("cuda")
