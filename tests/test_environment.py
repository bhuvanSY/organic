"""
Unit tests for environment, seed determinism, and hardware fallback.
"""

import numpy as np
import pytest
import torch

from chemrxn.utils.device import get_device, get_device_info, to_device
from chemrxn.utils.seed import seed_everything


def test_seed_everything_determinism():
    seed_everything(123)
    a1 = torch.randn(10, 10)
    np1 = np.random.rand(10)

    seed_everything(123)
    a2 = torch.randn(10, 10)
    np2 = np.random.rand(10)

    assert torch.equal(a1, a2)
    assert np.array_equal(np1, np2)


def test_get_device_resolution():
    cpu_dev = get_device("cpu")
    assert cpu_dev.type == "cpu"

    auto_dev = get_device("auto")
    assert auto_dev.type in ("cuda", "cpu")

    with pytest.raises(ValueError):
        get_device("invalid_device_name")


def test_get_device_info():
    info = get_device_info()
    assert "cuda_available" in info
    assert "torch_version" in info
    assert "device_type" in info
    assert info["device_type"] in ("cuda", "cpu")


def test_to_device_tensor_and_nested():
    cpu = torch.device("cpu")
    t = torch.randn(4, 4)
    data = {
        "tensor": t,
        "list": [t, t],
        "number": 42,
    }
    moved = to_device(data, cpu)
    assert moved["tensor"].device == cpu
    assert moved["list"][0].device == cpu
    assert moved["number"] == 42
