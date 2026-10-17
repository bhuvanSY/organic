"""
Hardware acceleration and device management utilities.
Supports NVIDIA CUDA with safe CPU fallback.
"""

from typing import Any, Dict
import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Resolves the torch.device to use.

    Args:
        preference: "auto", "cuda", or "cpu".

    Returns:
        torch.device instance.
    """
    pref = preference.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {preference}. Must be 'auto', 'cuda', or 'cpu'.")


def get_device_info() -> Dict[str, Any]:
    """
    Collects comprehensive hardware capability information.
    """
    cuda_available = torch.cuda.is_available()
    info: Dict[str, Any] = {
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "device_type": "cuda" if cuda_available else "cpu",
    }

    if cuda_available:
        current_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(current_idx)
        total_memory_gb = round(props.total_memory / (1024 ** 3), 2)
        allocated_mb = round(torch.cuda.memory_allocated(current_idx) / (1024 ** 2), 2)
        reserved_mb = round(torch.cuda.memory_reserved(current_idx) / (1024 ** 2), 2)

        info.update({
            "device_index": current_idx,
            "device_name": props.name,
            "total_memory_gb": total_memory_gb,
            "allocated_memory_mb": allocated_mb,
            "reserved_memory_mb": reserved_mb,
            "multi_processor_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}",
            "supports_fp16": props.major >= 7,
            "supports_bf16": props.major >= 8,
        })
    else:
        import platform
        import os
        info.update({
            "device_name": platform.processor() or "CPU",
            "cpu_count": os.cpu_count(),
            "total_memory_gb": "N/A",
        })

    return info


def to_device(data: Any, device: torch.device) -> Any:
    """
    Recursively moves tensors or collections of tensors (dict, list, tuple) to device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        converted = [to_device(item, device) for item in data]
        return type(data)(converted)
    return data
