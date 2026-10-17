"""
Deterministic seed utilities for reproducible chemistry research.
"""

import os
import random
from typing import Optional
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> int:
    """
    Sets deterministic seeds across Python standard library, NumPy, and PyTorch.
    Ensures scientific reproducibility across training runs and tests.

    Args:
        seed: Integer seed value.
        deterministic_cudnn: If True, configures cuDNN to use deterministic algorithms.

    Returns:
        The seed used.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
