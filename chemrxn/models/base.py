"""
Base abstract class for all ChemRxn-Intelligence neural network models.
Enforces scratch random initialization and prevents pretrained chemical model leakage.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


class BaseReactionModel(nn.Module, ABC):
    """
    Abstract base class for scratch-initialized reaction prediction models.
    """

    def __init__(self):
        super().__init__()
        self.is_pretrained = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward computation pass."""
        pass

    @abstractmethod
    def _init_weights(self) -> None:
        """
        Initializes all model weights from scratch using standard random distributions.
        Guarantees no pretrained weights are loaded.
        """
        pass

    def count_parameters(self) -> Dict[str, int]:
        """
        Returns total and trainable parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": total - trainable,
        }

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves model weights and training state.
        """
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_class": self.__class__.__name__,
            "epoch": epoch,
            "metrics": metrics or {},
            "extra_config": extra_config or {},
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, p)

    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> Dict[str, Any]:
        """
        Loads model weights from a local checkpoint.
        """
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        checkpoint = torch.load(p, map_location=map_location or "cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
