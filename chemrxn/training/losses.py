"""
Loss functions for chemical sequence and fingerprint predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with padding index masking and optional label smoothing.
    """

    def __init__(self, ignore_index: int = 0, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            targets: Tensor of shape (batch_size, seq_len)
        """
        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)

        return F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class FingerprintLoss(nn.Module):
    """
    Binary Cross-Entropy loss combined with continuous Tanimoto/Dice surrogate
    for fingerprint bit predictions.
    """

    def __init__(self, bce_weight: float = 0.7, dice_weight: float = 0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(preds, targets)
        
        # Soft continuous Tanimoto similarity loss: 1 - (intersection / union)
        intersection = (preds * targets).sum(dim=-1)
        union = preds.sum(dim=-1) + targets.sum(dim=-1) - intersection
        tanimoto_loss = 1.0 - ((intersection + 1e-6) / (union + 1e-6)).mean()

        return self.bce_weight * bce_loss + self.dice_weight * tanimoto_loss
