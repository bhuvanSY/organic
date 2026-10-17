"""
Baseline Reaction Fingerprint MLP Neural Network.
Implemented from scratch with randomly initialized weights.
Maps reactant Morgan fingerprints to predicted product fingerprints.
"""

from typing import List, Optional
import torch
import torch.nn as nn

from chemrxn.models.base import BaseReactionModel


class FingerprintBlock(nn.Module):
    """Feedforward residual block for molecular fingerprints."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc1(x)
        out = self.ln(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.fc2(out)
        return self.act(residual + out)


class FingerprintReactionMLP(BaseReactionModel):
    """
    Scratch-initialized multi-layer perceptron for reaction fingerprint prediction.
    No pretrained weights or chemical language models used.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 1024,
        output_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        blocks = [FingerprintBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        self.transformer_blocks = nn.Sequential(*blocks)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # Outputs predicted bit probabilities in [0, 1]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization for weights, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, input_dim) representing reactant fingerprints.
        Returns:
            Tensor of shape (batch_size, output_dim) representing predicted product bit probabilities.
        """
        h = self.input_proj(x)
        h = self.transformer_blocks(h)
        return self.output_proj(h)
