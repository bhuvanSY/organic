"""
Unit tests for scratch-initialized neural network architectures.
Verifies no pretrained leakage and correct parameter initialization.
"""

import pytest
import torch

from chemrxn.models.fingerprint_mlp import FingerprintReactionMLP
from chemrxn.models.seq2seq_transformer import Seq2SeqReactionTransformer


def test_fingerprint_mlp_scratch_initialization():
    model = FingerprintReactionMLP(input_dim=256, hidden_dim=128, output_dim=256, num_layers=2)
    assert not model.is_pretrained
    params = model.count_parameters()
    assert params["total_parameters"] > 0
    assert params["trainable_parameters"] == params["total_parameters"]

    x = torch.randn(2, 256)
    out = model(x)
    assert out.shape == (2, 256)
    # Sigmoid outputs must be within [0, 1]
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_transformer_scratch_initialization():
    model = Seq2SeqReactionTransformer(
        vocab_size=50,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
    )
    assert not model.is_pretrained
    params = model.count_parameters()
    assert params["total_parameters"] > 0

    src = torch.randint(0, 50, (2, 16))
    tgt = torch.randint(0, 50, (2, 16))
    logits = model(src, tgt)
    assert logits.shape == (2, 16, 50)
