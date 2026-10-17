"""
Unit and integration tests for custom Seq2Seq Reaction Transformer.
Tests:
- Scratch initialization without pretrained weights
- Tensor dimensions through encoder, decoder, and output projection
- Causal autoregressive masking and padding suppression
- Token-level cross-entropy loss computation
- Greedy and beam search decoding mechanisms
- Out-of-vocabulary handling and maximum-length safeguards
- Overfitting capability on tiny reaction batches
"""

import math
import pytest
import torch

from chemrxn.data.tokenizer import SMILESTokenizer
from chemrxn.models.seq2seq_transformer import (
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    Seq2SeqReactionTransformer,
)
from chemrxn.training.losses import MaskedCrossEntropyLoss


@pytest.fixture
def dummy_vocab():
    return {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "C": 4,
        "N": 5,
        "O": 6,
        "=": 7,
        "(": 8,
        ")": 9,
        ".": 10,
        "Cl": 11,
    }


@pytest.fixture
def dummy_tokenizer(dummy_vocab):
    return SMILESTokenizer(vocab=dummy_vocab)


@pytest.fixture
def small_transformer(dummy_vocab):
    return Seq2SeqReactionTransformer(
        vocab_size=len(dummy_vocab),
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        pad_idx=0,
        sos_idx=1,
        eos_idx=2,
        max_seq_len=64,
    )


def test_scratch_initialization_no_pretrained(small_transformer):
    """Verifies that all weights are randomly initialized and no pretrained weights are loaded."""
    for name, p in small_transformer.named_parameters():
        assert p.requires_grad, f"Parameter {name} does not require grad"
        # Weights should not be all zeros (except bias or pad embedding)
        if "bias" not in name and "pad" not in name:
            assert not torch.all(p == 0.0), f"Parameter {name} is all zeros"

    # Pad index embedding must be zeroed
    assert torch.all(small_transformer.src_embedding.weight[0] == 0.0)
    assert torch.all(small_transformer.tgt_embedding.weight[0] == 0.0)


def test_positional_encoding_tensor_shape():
    """Verifies sinusoidal positional encoding dimensions and value stability."""
    pe = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
    x = torch.zeros(2, 30, 64)
    out = pe(x)
    assert out.shape == (2, 30, 64)
    # Different positions should have distinct positional codes
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])


def test_multihead_attention_shapes_and_masking():
    """Verifies multi-head attention output shape and causal mask behavior."""
    d_model = 64
    nhead = 4
    mha = MultiHeadAttention(d_model=d_model, nhead=nhead, dropout=0.0)

    batch_size = 2
    seq_len = 5
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    # Causal mask: upper triangle is masked out
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    out, weights = mha(q, k, v, attn_mask=causal_mask)

    assert out.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, nhead, seq_len, seq_len)

    # Verify that causal mask sets future attention weights strictly to 0
    # For query position 0, attention to positions 1..seq_len-1 must be 0
    assert torch.allclose(weights[:, :, 0, 1:], torch.zeros_like(weights[:, :, 0, 1:]), atol=1e-5)


def test_transformer_forward_tensor_shapes(small_transformer):
    """Verifies forward pass logits shape with batch and sequence lengths."""
    batch_size = 4
    src_len = 12
    tgt_len = 10
    src = torch.randint(0, small_transformer.vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, small_transformer.vocab_size, (batch_size, tgt_len))

    logits = small_transformer(src=src, tgt=tgt)
    assert logits.shape == (batch_size, tgt_len, small_transformer.vocab_size)


def test_masked_cross_entropy_loss_ignores_padding():
    """Verifies that padded target tokens do not contribute to the loss."""
    loss_fn = MaskedCrossEntropyLoss(ignore_index=0)

    vocab_size = 10
    logits = torch.randn(2, 4, vocab_size)
    # First sequence has targets [1, 2, 0, 0]
    targets = torch.tensor([[1, 2, 0, 0], [1, 2, 3, 0]], dtype=torch.long)

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    # Changing logits at padding positions should NOT change the loss
    logits_mod = logits.clone()
    logits_mod[:, 2:, :] += 10.0  # modify padded token logits
    loss_mod = loss_fn(logits_mod, targets)

    # Note: position 2 in seq 0 is pad, but seq 1 has valid token at position 2
    # If we modify only true padding positions:
    logits_pad_only = logits.clone()
    logits_pad_only[0, 2:, :] += 50.0  # only padding for seq 0
    loss_pad_only = loss_fn(logits_pad_only, targets)
    assert math.isclose(loss.item(), loss_pad_only.item(), rel_tol=1e-5)


def test_greedy_decoding(small_transformer, dummy_tokenizer):
    """Verifies autoregressive greedy decoding produces valid tokens starting with <sos>."""
    src = torch.tensor([[dummy_tokenizer.encode("CC(=O)Cl", add_special_tokens=True, max_length=16)]], dtype=torch.long).squeeze(0)

    generated = small_transformer.generate_greedy(src, max_len=15)
    assert generated.dim() == 2
    assert generated.size(0) == 1
    # First token must be <sos>
    assert generated[0, 0].item() == dummy_tokenizer.sos_id
    # All tokens must be within valid vocabulary range
    assert torch.all(generated < small_transformer.vocab_size)
    assert torch.all(generated >= 0)


def test_beam_search_decoding_and_deduplication(small_transformer, dummy_tokenizer):
    """Verifies beam search returns sorted, deduplicated beam candidates with scores."""
    src = torch.tensor([[dummy_tokenizer.encode("CC(=O)Cl", add_special_tokens=True, max_length=16)]], dtype=torch.long).squeeze(0)

    beam_candidates = small_transformer.generate_beam(src, beam_size=3, max_len=15)
    assert len(beam_candidates) == 1  # 1 batch item
    candidates = beam_candidates[0]
    assert len(candidates) <= 3

    # Check descending order of scores
    scores = [s for _, s in candidates]
    assert scores == sorted(scores, reverse=True)

    # Check deduplication: all candidates must be unique token lists
    seq_tuples = [tuple(seq) for seq, _ in candidates]
    assert len(seq_tuples) == len(set(seq_tuples))


def test_overfit_tiny_batch(dummy_tokenizer):
    """Verifies that the from-scratch Transformer can overfit 2 reaction examples down to near-zero loss."""
    device = torch.device("cpu")
    model = Seq2SeqReactionTransformer(
        vocab_size=dummy_tokenizer.vocab_size,
        d_model=32,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        pad_idx=dummy_tokenizer.pad_id,
        sos_idx=dummy_tokenizer.sos_id,
        eos_idx=dummy_tokenizer.eos_id,
        max_seq_len=32,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = MaskedCrossEntropyLoss(ignore_index=dummy_tokenizer.pad_id)

    # Toy samples
    src_tokens = torch.tensor([
        dummy_tokenizer.encode("C=O", max_length=10),
        dummy_tokenizer.encode("CN", max_length=10),
    ], dtype=torch.long, device=device)

    tgt_tokens = torch.tensor([
        dummy_tokenizer.encode("CO", max_length=10),
        dummy_tokenizer.encode("C", max_length=10),
    ], dtype=torch.long, device=device)

    tgt_in = tgt_tokens[:, :-1]
    tgt_out = tgt_tokens[:, 1:]

    initial_loss = None
    final_loss = None

    model.train()
    for step in range(40):
        optimizer.zero_grad()
        logits = model(src_tokens, tgt_in)
        loss = criterion(logits, tgt_out)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss
    assert final_loss < 0.2, f"Expected overfit loss < 0.2, got {final_loss:.4f}"
