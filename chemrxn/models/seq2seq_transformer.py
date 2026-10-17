"""
Seq2Seq Transformer for SMILES-to-SMILES Organic Reaction Prediction.
Built from scratch in PyTorch with every predictive weight randomly initialized.
Zero pretrained models, chemical language models, or hosted predictive APIs.

Includes:
- Token and positional embeddings
- Custom Multi-Head Self-Attention and Cross-Attention
- Position-wise Feed-Forward Networks with GELU
- Residual connections and Pre-Layer Normalization
- Source padding masks and causal autoregressive decoder masks
- Output projection layer with optional weight tying
- Autoregressive Greedy Decoding and Beam Search Decoding with deduplication
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemrxn.models.base import BaseReactionModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as defined in Vaswani et al. (Attention Is All You Need).
    Adds positional coordinates to token embeddings without trainable weights.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional signal added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention supporting both self-attention and cross-attention.
    Projects Q, K, V, computes scaled dot-product attention with arbitrary masks,
    and projects the concatenated heads back to d_model.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, tgt_len, d_model)
            key: (batch_size, src_len, d_model)
            value: (batch_size, src_len, d_model)
            attn_mask: 2D (tgt_len, src_len) or 4D mask where True or -inf marks invalid positions
            key_padding_mask: (batch_size, src_len) where True indicates padding token
        Returns:
            out: (batch_size, tgt_len, d_model)
            attn_weights: (batch_size, nhead, tgt_len, src_len)
        """
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        # 1. Linear projections and split into heads: (B, H, L, d_k)
        q = self.q_proj(query).view(batch_size, tgt_len, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, src_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, src_len, self.nhead, self.d_k).transpose(1, 2)

        # 2. Scaled dot-product: (B, H, tgt_len, src_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Apply causal subsequent mask if provided
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # Shape: (tgt_len, src_len) -> (1, 1, tgt_len, src_len)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask

        # 4. Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask shape: (batch_size, src_len) -> (batch_size, 1, 1, src_len)
            mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, float("-inf"))

        # 5. Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        # Numerical guard: if an entire row was masked (e.g. all-pad query), softmax yields NaNs
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_dropped = self.dropout(attn_weights)

        # 6. Weighted sum and re-combine heads: (B, tgt_len, d_model)
        context = torch.matmul(attn_dropped, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.d_model)

        out = self.out_proj(context)
        return out, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise two-layer Feed-Forward Network with GELU activation.
    FFN(x) = max(0, x W1 + b1) W2 + b2
    """

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))


class TransformerEncoderLayer(nn.Module):
    """
    Single Encoder Layer using Pre-LayerNorm architecture:
    x -> LN -> MultiHeadSelfAttention -> Dropout + Residual
      -> LN -> FeedForward -> Dropout + Residual
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        normed = self.norm1(src)
        attn_out, _ = self.self_attn(normed, normed, normed, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_out)

        # Pre-LN Feed-Forward
        ffn_out = self.feed_forward(self.norm2(src))
        src = src + ffn_out
        return src


class TransformerDecoderLayer(nn.Module):
    """
    Single Decoder Layer using Pre-LayerNorm architecture:
    1. Masked Causal Multi-Head Self-Attention
    2. Multi-Head Cross-Attention attending to encoder memory
    3. Position-wise Feed-Forward Network
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. Masked Self-Attention
        normed = self.norm1(tgt)
        self_out, _ = self.self_attn(
            normed, normed, normed,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout(self_out)

        # 2. Cross-Attention
        normed = self.norm2(tgt)
        cross_out, _ = self.cross_attn(
            normed, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout(cross_out)

        # 3. Position-wise FFN
        ffn_out = self.feed_forward(self.norm3(tgt))
        tgt = tgt + ffn_out
        return tgt


class Seq2SeqReactionTransformer(BaseReactionModel):
    """
    Full Encoder-Decoder Transformer for Organic Reaction Prediction.
    Trained from scratch without pretrained weights.

    Translates:
        Input: Reactants (+ optional reagents/catalysts) -> SMILES tokens
        Output: Product -> SMILES tokens
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        sos_idx: int = 1,
        eos_idx: int = 2,
        max_seq_len: int = 256,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_seq_len = max_seq_len

        # Embeddings & Positional Encodings
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output projection head: d_model -> vocab_size
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.output_head.weight = self.tgt_embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Randomly initializes all model weights from scratch.
        Uses Xavier uniform initialization for projection matrices and scaled normal for embeddings.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Embedding initialization
        scale = math.sqrt(1.0 / self.d_model)
        nn.init.normal_(self.src_embedding.weight, mean=0.0, std=scale)
        nn.init.normal_(self.tgt_embedding.weight, mean=0.0, std=scale)

        if self.pad_idx is not None:
            with torch.no_grad():
                self.src_embedding.weight[self.pad_idx].fill_(0.0)
                self.tgt_embedding.weight[self.pad_idx].fill_(0.0)

    @staticmethod
    def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """
        Generates lower-triangular causal mask to prevent decoder from looking ahead.
        Returns BoolTensor where True indicates positions that should be masked out.
        """
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encodes source reactant tokens into context memory.
        Args:
            src: (batch_size, src_len)
            src_key_padding_mask: (batch_size, src_len)
        Returns:
            memory: (batch_size, src_len, d_model)
        """
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        out = src_emb
        for layer in self.encoder_layers:
            out = layer(out, src_key_padding_mask=src_key_padding_mask)

        return self.encoder_norm(out)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decodes target prefix tokens conditioned on encoder context memory.
        Args:
            tgt: (batch_size, tgt_len)
            memory: (batch_size, src_len, d_model)
        Returns:
            hidden: (batch_size, tgt_len, d_model)
        """
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        out = tgt_emb
        for layer in self.decoder_layers:
            out = layer(
                out,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        return self.decoder_norm(out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass.
        Args:
            src: (batch_size, src_len)
            tgt: (batch_size, tgt_len)
            src_key_padding_mask: (batch_size, src_len)
            tgt_key_padding_mask: (batch_size, tgt_len)
        Returns:
            logits: (batch_size, tgt_len, vocab_size)
        """
        device = src.device
        tgt_len = tgt.size(1)

        # If padding masks not explicitly provided, create from pad_idx
        if src_key_padding_mask is None:
            src_key_padding_mask = (src == self.pad_idx)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = (tgt == self.pad_idx)

        # Causal mask for autoregressive teacher forcing
        causal_mask = self.generate_causal_mask(tgt_len, device=device)

        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        dec_out = self.decode(
            tgt,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        logits = self.output_head(dec_out)
        return logits

    # -------------------------------------------------------------------------
    # Autoregressive Decoding Methods
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def generate_greedy(
        self,
        src: torch.Tensor,
        max_len: int = 140,
    ) -> torch.Tensor:
        """
        Autoregressively generates product tokens using greedy decoding (argmax).
        Args:
            src: (batch_size, src_len)
            max_len: maximum generation steps
        Returns:
            generated_ids: (batch_size, generated_len)
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)

        src_padding_mask = (src == self.pad_idx)
        memory = self.encode(src, src_key_padding_mask=src_padding_mask)

        # Initialize target with <sos> token: (B, 1)
        generated = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_len = generated.size(1)
            causal_mask = self.generate_causal_mask(tgt_len, device=device)

            dec_out = self.decode(
                generated,
                memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            # Predict next token from the last position
            next_logits = self.output_head(dec_out[:, -1, :])  # (B, vocab_size)
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (B, 1)

            # Mask out newly generated tokens for already finished sequences
            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, self.pad_idx), next_token)
            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == self.eos_idx)
            if finished.all():
                break

        return generated

    @torch.no_grad()
    def generate_beam(
        self,
        src: torch.Tensor,
        beam_size: int = 4,
        max_len: int = 140,
        length_penalty_alpha: float = 0.6,
    ) -> List[List[Tuple[List[int], float]]]:
        """
        Beam search decoding with length penalty and duplicate candidate removal.
        Args:
            src: (batch_size, src_len)
            beam_size: number of candidate beams to track
            max_len: maximum generation length
            length_penalty_alpha: exponent alpha for length normalization
        Returns:
            List of batches, where each batch item contains sorted tuples of (token_id_list, normalized_score)
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)

        results: List[List[Tuple[List[int], float]]] = []

        for b in range(batch_size):
            src_b = src[b : b + 1]  # (1, src_len)
            src_mask_b = (src_b == self.pad_idx)
            memory_b = self.encode(src_b, src_key_padding_mask=src_mask_b)

            # Each beam hypothesis: (token_ids_tensor, log_prob_sum, is_finished)
            beams = [(torch.tensor([[self.sos_idx]], dtype=torch.long, device=device), 0.0, False)]
            completed_beams: List[Tuple[List[int], float]] = []

            for _ in range(max_len):
                candidates: List[Tuple[torch.Tensor, float, bool]] = []
                all_finished = True

                for seq, score, is_done in beams:
                    if is_done:
                        candidates.append((seq, score, True))
                        continue

                    all_finished = False
                    tgt_len = seq.size(1)
                    causal_mask = self.generate_causal_mask(tgt_len, device=device)

                    dec_out = self.decode(
                        seq,
                        memory_b,
                        tgt_mask=causal_mask,
                        memory_key_padding_mask=src_mask_b,
                    )
                    next_logits = self.output_head(dec_out[:, -1, :])  # (1, vocab_size)
                    log_probs = F.log_softmax(next_logits, dim=-1).squeeze(0)  # (vocab_size,)

                    # Top beam_size expansions
                    topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)

                    for lp, tok in zip(topk_log_probs, topk_tokens):
                        tok_id = tok.item()
                        new_seq = torch.cat([seq, torch.tensor([[tok_id]], dtype=torch.long, device=device)], dim=1)
                        new_score = score + lp.item()
                        is_eos = (tok_id == self.eos_idx)
                        candidates.append((new_seq, new_score, is_eos))

                if all_finished:
                    break

                # Sort candidates by length-normalized score
                def score_fn(cand):
                    seq, score, _ = cand
                    length = seq.size(1)
                    lp = ((5.0 + length) / 6.0) ** length_penalty_alpha
                    return score / lp

                candidates.sort(key=score_fn, reverse=True)

                # Prune and deduplicate candidates
                seen_sequences = set()
                new_beams = []
                for cand in candidates:
                    seq_list = cand[0][0].tolist()
                    seq_tuple = tuple(seq_list)
                    if seq_tuple not in seen_sequences:
                        seen_sequences.add(seq_tuple)
                        new_beams.append(cand)
                    if len(new_beams) == beam_size:
                        break

                beams = new_beams

            # Harvest finished and active beams
            final_candidates: List[Tuple[List[int], float]] = []
            for seq, score, _ in beams:
                seq_list = seq[0].tolist()
                length = len(seq_list)
                lp = ((5.0 + length) / 6.0) ** length_penalty_alpha
                norm_score = score / lp
                final_candidates.append((seq_list, round(norm_score, 4)))

            final_candidates.sort(key=lambda item: item[1], reverse=True)
            results.append(final_candidates)

        return results

