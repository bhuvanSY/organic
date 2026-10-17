"""
Neural Reinforcement Learning Policy for Candidate Selection and Abstention.
Implements:
1. Permutation-equivariant candidate encoder
2. Cross-attention with reactant context
3. Explicit abstention action logit
4. Value network head for variance reduction in Policy Gradients
5. Randomly initialized weights (zero pretrained predictive weights)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PolicyOutput:
    logits: torch.Tensor          # [K + 1] action logits
    probs: torch.Tensor           # [K + 1] action probabilities
    value: torch.Tensor           # Scalar state-value estimate V(s)
    dist: Categorical


class RLCandidatePolicy(nn.Module):
    """
    Actor-Critic Policy Network for discrete candidate selection among K options or Abstaining.
    Actions:
      0 .. K-1: Select candidate k
      K: Abstain
    """
    def __init__(
        self,
        context_dim: int = 8,
        candidate_dim: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.candidate_dim = candidate_dim
        self.hidden_dim = hidden_dim

        # Context Encoder (Reactant properties & candidate pool statistics)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Candidate Encoder (Per-candidate multi-modal scores & physical features)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Candidate Scoring Head (combines per-candidate state + dot product with context)
        self.candidate_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Abstention Head (evaluates context + pool summary to decide if abstaining is optimal)
        self.abstain_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Critic / Value Head V(s) for Advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Random weight initialization from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        context_tensor: torch.Tensor,     # [D_ctx] or [B, D_ctx]
        candidate_tensor: torch.Tensor,   # [K, D_cand] or [B, K, D_cand]
    ) -> PolicyOutput:
        """
        Computes action logits for K candidates + 1 abstention action.
        """
        # Ensure batch dimension for single inputs
        is_batched = context_tensor.dim() == 2
        if not is_batched:
            context_tensor = context_tensor.unsqueeze(0)        # [1, D_ctx]
            candidate_tensor = candidate_tensor.unsqueeze(0)    # [1, K, D_cand]

        b, k, _ = candidate_tensor.shape

        h_ctx = self.context_encoder(context_tensor)            # [B, hidden_dim]
        h_cand = self.candidate_encoder(candidate_tensor)       # [B, K, hidden_dim]

        # Candidate selection logits: intrinsic score + cross-attention dot product with context
        cand_intrinsic = self.candidate_scorer(h_cand).squeeze(-1) # [B, K]
        # Dot product: h_cand @ h_ctx
        dot_scores = torch.bmm(h_cand, h_ctx.unsqueeze(-1)).squeeze(-1) / np.sqrt(self.hidden_dim) # [B, K]
        cand_logits = cand_intrinsic + dot_scores # [B, K]

        # Pool candidate representations for abstention & value estimation
        cand_pool = torch.mean(h_cand, dim=1) # [B, hidden_dim]
        state_repr = torch.cat([h_ctx, cand_pool], dim=-1) # [B, 2 * hidden_dim]

        # Abstention logit
        abstain_logit = self.abstain_head(state_repr) # [B, 1]

        # Total action logits: [B, K + 1]
        all_logits = torch.cat([cand_logits, abstain_logit], dim=-1)

        # Critic value V(s)
        value = self.value_head(state_repr).squeeze(-1) # [B]

        if not is_batched:
            all_logits = all_logits.squeeze(0)
            value = value.squeeze(0)

        probs = F.softmax(all_logits, dim=-1)
        dist = Categorical(probs=probs)

        return PolicyOutput(
            logits=all_logits,
            probs=probs,
            value=value,
            dist=dist,
        )

    def act(
        self,
        context_tensor: torch.Tensor,
        candidate_tensor: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[int, float, torch.Tensor, float]:
        """
        Samples an action given current episode context and candidates.
        Returns:
            (action_idx, action_prob, log_prob, state_value)
        """
        out = self.forward(context_tensor, candidate_tensor)
        if deterministic:
            action = int(torch.argmax(out.probs).item())
        else:
            action = int(out.dist.sample().item())

        action_prob = float(out.probs[action].item())
        log_prob = out.dist.log_prob(torch.tensor(action, device=context_tensor.device))
        val = float(out.value.item())

        return action, action_prob, log_prob, val

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict] = None):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "context_dim": self.context_dim,
                "candidate_dim": self.candidate_dim,
                "hidden_dim": self.hidden_dim,
            },
            "extra_meta": extra_meta or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "RLCandidatePolicy":
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint["config"]
        policy = cls(
            context_dim=cfg["context_dim"],
            candidate_dim=cfg["candidate_dim"],
            hidden_dim=cfg["hidden_dim"],
        )
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.to(device)
        return policy
