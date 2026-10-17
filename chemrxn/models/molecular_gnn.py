"""
From-scratch Molecular Graph Neural Network (MPNN) in PyTorch.
Implements:
1. Edge-conditioned Message Passing layers (MPNNLayer)
2. Graph pooling (mean + sum readout)
3. Atom-level Reaction-Center Prediction Head
4. Graph-Conditioned Product-Candidate Scoring Head (Reaction-Pair Representation)
5. Zero pretrained weights with full PyTorch tensor vectorization
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemrxn.chemistry.graphs import (
    BatchMolecularGraph,
    MolecularGraph,
    NODE_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    batch_molecular_graphs,
    mol_to_graph,
)


class MPNNLayer(nn.Module):
    """
    Edge-conditioned message passing layer.
    Computes messages m_uv = MLP([h_u, h_v, e_uv]),
    aggregates via sum-scatter, and updates node states with Pre-LayerNorm residual connection.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, hidden_dim]
        """
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)

        if num_edges > 0:
            src, dst = edge_index[0], edge_index[1]
            # Form edge inputs: [h_src, h_dst, edge_attr]
            msg_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)
            messages = self.message_mlp(msg_input)  # [num_edges, hidden_dim]

            # Vectorized aggregation using native PyTorch index_add_ (match dtype for AMP)
            agg_messages = torch.zeros(num_nodes, self.hidden_dim, device=h.device, dtype=messages.dtype)
            agg_messages.index_add_(0, dst, messages)
        else:
            # Handle molecules with 0 bonds (e.g. isolated ions or single atoms)
            agg_messages = torch.zeros(num_nodes, self.hidden_dim, device=h.device, dtype=h.dtype)

        # Update node representation
        update_input = torch.cat([h, agg_messages], dim=-1)
        delta_h = self.update_mlp(update_input)
        h_out = self.norm(h + delta_h)
        return h_out


class ReactionCenterHead(nn.Module):
    """
    Predicts whether an atom is part of the reaction center (undergoes bond change / participates in reaction).
    Evaluated on atom embeddings h_v in the reactant molecular graph.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_nodes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_nodes: [total_nodes, hidden_dim]
        Returns:
            logits: [total_nodes]
        """
        logits = self.mlp(h_nodes).squeeze(-1)
        return logits


class ReactionPairScorer(nn.Module):
    """
    Scores compatibility between a reactant graph representation and a candidate product graph representation.
    Reaction-pair representation: [h_R, h_P, |h_R - h_P|, h_R * h_P] -> MLP -> Scalar Score.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_reactants: torch.Tensor, h_products: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_reactants: [B, hidden_dim]
            h_products: [B, hidden_dim]
        Returns:
            scores: [B]
        """
        diff = torch.abs(h_reactants - h_products)
        prod = h_reactants * h_products
        pair_repr = torch.cat([h_reactants, h_products, diff, prod], dim=-1)
        scores = self.mlp(pair_repr).squeeze(-1)
        return scores


class MolecularReactionGNN(nn.Module):
    """
    End-to-end Molecular Reaction Graph Neural Network.
    - Zero pretrained weights (random initialization).
    - Disjoint-union graph batching.
    - Multi-task: Reaction center prediction + Product candidate ranking.
    """
    def __init__(
        self,
        node_in_dim: int = NODE_FEATURE_DIM,
        edge_in_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding projections
        self.node_embedding = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Message Passing Layers
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Graph Pooling Readout (Mean + Sum)
        self.readout_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Heads
        self.reaction_center_head = ReactionCenterHead(hidden_dim=hidden_dim, dropout=dropout)
        self.candidate_scorer = ReactionPairScorer(hidden_dim=hidden_dim, dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        """Random weight initialization from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def encode_graph(self, batch_graph: BatchMolecularGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs message passing and pools graph representations.
        Returns:
            h_nodes: [total_nodes, hidden_dim]
            h_graph: [num_graphs, hidden_dim]
        """
        x = batch_graph.x
        edge_index = batch_graph.edge_index
        edge_attr = batch_graph.edge_attr
        batch = batch_graph.batch
        num_graphs = batch_graph.num_graphs

        h = self.node_embedding(x)
        if edge_attr.size(0) > 0:
            e = self.edge_embedding(edge_attr)
        else:
            e = torch.zeros(0, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Message passing layers
        for layer in self.layers:
            h = layer(h, edge_index, e)

        # Global Pooling (Mean + Sum)
        # Sum pool
        h_sum = torch.zeros(num_graphs, self.hidden_dim, device=x.device, dtype=x.dtype)
        h_sum.index_add_(0, batch, h)

        # Counts for Mean pool
        counts = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        counts.index_add_(0, batch, torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype))
        counts = torch.clamp(counts, min=1.0)
        h_mean = h_sum / counts

        # Pooled graph embedding
        h_graph = self.readout_mlp(torch.cat([h_mean, h_sum], dim=-1))
        return h, h_graph

    def predict_reaction_center_logits(self, reactant_batch_graph: BatchMolecularGraph) -> torch.Tensor:
        """Computes per-atom reaction center logits for reactants."""
        h_nodes, _ = self.encode_graph(reactant_batch_graph)
        logits = self.reaction_center_head(h_nodes)
        return logits

    def score_candidates(
        self,
        reactant_batch_graph: BatchMolecularGraph,
        candidate_batch_graph: BatchMolecularGraph,
    ) -> torch.Tensor:
        """
        Computes compatibility score for corresponding (reactant, candidate_product) pairs.
        Returns:
            scores: [num_graphs]
        """
        _, h_r = self.encode_graph(reactant_batch_graph)
        _, h_c = self.encode_graph(candidate_batch_graph)
        scores = self.candidate_scorer(h_r, h_c)
        return scores

    def score_candidate_pool(
        self,
        reactant_graph: MolecularGraph,
        candidate_graphs: List[MolecularGraph],
    ) -> torch.Tensor:
        """
        Convenience method to score a single reactant against K candidate products.
        Returns:
            scores: [K]
        """
        if not candidate_graphs:
            return torch.empty(0)

        k = len(candidate_graphs)
        reactant_batched = batch_molecular_graphs([reactant_graph] * k)
        candidate_batched = batch_molecular_graphs(candidate_graphs)

        device = next(self.parameters()).device
        reactant_batched = reactant_batched.to(device)
        candidate_batched = candidate_batched.to(device)

        with torch.no_grad():
            scores = self.score_candidates(reactant_batched, candidate_batched)
        return scores

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict] = None):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "node_in_dim": self.node_in_dim,
                "edge_in_dim": self.edge_in_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
            },
            "extra_meta": extra_meta or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "MolecularReactionGNN":
        checkpoint = torch.load(path, map_location=device)
        cfg = checkpoint["config"]
        model = cls(
            node_in_dim=cfg["node_in_dim"],
            edge_in_dim=cfg["edge_in_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model
