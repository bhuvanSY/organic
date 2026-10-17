"""
Trainer for the Molecular Reaction Graph Neural Network.
Supports:
1. Multi-task learning (Reaction-Center Prediction + Product-Candidate Ranking)
2. Vectorized disjoint-graph batching
3. Mixed precision (AMP) when CUDA is available
4. Checkpointing, validation loops, and metric tracking
"""

from dataclasses import dataclass
import os
import time
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from chemrxn.chemistry.graphs import (
    BatchMolecularGraph,
    MolecularGraph,
    batch_molecular_graphs,
    detect_reaction_centers,
    mol_to_graph,
)
from chemrxn.models.molecular_gnn import MolecularReactionGNN


@dataclass
class GNNReactionBatch:
    bg_reactants: BatchMolecularGraph
    bg_products_pos: BatchMolecularGraph
    bg_products_neg: BatchMolecularGraph
    rc_targets: torch.Tensor  # [total_reactant_atoms]

    def to(self, device: Union[str, torch.device]) -> "GNNReactionBatch":
        return GNNReactionBatch(
            bg_reactants=self.bg_reactants.to(device),
            bg_products_pos=self.bg_products_pos.to(device),
            bg_products_neg=self.bg_products_neg.to(device),
            rc_targets=self.rc_targets.to(device),
        )


def collate_gnn_reaction_batch(
    samples: List[Tuple[str, str, str]],  # (reactants, true_product, neg_product)
) -> Optional[GNNReactionBatch]:
    """
    Collates a list of (reactant_smi, product_smi, neg_smi) tuples into a GNNReactionBatch.
    Skips invalid molecules gracefully.
    """
    r_graphs: List[MolecularGraph] = []
    p_pos_graphs: List[MolecularGraph] = []
    p_neg_graphs: List[MolecularGraph] = []
    rc_labels_list: List[torch.Tensor] = []

    for r_smi, p_smi, neg_smi in samples:
        g_r = mol_to_graph(r_smi)
        g_p = mol_to_graph(p_smi)
        g_neg = mol_to_graph(neg_smi)

        if g_r is None or g_p is None or g_neg is None:
            continue

        # Extract reaction center targets
        centers, _ = detect_reaction_centers(r_smi, p_smi)
        if len(centers) != g_r.num_nodes:
            # Fallback if atom count mismatch
            centers = [0] * g_r.num_nodes

        r_graphs.append(g_r)
        p_pos_graphs.append(g_p)
        p_neg_graphs.append(g_neg)
        rc_labels_list.append(torch.tensor(centers, dtype=torch.float32))

    if not r_graphs:
        return None

    bg_r = batch_molecular_graphs(r_graphs)
    bg_pos = batch_molecular_graphs(p_pos_graphs)
    bg_neg = batch_molecular_graphs(p_neg_graphs)
    rc_targets = torch.cat(rc_labels_list, dim=0)

    return GNNReactionBatch(
        bg_reactants=bg_r,
        bg_products_pos=bg_pos,
        bg_products_neg=bg_neg,
        rc_targets=rc_targets,
    )


class GNNReactionTrainer:
    """
    Manages training and validation for MolecularReactionGNN.
    """
    def __init__(
        self,
        model: MolecularReactionGNN,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        loss_center_weight: float = 0.5,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        grad_clip: float = 1.0,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_center_weight = loss_center_weight
        self.grad_clip = grad_clip

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.use_amp = use_amp and ("cuda" in str(self.device))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.margin_loss = nn.MarginRankingLoss(margin=0.5)

    def train_epoch(
        self,
        train_batches: List[GNNReactionBatch],
    ) -> Dict[str, float]:
        """Runs one full training epoch across batches."""
        self.model.train()
        total_loss = 0.0
        total_rc_loss = 0.0
        total_rank_loss = 0.0
        correct_rank = 0
        total_samples = 0
        total_atoms = 0
        correct_atoms = 0

        for batch in train_batches:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                # 1. Reaction Center Prediction
                rc_logits = self.model.predict_reaction_center_logits(batch.bg_reactants)
                rc_loss = self.bce_loss(rc_logits, batch.rc_targets)

                # 2. Candidate Ranking
                pos_scores = self.model.score_candidates(batch.bg_reactants, batch.bg_products_pos)
                neg_scores = self.model.score_candidates(batch.bg_reactants, batch.bg_products_neg)

                # Target = 1 means pos_scores should be higher than neg_scores by at least margin
                ones_target = torch.ones_like(pos_scores)
                rank_loss = self.margin_loss(pos_scores, neg_scores, ones_target)

                # Combined Multi-Task Loss
                loss = (
                    self.loss_center_weight * rc_loss
                    + (1.0 - self.loss_center_weight) * rank_loss
                )

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            total_rc_loss += float(rc_loss.item())
            total_rank_loss += float(rank_loss.item())

            # Metrics
            correct_rank += int((pos_scores > neg_scores).sum().item())
            total_samples += batch.bg_reactants.num_graphs

            atom_preds = (torch.sigmoid(rc_logits) >= 0.5).float()
            correct_atoms += int((atom_preds == batch.rc_targets).sum().item())
            total_atoms += batch.rc_targets.numel()

        num_b = max(len(train_batches), 1)
        return {
            "loss": total_loss / num_b,
            "rc_loss": total_rc_loss / num_b,
            "rank_loss": total_rank_loss / num_b,
            "rank_accuracy": correct_rank / max(total_samples, 1),
            "rc_atom_accuracy": correct_atoms / max(total_atoms, 1),
        }

    @torch.no_grad()
    def evaluate(
        self,
        val_batches: List[GNNReactionBatch],
    ) -> Dict[str, float]:
        """Runs evaluation over validation batches."""
        self.model.eval()
        total_loss = 0.0
        total_rc_loss = 0.0
        total_rank_loss = 0.0
        correct_rank = 0
        total_samples = 0
        total_atoms = 0
        correct_atoms = 0

        for batch in val_batches:
            batch = batch.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                rc_logits = self.model.predict_reaction_center_logits(batch.bg_reactants)
                rc_loss = self.bce_loss(rc_logits, batch.rc_targets)

                pos_scores = self.model.score_candidates(batch.bg_reactants, batch.bg_products_pos)
                neg_scores = self.model.score_candidates(batch.bg_reactants, batch.bg_products_neg)
                ones_target = torch.ones_like(pos_scores)
                rank_loss = self.margin_loss(pos_scores, neg_scores, ones_target)

                loss = (
                    self.loss_center_weight * rc_loss
                    + (1.0 - self.loss_center_weight) * rank_loss
                )

            total_loss += float(loss.item())
            total_rc_loss += float(rc_loss.item())
            total_rank_loss += float(rank_loss.item())

            correct_rank += int((pos_scores > neg_scores).sum().item())
            total_samples += batch.bg_reactants.num_graphs

            atom_preds = (torch.sigmoid(rc_logits) >= 0.5).float()
            correct_atoms += int((atom_preds == batch.rc_targets).sum().item())
            total_atoms += batch.rc_targets.numel()

        num_b = max(len(val_batches), 1)
        return {
            "val_loss": total_loss / num_b,
            "val_rc_loss": total_rc_loss / num_b,
            "val_rank_loss": total_rank_loss / num_b,
            "val_rank_accuracy": correct_rank / max(total_samples, 1),
            "val_rc_atom_accuracy": correct_atoms / max(total_atoms, 1),
        }
