"""
Candidate product generation and graph-conditioned re-ranking pipeline.
Interfaces between the generative Seq2Seq Transformer (candidate generator)
and the Molecular Reaction GNN (candidate scorer).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from chemrxn.chemistry.graphs import MolecularGraph, mol_to_graph
from chemrxn.chemistry.smiles import canonicalize_smiles, is_valid_smiles
from chemrxn.data.tokenizer import SMILESTokenizer
from chemrxn.models.molecular_gnn import MolecularReactionGNN
from chemrxn.models.seq2seq_transformer import Seq2SeqReactionTransformer


@dataclass
class RankedCandidate:
    smiles: str
    is_valid: bool
    transformer_score: float     # Log-probability from Transformer
    gnn_score: float             # Compatibility score from GNN
    hybrid_score: float          # Weighted combination
    rank: int


@dataclass
class CandidateRankingResult:
    reactant_smiles: str
    target_product_smiles: Optional[str]
    candidates: List[RankedCandidate]
    top1_match_transformer: bool
    top1_match_gnn: bool
    top1_match_hybrid: bool


class HybridCandidateRanker:
    """
    Orchestrates candidate generation from Seq2Seq Transformer beam search
    and structural re-ranking using the Molecular Reaction GNN.
    """
    def __init__(
        self,
        transformer_model: Seq2SeqReactionTransformer,
        gnn_model: MolecularReactionGNN,
        tokenizer: SMILESTokenizer,
        device: Union[str, torch.device] = "cpu",
        hybrid_lambda: float = 0.5,
    ):
        self.transformer = transformer_model.to(device)
        self.gnn = gnn_model.to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.hybrid_lambda = hybrid_lambda

        self.transformer.eval()
        self.gnn.eval()

    def generate_candidates_transformer(
        self,
        reactant_smiles: str,
        beam_size: int = 5,
        max_len: int = 140,
    ) -> List[Tuple[str, float]]:
        """
        Generates top-K candidate products with sequence log-probabilities using beam search.
        Returns:
            List of (candidate_smiles, log_prob)
        """
        token_ids = self.tokenizer.encode(reactant_smiles, add_special_tokens=True)
        src_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            batch_beams = self.transformer.generate_beam(
                src=src_tensor,
                beam_size=beam_size,
                max_len=max_len,
                length_penalty_alpha=0.6,
            )
            beams = batch_beams[0] if batch_beams else []

        candidates: List[Tuple[str, float]] = []
        for token_seq, log_prob in beams:
            # Decode tokens back to SMILES string
            smi = self.tokenizer.decode(token_seq, skip_special_tokens=True)
            canon = canonicalize_smiles(smi, remove_mapping=True) or smi
            candidates.append((canon, float(log_prob)))

        return candidates

    def score_candidates_with_gnn(
        self,
        reactant_smiles: str,
        candidate_smiles_list: List[str],
    ) -> List[float]:
        """
        Evaluates structural compatibility scores for each candidate using the GNN.
        Invalid/unparseable SMILES receive a large negative penalty score (-999.0).
        """
        reactant_graph = mol_to_graph(reactant_smiles)
        if reactant_graph is None:
            return [-999.0] * len(candidate_smiles_list)

        valid_graphs: List[MolecularGraph] = []
        valid_indices: List[int] = []

        scores = [-999.0] * len(candidate_smiles_list)

        for i, c_smi in enumerate(candidate_smiles_list):
            g = mol_to_graph(c_smi)
            if g is not None:
                valid_graphs.append(g)
                valid_indices.append(i)

        if valid_graphs:
            with torch.no_grad():
                gnn_scores = self.gnn.score_candidate_pool(reactant_graph, valid_graphs)
                for idx, score_val in zip(valid_indices, gnn_scores.cpu().tolist()):
                    scores[idx] = float(score_val)

        return scores

    def rank_candidates(
        self,
        reactant_smiles: str,
        candidates_with_scores: List[Tuple[str, float]],
        target_product_smiles: Optional[str] = None,
        hybrid_lambda: Optional[float] = None,
    ) -> CandidateRankingResult:
        """
        Combines Transformer generation probabilities with GNN structural scores.
        """
        lmbda = self.hybrid_lambda if hybrid_lambda is None else hybrid_lambda
        candidate_strings = [c[0] for c in candidates_with_scores]
        tf_scores = [c[1] for c in candidates_with_scores]

        # Compute GNN scores
        gnn_scores = self.score_candidates_with_gnn(reactant_smiles, candidate_strings)

        # Normalize GNN scores to comparable dynamic range with min-max over finite values
        finite_gnn = [s for s in gnn_scores if s > -900.0]
        if finite_gnn:
            min_g, max_g = min(finite_gnn), max(finite_gnn)
            range_g = (max_g - min_g) if (max_g - min_g) > 1e-6 else 1.0
            norm_gnn = [((s - min_g) / range_g) if s > -900.0 else -10.0 for s in gnn_scores]
        else:
            norm_gnn = [-10.0] * len(gnn_scores)

        # Normalize Transformer scores
        min_tf, max_tf = min(tf_scores), max(tf_scores)
        range_tf = (max_tf - min_tf) if (max_tf - min_tf) > 1e-6 else 1.0
        norm_tf = [((s - min_tf) / range_tf) for s in tf_scores]

        # Assemble candidate items
        candidate_items: List[RankedCandidate] = []
        for smi, tf_raw, n_tf, gnn_raw, n_gnn in zip(
            candidate_strings, tf_scores, norm_tf, gnn_scores, norm_gnn
        ):
            hybrid = lmbda * n_tf + (1.0 - lmbda) * n_gnn
            valid = is_valid_smiles(smi)
            candidate_items.append(
                RankedCandidate(
                    smiles=smi,
                    is_valid=valid,
                    transformer_score=tf_raw,
                    gnn_score=gnn_raw,
                    hybrid_score=hybrid,
                    rank=0,
                )
            )

        # Sort by hybrid score descending
        sorted_by_hybrid = sorted(candidate_items, key=lambda c: c.hybrid_score, reverse=True)
        for r, c in enumerate(sorted_by_hybrid, 1):
            c.rank = r

        # Evaluate Top-1 matches against target product (if provided)
        top1_tf = False
        top1_gnn = False
        top1_hybrid = False

        if target_product_smiles:
            target_canon = canonicalize_smiles(target_product_smiles, remove_mapping=True) or target_product_smiles

            # Top-1 by Transformer
            sorted_tf = sorted(candidate_items, key=lambda c: c.transformer_score, reverse=True)
            top1_tf = (sorted_tf[0].smiles == target_canon) if sorted_tf else False

            # Top-1 by GNN
            sorted_gnn = sorted(candidate_items, key=lambda c: c.gnn_score, reverse=True)
            top1_gnn = (sorted_gnn[0].smiles == target_canon) if sorted_gnn else False

            # Top-1 by Hybrid
            top1_hybrid = (sorted_by_hybrid[0].smiles == target_canon) if sorted_by_hybrid else False

        return CandidateRankingResult(
            reactant_smiles=reactant_smiles,
            target_product_smiles=target_product_smiles,
            candidates=sorted_by_hybrid,
            top1_match_transformer=top1_tf,
            top1_match_gnn=top1_gnn,
            top1_match_hybrid=top1_hybrid,
        )
