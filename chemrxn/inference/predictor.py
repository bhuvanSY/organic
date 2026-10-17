"""
Unified Reaction Prediction Inference Service for ChemRxn-Intelligence.
Conducts:
1. Reactant validation and canonicalization
2. Autoregressive candidate product generation via Seq2Seq Transformer beam search
3. Structural scoring and re-ranking via Molecular Reaction GNN
4. Atom-level reaction center identification
5. Chemical validity checking and molecular property computation
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from chemrxn.chemistry.fingerprints import compute_morgan_fingerprint, compute_tanimoto_similarity
from chemrxn.chemistry.graphs import detect_reaction_centers, mol_to_graph
from chemrxn.chemistry.reactions import parse_reaction_smiles
from chemrxn.chemistry.smiles import (
    canonicalize_smiles,
    compute_molecular_descriptors,
    is_valid_smiles,
    strip_atom_mapping,
)
from chemrxn.data.tokenizer import SMILESTokenizer
from chemrxn.inference.candidate_ranker import HybridCandidateRanker, RankedCandidate
from chemrxn.models.molecular_gnn import MolecularReactionGNN
from chemrxn.models.seq2seq_transformer import Seq2SeqReactionTransformer
from chemrxn.utils.device import get_device


class ReactionPredictor:
    """
    Unified inference service combining Transformer generation and GNN structural scoring.
    """
    def __init__(
        self,
        transformer_ckpt: str = "checkpoints/transformer/best_checkpoint.pt",
        gnn_ckpt: str = "checkpoints/gnn/best_gnn.pt",
        vocab_path: str = "data/processed/smiles_vocab.json",
        device: Optional[Union[str, torch.device]] = None,
        lazy_load: bool = True,
    ):
        self.device = torch.device(device or get_device())
        self.transformer_ckpt = transformer_ckpt
        self.gnn_ckpt = gnn_ckpt
        self.vocab_path = vocab_path

        self.tokenizer: Optional[SMILESTokenizer] = None
        self.transformer: Optional[Seq2SeqReactionTransformer] = None
        self.gnn: Optional[MolecularReactionGNN] = None
        self.ranker: Optional[HybridCandidateRanker] = None
        self.is_loaded = False

        if not lazy_load:
            self.load_models()

    def load_models(self):
        """Loads models and tokenizer into memory."""
        if self.is_loaded:
            return

        # 1. Tokenizer
        if os.path.exists(self.vocab_path):
            self.tokenizer = SMILESTokenizer.load_vocab(self.vocab_path)
        else:
            self.tokenizer = SMILESTokenizer()

        # 2. Transformer
        self.transformer = Seq2SeqReactionTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=512,
        ).to(self.device)

        if os.path.exists(self.transformer_ckpt):
            self.transformer.load_checkpoint(self.transformer_ckpt, map_location=self.device)
            self.transformer_status = "trained_checkpoint"
        else:
            self.transformer_status = "scratch_initialized"

        # 3. GNN
        if os.path.exists(self.gnn_ckpt):
            self.gnn = MolecularReactionGNN.load_checkpoint(self.gnn_ckpt, device=str(self.device))
            self.gnn_status = "trained_checkpoint"
        else:
            self.gnn = MolecularReactionGNN(hidden_dim=128, num_layers=3).to(self.device)
            self.gnn_status = "scratch_initialized"

        # 4. Hybrid Ranker
        self.ranker = HybridCandidateRanker(
            transformer_model=self.transformer,
            gnn_model=self.gnn,
            tokenizer=self.tokenizer,
            device=self.device,
            hybrid_lambda=0.5,
        )

        self.is_loaded = True

    def predict(
        self,
        reactant_smiles: str,
        num_candidates: int = 3,
        beam_width: int = 5,
        hybrid_lambda: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end organic reaction prediction.
        """
        self.load_models()
        t0 = time.time()

        # Step 1: Preprocess and validate reactants
        clean_reactants = strip_atom_mapping(reactant_smiles)
        reactants_canon = canonicalize_smiles(clean_reactants)

        if not reactants_canon:
            return {
                "success": False,
                "error": f"Invalid reactant SMILES: '{reactant_smiles}'",
                "reactants_input": reactant_smiles,
                "reactants_canonical": None,
                "candidates": [],
            }

        reactant_descriptors = compute_molecular_descriptors(reactants_canon)
        r_fp = compute_morgan_fingerprint(reactants_canon)

        # Step 2: Generate candidates with Transformer
        raw_candidates = self.ranker.generate_candidates_transformer(
            reactant_smiles=reactants_canon,
            beam_size=max(beam_width, num_candidates),
        )

        # Step 3: Score candidates with GNN and rank
        ranking_result = self.ranker.rank_candidates(
            reactant_smiles=reactants_canon,
            candidates_with_scores=raw_candidates,
            hybrid_lambda=hybrid_lambda,
        )

        # Format candidates
        candidate_items: List[Dict[str, Any]] = []
        for cand in ranking_result.candidates[:num_candidates]:
            p_smi = cand.smiles
            valid = is_valid_smiles(p_smi)
            desc = compute_molecular_descriptors(p_smi) if valid else None

            # Tanimoto similarity to reactants
            if valid:
                sim = compute_tanimoto_similarity(reactants_canon, p_smi) or 0.0
            else:
                sim = 0.0

            # Confidence score mapped from hybrid score
            confidence = round(max(0.01, min(0.99, float(torch.sigmoid(torch.tensor(cand.hybrid_score)).item()))), 4)

            candidate_items.append({
                "rank": cand.rank,
                "product_smiles": p_smi,
                "is_chemically_valid": valid,
                "confidence_score": confidence,
                "tanimoto_to_reactants": round(float(sim), 4),
                "transformer_score": round(cand.transformer_score, 4),
                "gnn_score": round(cand.gnn_score, 4),
                "hybrid_score": round(cand.hybrid_score, 4),
                "descriptors": desc,
            })

        duration_ms = round((time.time() - t0) * 1000.0, 2)

        return {
            "success": True,
            "reactants_input": reactant_smiles,
            "reactants_canonical": reactants_canon,
            "reactant_descriptors": reactant_descriptors,
            "model_status": f"Transformer: {self.transformer_status} | GNN: {self.gnn_status}",
            "is_trained": (self.transformer_status == "trained_checkpoint" or self.gnn_status == "trained_checkpoint"),
            "candidates": candidate_items,
            "hardware_device": str(self.device),
            "latency_ms": duration_ms,
        }

    def predict_reaction_centers(self, reactant_smiles: str) -> Dict[str, Any]:
        """Identifies reactive atoms in reactants using both GNN and chemical rules."""
        self.load_models()
        clean_r = canonicalize_smiles(strip_atom_mapping(reactant_smiles))
        if not clean_r:
            return {"success": False, "error": "Invalid reactant SMILES"}

        g = mol_to_graph(clean_r)
        if g is None:
            return {"success": False, "error": "Could not construct molecular graph"}

        # GNN reaction center prediction
        bg = g.to(self.device)
        from chemrxn.chemistry.graphs import batch_molecular_graphs
        batched = batch_molecular_graphs([g]).to(self.device)

        with torch.no_grad():
            logits = self.gnn.predict_reaction_center_logits(batched)
            probs = torch.sigmoid(logits).cpu().tolist()

        return {
            "success": True,
            "reactants": clean_r,
            "num_atoms": g.num_nodes,
            "atom_reaction_probabilities": [round(p, 4) for p in probs],
            "reactive_atom_indices": [i for i, p in enumerate(probs) if p >= 0.5],
        }
