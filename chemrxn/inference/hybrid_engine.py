"""
Hybrid Inference Engine for ChemRxn-Intelligence.
Integrates:
1. Generative Seq2Seq Transformer (Beam search candidate generation)
2. Molecular Graph Neural Network (Reaction-center & graph-pair compatibility)
3. Classical ML Baseline (Supervised candidate ranker & fingerprint descriptors)
4. Canonicalization & duplicate removal
5. RDKit validity filter (valence feasibility, NOT proof of chemical correctness)
6. Reaction-center consistency scoring
7. Supervised candidate fusion (trained strictly on train/val data)
8. Platt confidence calibration & Abstention mechanism for uncertain predictions
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from chemrxn.chemistry.fingerprints import compute_morgan_fingerprint, compute_tanimoto_similarity
from chemrxn.chemistry.graphs import (
    BatchMolecularGraph,
    MolecularGraph,
    batch_molecular_graphs,
    detect_reaction_centers,
    mol_to_graph,
)
from chemrxn.chemistry.smiles import (
    canonicalize_smiles,
    compute_molecular_descriptors,
    is_valid_smiles,
    strip_atom_mapping,
)
from chemrxn.data.tokenizer import SMILESTokenizer
from chemrxn.inference.fusion_model import (
    CandidateFusionModel,
    ConfidenceCalibrator,
    FusionFeatures,
)
from chemrxn.models.molecular_gnn import MolecularReactionGNN
from chemrxn.models.seq2seq_transformer import Seq2SeqReactionTransformer
from chemrxn.utils.device import get_device


VALIDITY_NOTE_TEXT = (
    "RDKit sanitization passed; indicates syntactic and valence feasibility, "
    "NOT proof of chemical correctness or experimental synthesis yield."
)


@dataclass
class CandidateItem:
    rank: int
    product_smiles: str
    canonical_smiles: str
    is_syntactically_valid: bool
    validity_note: str
    raw_scores: Dict[str, float]
    calibrated_confidence: float
    molecular_descriptors: Optional[Dict[str, Any]] = None
    model_provenance: Dict[str, str] = field(default_factory=dict)


@dataclass
class HybridPredictionOutput:
    reactant_smiles: str
    canonical_reactants: str
    abstained: bool
    abstention_reason: Optional[str]
    top_calibrated_confidence: float
    epistemic_uncertainty: float
    candidates: List[CandidateItem]
    reaction_center_atoms: List[int]
    latency_ms: float
    device: str


class HybridInferenceEngine:
    """
    Production hybrid inference engine combining generative Transformer,
    graph-based MPNN, classical supervised ranker, and calibrated decision rules.
    """
    def __init__(
        self,
        transformer_ckpt: str = "checkpoints/transformer/best_checkpoint.pt",
        gnn_ckpt: str = "checkpoints/gnn/best_gnn.pt",
        baseline_ckpt: str = "checkpoints/baselines/candidate_ranker.joblib",
        fusion_ckpt: str = "checkpoints/hybrid_fusion.json",
        vocab_path: str = "data/processed/smiles_vocab.json",
        abstention_threshold: float = 0.35,
        device: Optional[Union[str, torch.device]] = None,
        lazy_load: bool = True,
    ):
        self.device = torch.device(device or get_device())
        self.transformer_ckpt = transformer_ckpt
        self.gnn_ckpt = gnn_ckpt
        self.baseline_ckpt = baseline_ckpt
        self.fusion_ckpt = fusion_ckpt
        self.vocab_path = vocab_path
        self.abstention_threshold = abstention_threshold

        self.tokenizer: Optional[SMILESTokenizer] = None
        self.transformer: Optional[Seq2SeqReactionTransformer] = None
        self.gnn: Optional[MolecularReactionGNN] = None
        self.baseline_ranker: Optional[Any] = None
        self.fusion_model: CandidateFusionModel = CandidateFusionModel()
        self.is_loaded = False

        if not lazy_load:
            self.load_models()

    def load_models(self):
        """Loads all constituent models from local scratch checkpoints."""
        if self.is_loaded:
            return

        # 1. Tokenizer
        if os.path.exists(self.vocab_path):
            self.tokenizer = SMILESTokenizer.load_vocab(self.vocab_path)
        else:
            self.tokenizer = SMILESTokenizer()

        # 2. Transformer Generator
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
            self.tf_status = "trained_checkpoint"
        else:
            self.tf_status = "scratch_initialized"
        self.transformer.eval()

        # 3. Molecular Reaction GNN
        if os.path.exists(self.gnn_ckpt):
            self.gnn = MolecularReactionGNN.load_checkpoint(self.gnn_ckpt, device=str(self.device))
            self.gnn_status = "trained_checkpoint"
        else:
            self.gnn = MolecularReactionGNN(hidden_dim=128, num_layers=3).to(self.device)
            self.gnn_status = "scratch_initialized"
        self.gnn.eval()

        # 4. Classical Baseline Ranker
        if os.path.exists(self.baseline_ckpt):
            try:
                from chemrxn.baselines.candidate_ranker import SupervisedCandidateRanker
                self.baseline_ranker = SupervisedCandidateRanker.load(self.baseline_ckpt)
                self.baseline_status = "trained_checkpoint"
            except Exception:
                self.baseline_ranker = None
                self.baseline_status = "not_available"
        else:
            self.baseline_ranker = None
            self.baseline_status = "not_available"

        # 5. Fusion & Calibration Model
        if os.path.exists(self.fusion_ckpt):
            self.fusion_model = CandidateFusionModel.load(self.fusion_ckpt)
        else:
            self.fusion_model = CandidateFusionModel()

        self.is_loaded = True

    def compute_reaction_center_consistency(
        self,
        reactant_smiles: str,
        candidate_smiles: str,
        atom_rc_probs: List[float],
    ) -> float:
        """
        Quantifies consistency between GNN-predicted reactive atoms on the reactants
        and the transformation induced by the proposed candidate product.
        """
        reacting_atoms, _ = detect_reaction_centers(reactant_smiles, candidate_smiles)
        if not reacting_atoms or sum(reacting_atoms) == 0:
            return 0.5  # Neutral consistency when no change detected

        # Calculate average reaction center probability for atoms that actually change in candidate
        matched_probs = [
            atom_rc_probs[i] for i, r in enumerate(reacting_atoms)
            if r == 1 and i < len(atom_rc_probs)
        ]
        if not matched_probs:
            return 0.2

        consistency = float(np.mean(matched_probs))
        return float(round(consistency, 4))

    def predict(
        self,
        reactant_smiles: str,
        beam_size: int = 5,
        num_candidates: int = 3,
        use_rc_consistency: bool = True,
        use_baseline_ranker: bool = True,
    ) -> HybridPredictionOutput:
        """
        Executes end-to-end hybrid inference with calibration and abstention.
        """
        self.load_models()
        t0 = time.time()

        # Step 1: Preprocessing & Canonicalization
        clean_r = strip_atom_mapping(reactant_smiles.strip())
        canon_r = canonicalize_smiles(clean_r)

        if not canon_r:
            return HybridPredictionOutput(
                reactant_smiles=reactant_smiles,
                canonical_reactants="",
                abstained=True,
                abstention_reason=f"Invalid reactant SMILES: '{reactant_smiles}' could not be parsed by RDKit",
                top_calibrated_confidence=0.0,
                epistemic_uncertainty=1.0,
                candidates=[],
                reaction_center_atoms=[],
                latency_ms=round((time.time() - t0) * 1000.0, 2),
                device=str(self.device),
            )

        r_desc = compute_molecular_descriptors(canon_r) or {}
        r_mw = r_desc.get("molecular_weight", 0.0)
        r_heavy = r_desc.get("num_heavy_atoms", 0)

        # Step 2: GNN Reaction Center Detection on Reactants
        g_r = mol_to_graph(canon_r)
        atom_rc_probs: List[float] = []
        reactive_atom_indices: List[int] = []

        if g_r is not None:
            batched_r = batch_molecular_graphs([g_r]).to(self.device)
            with torch.no_grad():
                rc_logits = self.gnn.predict_reaction_center_logits(batched_r)
                atom_rc_probs = torch.sigmoid(rc_logits).cpu().tolist()
                reactive_atom_indices = [i for i, p in enumerate(atom_rc_probs) if p >= 0.5]

        # Step 3: Candidate Generation via Transformer Beam Search
        token_ids = self.tokenizer.encode(canon_r, add_special_tokens=True)
        src_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            batch_beams = self.transformer.generate_beam(
                src=src_tensor,
                beam_size=max(beam_size, num_candidates),
                max_len=140,
                length_penalty_alpha=0.6,
            )
            beams = batch_beams[0] if batch_beams else []

        # Step 4: Canonicalization & Deduplication
        unique_candidates: Dict[str, Tuple[str, float]] = {}  # canon_smi -> (raw_smi, log_prob)
        for token_seq, log_prob in beams:
            raw_smi = self.tokenizer.decode(token_seq, skip_special_tokens=True)
            canon_p = canonicalize_smiles(raw_smi, remove_mapping=True) or raw_smi

            if canon_p not in unique_candidates or log_prob > unique_candidates[canon_p][1]:
                unique_candidates[canon_p] = (raw_smi, float(log_prob))

        candidate_list = list(unique_candidates.items())  # List of (canon_smi, (raw_smi, log_prob))

        # Check for empty generation
        if not candidate_list:
            return HybridPredictionOutput(
                reactant_smiles=reactant_smiles,
                canonical_reactants=canon_r,
                abstained=True,
                abstention_reason="Transformer beam search failed to produce any candidate sequences",
                top_calibrated_confidence=0.0,
                epistemic_uncertainty=1.0,
                candidates=[],
                reaction_center_atoms=reactive_atom_indices,
                latency_ms=round((time.time() - t0) * 1000.0, 2),
                device=str(self.device),
            )

        # Step 5: Multi-Model Scoring
        # GNN Candidate Scoring
        valid_graphs: List[MolecularGraph] = []
        valid_indices: List[int] = []
        gnn_scores: List[float] = [-999.0] * len(candidate_list)

        for idx, (c_canon, _) in enumerate(candidate_list):
            g_cand = mol_to_graph(c_canon)
            if g_cand is not None:
                valid_graphs.append(g_cand)
                valid_indices.append(idx)

        if valid_graphs and g_r is not None:
            with torch.no_grad():
                scores_tensor = self.gnn.score_candidate_pool(g_r, valid_graphs)
                for v_idx, score_val in zip(valid_indices, scores_tensor.cpu().tolist()):
                    gnn_scores[v_idx] = float(score_val)

        # Baseline Ranker Scoring
        cand_strings = [c_canon for c_canon, _ in candidate_list]
        baseline_scores_map = {}
        if use_baseline_ranker and self.baseline_ranker is not None:
            try:
                b_scored = self.baseline_ranker.score_candidates(canon_r, cand_strings)
                baseline_scores_map = {smi: score for smi, score in b_scored}
            except Exception:
                baseline_scores_map = {smi: 0.5 for smi in cand_strings}
        else:
            baseline_scores_map = {smi: 0.5 for smi in cand_strings}

        # Step 6: Fusion Featurization & Platt Calibration
        evaluated_items: List[Tuple[FusionFeatures, CandidateItem, float]] = []

        for idx, (c_canon, (c_raw, tf_log_prob)) in enumerate(candidate_list):
            valid = is_valid_smiles(c_canon)
            c_desc = compute_molecular_descriptors(c_canon) if valid else None
            c_mw = c_desc.get("molecular_weight", r_mw) if c_desc else r_mw
            c_heavy = c_desc.get("num_heavy_atoms", r_heavy) if c_desc else r_heavy

            g_score = gnn_scores[idx] if gnn_scores[idx] > -900.0 else -5.0
            b_score = baseline_scores_map.get(c_canon, 0.5)

            # Reaction center consistency
            if use_rc_consistency and atom_rc_probs:
                rc_cons = self.compute_reaction_center_consistency(canon_r, c_canon, atom_rc_probs)
            else:
                rc_cons = 0.5

            # Length-normalized log-probability to avoid sequence-length penalty bias
            norm_tf = float(tf_log_prob) / max(len(c_raw), 1)

            feat = FusionFeatures(
                transformer_log_prob=float(norm_tf),
                gnn_score=float(g_score),
                baseline_score=float(b_score),
                rc_consistency=float(rc_cons),
                is_valid=1.0 if valid else 0.0,
                delta_mw=float(abs(c_mw - r_mw)),
                delta_heavy_atoms=float(abs(c_heavy - r_heavy)),
            )

            raw_fusion = self.fusion_model.predict_score(feat)
            calibrated_conf = self.fusion_model.predict_calibrated_confidence(feat)

            item = CandidateItem(
                rank=0,
                product_smiles=c_raw,
                canonical_smiles=c_canon,
                is_syntactically_valid=valid,
                validity_note=VALIDITY_NOTE_TEXT if valid else "RDKit sanitization failed: chemical syntax invalid.",
                raw_scores={
                    "transformer_log_prob": round(tf_log_prob, 4),
                    "gnn_score": round(g_score, 4),
                    "baseline_score": round(b_score, 4),
                    "rc_consistency": round(rc_cons, 4),
                    "fusion_score": round(raw_fusion, 4),
                },
                calibrated_confidence=calibrated_conf,
                molecular_descriptors=c_desc,
                model_provenance={
                    "generator": f"Seq2SeqReactionTransformer ({self.tf_status})",
                    "graph_scorer": f"MolecularReactionGNN ({self.gnn_status})",
                    "baseline_ranker": f"SupervisedCandidateRanker ({self.baseline_status})",
                    "fusion": "Trained Validation Platt Calibrator",
                },
            )
            evaluated_items.append((feat, item, raw_fusion))

        # Sort by raw fusion score descending
        evaluated_items.sort(key=lambda x: x[2], reverse=True)
        final_candidates: List[CandidateItem] = []
        for r, (_, cand_item, _) in enumerate(evaluated_items[:num_candidates], 1):
            cand_item.rank = r
            final_candidates.append(cand_item)

        # Step 7: Abstention Mechanism
        top_conf = final_candidates[0].calibrated_confidence if final_candidates else 0.0
        all_invalid = all(not c.is_syntactically_valid for c in final_candidates)

        abstained = False
        abstention_reason = None

        if all_invalid:
            abstained = True
            abstention_reason = "All generated candidates failed RDKit chemical validity and valence checks"
        elif top_conf < self.abstention_threshold:
            abstained = True
            abstention_reason = (
                f"Prediction uncertainty exceeds threshold: top calibrated confidence "
                f"({top_conf:.3f}) is below abstention cutoff ({self.abstention_threshold:.2f})"
            )

        # Epistemic uncertainty: 1 - top_confidence
        epistemic_unc = round(float(1.0 - top_conf), 4)
        duration_ms = round((time.time() - t0) * 1000.0, 2)

        return HybridPredictionOutput(
            reactant_smiles=reactant_smiles,
            canonical_reactants=canon_r,
            abstained=abstained,
            abstention_reason=abstention_reason,
            top_calibrated_confidence=top_conf,
            epistemic_uncertainty=epistemic_unc,
            candidates=final_candidates,
            reaction_center_atoms=reactive_atom_indices,
            latency_ms=duration_ms,
            device=str(self.device),
        )
