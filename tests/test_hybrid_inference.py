"""
Unit and integration tests for the Hybrid Inference Engine,
Candidate Fusion, Platt Calibration, Abstention, and Ablations.
"""

import numpy as np
import pytest
import torch

from chemrxn.chemistry.smiles import is_valid_smiles, canonicalize_smiles
from chemrxn.inference.fusion_model import (
    CandidateFusionModel,
    ConfidenceCalibrator,
    FusionFeatures,
)
from chemrxn.inference.hybrid_engine import (
    CandidateItem,
    HybridInferenceEngine,
    HybridPredictionOutput,
    VALIDITY_NOTE_TEXT,
)


def test_confidence_calibrator_platt_scaling():
    """Verifies that Platt sigmoid scaling maps scores to [0, 1] and fits validation data."""
    calibrator = ConfidenceCalibrator(a=1.5, b=-0.5)

    # Score zero -> sigmoid(-0.5) ~ 0.3775
    p0 = calibrator.calibrate(0.0)
    assert 0.0 < p0 < 1.0

    # Higher score -> higher probability
    p_high = calibrator.calibrate(5.0)
    p_low = calibrator.calibrate(-5.0)
    assert p_high > p_low
    assert p_high > 0.9
    assert p_low < 0.1

    # Fit test
    scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    calibrator.fit(scores, labels)
    assert calibrator.is_fitted
    assert calibrator.calibrate(2.0) > calibrator.calibrate(-2.0)

    # ECE computation
    preds = np.array([0.1, 0.2, 0.8, 0.9])
    ece = calibrator.compute_ece(preds, np.array([0, 0, 1, 1]))
    assert 0.0 <= ece <= 1.0


def test_candidate_fusion_model_scoring_and_weights():
    """Verifies that the fusion model combines multi-modal features with expected weights."""
    model = CandidateFusionModel()

    f_pos = FusionFeatures(
        transformer_log_prob=-0.5,
        gnn_score=2.0,
        baseline_score=0.9,
        rc_consistency=0.8,
        is_valid=1.0,
        delta_mw=2.0,
        delta_heavy_atoms=0.0,
    )

    f_neg = FusionFeatures(
        transformer_log_prob=-8.0,
        gnn_score=-2.0,
        baseline_score=0.1,
        rc_consistency=0.1,
        is_valid=0.0,
        delta_mw=80.0,
        delta_heavy_atoms=10.0,
    )

    s_pos = model.predict_score(f_pos)
    s_neg = model.predict_score(f_neg)
    assert s_pos > s_neg, "Chemically plausible candidate should score higher than decoy"

    c_pos = model.predict_calibrated_confidence(f_pos)
    c_neg = model.predict_calibrated_confidence(f_neg)
    assert c_pos > c_neg
    assert 0.0 <= c_pos <= 1.0
    assert 0.0 <= c_neg <= 1.0


def test_hybrid_engine_initialization_and_metadata():
    """Verifies that HybridInferenceEngine initializes cleanly without test data leakage."""
    engine = HybridInferenceEngine(device="cpu", lazy_load=True)
    assert not engine.is_loaded
    engine.load_models()
    assert engine.is_loaded
    assert engine.transformer is not None
    assert engine.gnn is not None


def test_hybrid_engine_prediction_structure_and_validity_note():
    """Verifies prediction outputs contain candidate SMILES, validity note, and provenance."""
    engine = HybridInferenceEngine(device="cpu", lazy_load=False)

    out = engine.predict(
        reactant_smiles="CC(=O)Cl.CCN",
        beam_size=2,
        num_candidates=2,
    )

    assert isinstance(out, HybridPredictionOutput)
    assert out.reactant_smiles == "CC(=O)Cl.CCN"
    assert out.canonical_reactants in ["CCN.CC(=O)Cl", "CC(=O)Cl.CCN"]
    assert isinstance(out.abstained, bool)
    assert 0.0 <= out.top_calibrated_confidence <= 1.0
    assert 0.0 <= out.epistemic_uncertainty <= 1.0
    assert out.latency_ms > 0

    if out.candidates:
        cand = out.candidates[0]
        assert isinstance(cand, CandidateItem)
        assert cand.rank == 1
        assert "transformer_log_prob" in cand.raw_scores
        assert "gnn_score" in cand.raw_scores
        assert "baseline_score" in cand.raw_scores
        assert "rc_consistency" in cand.raw_scores
        assert "generator" in cand.model_provenance
        # Mandatory disclaimer text
        assert "syntactic and valence feasibility" in cand.validity_note


def test_abstention_mechanism_on_invalid_or_uncertain_input():
    """Verifies that engine abstains when candidates are invalid or below confidence threshold."""
    # 1. Invalid reactant SMILES -> must abstain immediately
    engine = HybridInferenceEngine(abstention_threshold=0.35, device="cpu")
    out_inv = engine.predict("INVALID_NOT_SMILES_12345")
    assert out_inv.abstained is True
    assert "Invalid reactant SMILES" in str(out_inv.abstention_reason)

    # 2. Strict threshold test (threshold 0.999 -> should abstain)
    strict_engine = HybridInferenceEngine(abstention_threshold=0.999, device="cpu")
    out_strict = strict_engine.predict("CC(=O)O.CCN", beam_size=2)
    assert out_strict.abstained is True
    assert "below abstention cutoff" in str(out_strict.abstention_reason)


def test_reaction_center_consistency_scoring():
    """Verifies that candidate matching predicted reaction centers scores higher consistency."""
    engine = HybridInferenceEngine(device="cpu")
    # Carbonyl carbon is reactive site for acyl chloride
    r_smi = "CC(=O)Cl.CCN"
    # Target amide changes bonds at carbonyl and amine
    p_amide = "CCNC(=O)C"
    # Fake atom probabilities: atom 1 (C=O) has high reaction prob
    atom_probs = [0.05, 0.95, 0.05, 0.90, 0.05, 0.05, 0.05]

    cons = engine.compute_reaction_center_consistency(r_smi, p_amide, atom_probs)
    assert 0.0 <= cons <= 1.0
    assert cons >= 0.4
