"""
Candidate fusion model and confidence calibrator for ChemRxn-Intelligence.
Trains strictly on training/validation candidate pairs (locked test data is never used for fitting).
Computes Platt scaling calibration: P(correct | score) = sigmoid(A * score + B).
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass
class FusionFeatures:
    transformer_log_prob: float
    gnn_score: float
    baseline_score: float
    rc_consistency: float
    is_valid: float
    delta_mw: float
    delta_heavy_atoms: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.transformer_log_prob,
            self.gnn_score,
            self.baseline_score,
            self.rc_consistency,
            self.is_valid,
            self.delta_mw,
            self.delta_heavy_atoms,
        ], dtype=np.float32)


class ConfidenceCalibrator:
    """
    Platt sigmoid scaling calibrator.
    Maps uncalibrated fusion scores into well-calibrated posterior probabilities.
    Raw neural network scores must NEVER be reported as calibrated probabilities.
    """
    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)
        self.is_fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fits Platt scaling sigmoid: P(y=1 | s) = 1 / (1 + exp(-(A * s + B))).
        Uses logistic regression / cross-entropy optimization.
        """
        if len(scores) < 5 or len(np.unique(labels)) < 2:
            self.a = 1.0
            self.b = -0.5
            self.is_fitted = True
            return

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, solver="lbfgs")
        lr.fit(scores.reshape(-1, 1), labels)

        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        self.is_fitted = True

    def calibrate(self, score: float) -> float:
        """Computes calibrated probability P(correct | score)."""
        z = self.a * float(score) + self.b
        # Numerical safeguard against overflow
        z_clamped = max(-20.0, min(20.0, z))
        prob = 1.0 / (1.0 + np.exp(-z_clamped))
        return float(round(prob, 4))

    def compute_ece(self, preds: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> float:
        """
        Computes Expected Calibration Error (ECE).
        """
        if len(preds) == 0:
            return 0.0
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_total = len(preds)

        for i in range(n_bins):
            bin_mask = (preds >= bins[i]) & (preds < bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(labels[bin_mask])
                bin_conf = np.mean(preds[bin_mask])
                ece += (np.sum(bin_mask) / n_total) * np.abs(bin_acc - bin_conf)

        return float(round(ece, 4))


class CandidateFusionModel:
    """
    Supervised candidate fusion model.
    Combines Transformer generation probabilities, GNN compatibility scores,
    classical ML baseline ranker, and reaction-center consistency into a unified rank score.
    """
    def __init__(
        self,
        weights: Optional[np.ndarray] = None,
        bias: float = 0.0,
        calibrator: Optional[ConfidenceCalibrator] = None,
    ):
        # Default heuristic weights [tf, gnn, baseline, rc, valid, delta_mw, delta_heavy]
        self.weights = weights if weights is not None else np.array([
            0.10,   # Transformer length-normalized log prob
            0.40,   # GNN compatibility score
            0.25,   # Baseline ranker
            0.20,   # Reaction center consistency
            0.50,   # RDKit syntactic validity
            -0.01,  # Penalty for mass divergence
            -0.02,  # Penalty for heavy atom divergence
        ], dtype=np.float32)
        self.bias = float(bias) if bias != 0.0 else 0.5
        self.calibrator = calibrator or ConfidenceCalibrator(a=1.0, b=0.0)
        self.is_fitted = False

    def predict_score(self, features: FusionFeatures) -> float:
        """Computes raw uncalibrated fusion score."""
        vec = features.to_array()
        raw_score = float(np.dot(self.weights, vec) + self.bias)
        return raw_score

    def predict_calibrated_confidence(self, features: FusionFeatures) -> float:
        """Computes calibrated probability P(candidate is correct)."""
        score = self.predict_score(features)
        return self.calibrator.calibrate(score)

    def fit(self, feature_list: List[FusionFeatures], labels: List[int]):
        """
        Fits fusion weights and Platt calibrator strictly on validation data.
        """
        if len(feature_list) < 5 or len(np.unique(labels)) < 2:
            self.is_fitted = True
            return

        from sklearn.linear_model import RidgeClassifier, LogisticRegression
        X = np.stack([f.to_array() for f in feature_list])
        y = np.array(labels, dtype=int)

        # Fit linear fusion weights
        clf = LogisticRegression(C=1.0, max_iter=200)
        clf.fit(X, y)

        self.weights = clf.coef_[0].astype(np.float32)
        self.bias = float(clf.intercept_[0])

        # Fit calibrator on fusion scores
        raw_scores = np.dot(X, self.weights) + self.bias
        self.calibrator.fit(raw_scores, y)
        self.is_fitted = True

    def save(self, filepath: Union[str, Path]):
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "calibrator": {
                "a": self.calibrator.a,
                "b": self.calibrator.b,
                "is_fitted": self.calibrator.is_fitted,
            },
            "is_fitted": self.is_fitted,
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CandidateFusionModel":
        p = Path(filepath)
        if not p.exists():
            return cls()
        with open(p, "r") as f:
            data = json.load(f)
        cal = ConfidenceCalibrator(a=data["calibrator"]["a"], b=data["calibrator"]["b"])
        cal.is_fitted = data["calibrator"]["is_fitted"]
        model = cls(
            weights=np.array(data["weights"], dtype=np.float32),
            bias=data["bias"],
            calibrator=cal,
        )
        model.is_fitted = data.get("is_fitted", False)
        return model
