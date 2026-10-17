"""
Reaction-Class Classification Baseline.
Predicts transformation class (e.g., Acylation, Reduction, C-C Coupling) from reactant features.
IMPORTANT: This model classifies reaction types; it does NOT generate product SMILES.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np

from chemrxn.baselines.taxonomy import CLASS_TO_ID, ID_TO_CLASS, REACTION_CLASSES


class ReactionClassClassifier:
    """
    Supervised classifier for reaction category prediction using XGBoost or Scikit-Learn.

    Architectural Scope:
        Task: Maps Reactants (+ Reagents) -> Transformation Category (0..9).
        Distinction: Categorical classification only. Does NOT synthesize or output product SMILES.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        seed: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
    ):
        self.model_type = model_type.lower()
        self.seed = seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.classes_ = REACTION_CLASSES
        self.model = self._create_model()
        self.is_trained = False

    def _create_model(self):
        if self.model_type == "xgboost":
            try:
                import xgboost as xgb
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                return xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=0.1,
                    random_state=self.seed,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    device=device,
                    n_jobs=-1 if device == "cpu" else None,
                )
            except Exception:
                self.model_type = "hist_gb"

        if self.model_type == "hist_gb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=2,
                random_state=self.seed,
            )

        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.seed,
                n_jobs=-1,
            )

        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000, random_state=self.seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ReactionClassClassifier":
        """Fits classifier on reaction feature matrix and integer class labels."""
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns integer class predictions."""
        if not self.is_trained:
            raise RuntimeError("Model is untrained.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class probability distributions."""
        if not self.is_trained:
            raise RuntimeError("Model is untrained.")
        return self.model.predict_proba(X)

    def predict_named_classes(self, X: np.ndarray) -> List[str]:
        """Returns human-readable reaction class names."""
        preds = self.predict(X)
        return [ID_TO_CLASS.get(int(idx), "Unknown") for idx in preds]

    def save(self, filepath: Union[str, Path]) -> None:
        """Persists trained classifier artifact to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "model_type": self.model_type,
            "seed": self.seed,
            "classes_": self.classes_,
            "is_trained": self.is_trained,
            "purpose": "Reaction-class categorical prediction (not product generation)",
        }
        joblib.dump(artifact, p)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ReactionClassClassifier":
        """Loads persisted model from disk."""
        artifact = joblib.load(filepath)
        instance = cls(model_type=artifact["model_type"], seed=artifact["seed"])
        instance.model = artifact["model"]
        instance.classes_ = artifact["classes_"]
        instance.is_trained = artifact["is_trained"]
        return instance
